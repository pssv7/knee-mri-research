"""Generate one self-contained Small/224 nested-outer0 Colab notebook.

The original frozen runner is adapted with checked, minimal substitutions;
historical source, receipts, labels and split artifacts remain untouched.
"""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import colab_small_runtime

RESTORE_PATHS = '''# Restore missing settings without overwriting your custom paths.
for name, default in {
    'DRIVE_CACHE': '/content/drive/MyDrive/kagglehub',
    'SIDE_FILE': '',
    'DRIVE_OUT': '/content/drive/MyDrive/knee_runs/convnext_small_224_outer0_v1',
    'LOCAL_INPUTS': '/content/knee_small_inputs',
}.items():
    globals().setdefault(name, default)
'''

SMOKE = r'''
def smoke_model(module, train_loader, out, initial_hash):
    import torch, timm, lightning.pytorch as pl
    from torch.utils.data import DataLoader, Subset
    from src.model.dataset import collate_studies
    from src.model.net import KneeNet
    # Exercise maximum series count on training studies only, never the outer set.
    source = train_loader.dataset
    chosen = []
    for i in range(len(source)):
        if source[i]['volumes'].shape[0] == 5:
            chosen.append(i)
            if len(chosen) == 2:
                break
    if len(chosen) != 2:
        raise RuntimeError('Smoke needs two full five-series training studies')
    loader = DataLoader(Subset(source, chosen * 8), batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_studies)
    class Finite(pl.Callback):
        def __init__(self):
            self.losses = []
            self.steps = 0
        def on_before_optimizer_step(self, trainer, module, optimizer):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Nonfinite or missing gradients')
            self.steps += 1
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if not torch.isfinite(loss).all():
                raise RuntimeError('Nonfinite smoke loss')
            self.losses.append(float(loss.detach().cpu()))
    finite = Finite()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer = pl.Trainer(max_epochs=1, max_steps=2, accelerator='gpu', devices=1,
        precision='16-mixed', accumulate_grad_batches=8, callbacks=[finite],
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        limit_val_batches=0, num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader)
    torch.cuda.synchronize()
    if finite.steps != 2 or len(finite.losses) != 16:
        raise RuntimeError('Smoke did not complete exactly two optimizer steps')
    if not all(torch.isfinite(p).all() for p in module.parameters()):
        raise RuntimeError('Nonfinite updated parameters')
    training_peak = torch.cuda.max_memory_allocated()/2**30
    # Round-trip weights through a file and a fresh model, then compare outputs.
    model = module.model.to('cuda').eval()
    batch = next(iter(loader))
    args = [batch[k].cuda() for k in ('volumes', 'meta', 'mask')]
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        before = model(*args).float().cpu()
    temporary = out / 'transient_roundtrip.pt'
    torch.save(model.state_dict(), temporary)
    from src.model import net
    backbone = net.build_backbone('convnext_small.fb_in22k_ft_in1k', pretrained=False)
    restored = KneeNet(backbone, backbone.num_features,
                       **{k:v for k,v in net.backbone_defaults('convnext_small.fb_in22k_ft_in1k').items()
                          if k in ('input_mean', 'input_std')})
    restored.load_state_dict(torch.load(temporary, map_location='cpu', weights_only=True), strict=True)
    model.cpu()
    restored.to('cuda').eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        after = restored(*args).float().cpu()
    if not torch.isfinite(after).all() or not torch.allclose(before, after, atol=1e-5, rtol=1e-5):
        raise RuntimeError('Checkpoint round-trip changed predictions')
    temporary.unlink()
    receipt = {'status':'passed', 'optimizer_steps':finite.steps, 'microbatches':len(finite.losses),
        'initial_weights_sha256':initial_hash, 'finite_losses':finite.losses,
        'checkpoint_roundtrip_max_abs':float((before-after).abs().max()),
        'training_peak_allocated_gib':training_peak,
        'including_reload_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'elapsed_seconds':time.monotonic()-started, 'weights_retained':False,
        'runtime_versions':{'torch':torch.__version__, 'timm':timm.__version__,
            'lightning':pl.__version__, 'gpu':torch.cuda.get_device_name(),
            'devices':1, 'precision':'16-mixed'}}
    write_json_new(out / 'smoke_receipt.json', receipt)
    return receipt
'''


def small_source(baseline):
    """Fail closed if the frozen runner no longer has the expected shape."""
    source = baseline.split("if __name__ == '__main__':", 1)[0]
    edits = {
        "'backbone': 'convnext_tiny.fb_in22k_ft_in1k'": "'backbone': 'convnext_small.fb_in22k_ft_in1k'",
        'def run(design, input_config, out, execute=False):':
            'def run(design, input_config, out, execute=False, smoke=False):',
        '    checkpoint = pl.callbacks.ModelCheckpoint(': 
            '    if smoke:\n        return smoke_model(module, train_loader, out, initial.hexdigest())\n'
            '    checkpoint = pl.callbacks.ModelCheckpoint(',
        "'arm': 'control' if spec.get('sampling_policy', 'legacy') == 'legacy' else 'candidate'":
            "'arm': 'candidate'",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f'Frozen runner adapter needs review: {old}')
        source = source.replace(old, new)
    # Save the last training state too: preserve recoverable optimizer state on Drive.
    old = "monitor='val_macro', mode='max', save_top_k=1)"
    if source.count(old) != 1:
        raise ValueError('Checkpoint callback changed')
    source = source.replace(old, "monitor='val_macro', mode='max', save_top_k=1, save_last=True)")
    source += '\n' + SMOKE + '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One Small/224 nested-outer0 candidate')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    if (a.execute or a.smoke) and a.out is None:
        parser.error('Execution requires --out')
    if a.smoke and not a.execute:
        parser.error('--smoke requires --execute')
    print(json.dumps(run(a.design, a.inputs, a.out, a.execute, a.smoke), indent=2))
'''
    ast.parse(source)
    return source


def payload(root=ROOT):
    root = Path(root)
    frozen = root / 'outputs/distinct_sampling_pair_v1'
    results = root / 'outputs/kaggle_distinct_training_pair_v1_results'
    spec = json.loads((frozen / 'control/spec.json').read_text())
    files = {}
    for name, digest in spec['runtime_code_sha256'].items():
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Frozen runtime changed: {name}')
        files[name] = content
    for name in ['src/__init__.py', 'src/data/__init__.py', 'src/labels/__init__.py', 'src/validation/__init__.py']:
        files[name] = (root / name).read_bytes()
    files['tools/colab_small_runtime.py'] = (root / 'tools/colab_small_runtime.py').read_bytes()
    files['tools/train_small.py'] = small_source(files['tools/train_nested_proxy_control.py'].decode()).encode()
    for name in ['roles.csv', 'proxy_labels.parquet', 'class_counts.csv']:
        files['design/candidate/' + name] = (frozen / 'control' / name).read_bytes()
    for name in ['exclusions.csv', 'raw_counts.parquet']:
        files['design/' + name] = (frozen / name).read_bytes()
    spec.update(design='outer0_convnext_small_224_colab_v1', backbone='convnext_small.fb_in22k_ft_in1k',
                candidate_status='prepared_not_trained', sampling_policy='legacy',
                comparison_scope='exploratory historical Tiny control; Colab runtime may differ')
    spec['runtime_code_sha256'].update({name: hashlib.sha256(files[name]).hexdigest()
        for name in ['tools/train_small.py', 'tools/colab_small_runtime.py']})
    files['design/candidate/spec.json'] = (json.dumps(spec, indent=2) + '\n').encode()
    control = json.loads((results / 'control/run_contract.json').read_text())
    for name, key in [('selected.pt', 'checkpoint_sha256'), ('outer_predictions.parquet', 'outer_predictions_sha256')]:
        if colab_small_runtime.sha(results / 'control' / name) != control[key]:
            raise ValueError('Historical control artifact changed: ' + name)
    for name in ['run_contract.json', 'outer_predictions.parquet']:
        files['reference/' + name] = (results / 'control' / name).read_bytes()
    files['reference/inputs.json'] = (results / 'pair_source/inputs.json').read_bytes()
    expected = json.loads(files['reference/inputs.json'])
    files['metadata/train_series.csv'] = (root / 'data/meta/train_series.csv').read_bytes()
    if hashlib.sha256(files['metadata/train_series.csv']).hexdigest() != expected['series_meta']['sha256']:
        raise ValueError('Series metadata differs from control')
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files['bundle_manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return buffer.getvalue(), expected


def build(root=ROOT, destination=None):
    root = Path(root)
    archive, expected = payload(root)
    digest = hashlib.sha256(archive).hexdigest()
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': (source.strip() + '\n').splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# ConvNeXt-Small · 224 pixels · fold 0 · Colab
Train **one Small model** using your existing three full 224-pixel cache shards on Drive.
Choose a GPU runtime (A100 if available), set the paths below, and run cells in order.
About **35 GB free local disk** is needed. Original image arrays total 29.35 GB.

Frozen recipe: `convnext_small.fb_in22k_ft_in1k`, 16 slices × five series,
protocol series selection, original sampling, seed 0, 12 epochs, effective batch eight.
**Fold 0 means the current nested outer0 split:** 2,208 training / 553 inner-selection /
691 outer-evaluation studies. It is not the older 4,349-study five-fold split.
Labels and splits are included; no repository upload or label regeneration is needed.
Gold is excluded and never monitored. No 384-pixel cache is used.

A short smoke tests two optimizer steps and checkpoint reload on training images.
Full training starts fresh afterward. Checkpoints and logs save directly to Drive.
Only inner validation selects the checkpoint; outer predictions are produced afterward.

The saved Kaggle Tiny control is a **historical exploratory comparison**: Colab hardware
and runtime can differ. A favorable comparison needs a separate matched confirmation.
No additional folds, blends or submissions run automatically.
''')
    add('code', '''DRIVE_CACHE = "/content/drive/MyDrive/kagglehub"  # Folder containing all three original shards
SIDE_FILE = ""  # Optional full Drive path to side_predictions.parquet if outside DRIVE_CACHE
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/convnext_small_224_outer0_v1"
LOCAL_INPUTS = "/content/knee_small_inputs"
''')
    add('markdown', '## 1. Setup and mount Drive\nInternet is needed for packages and ImageNet pretrained weights. A Kaggle token is needed only if side predictions are missing from Drive.')
    add('code', '''%pip install -q "timm==1.0.26" "lightning==2.6.5" kagglehub pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" numpy pandas tqdm
import importlib, shutil, torch
for package in ("timm", "lightning.pytorch", "pyarrow", "sklearn", "cv2", "pydicom", "pandas", "kagglehub"):
    importlib.import_module(package)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Choose a Colab runtime with exactly one GPU")
from google.colab import drive
drive.mount("/content/drive")
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
print("Free local disk GiB:", round(shutil.disk_usage('/content').free / 2**30, 1))
''')
    add('markdown', '## 2. Restore the frozen code and fold\nThis compressed bundle contains metadata, fixed labels/splits, source and Tiny reference predictions; no MRI images, report text or trained weights.')
    add('code', f'''import base64, hashlib, io, json, pathlib, sys, zipfile, importlib
BUNDLE_SHA256 = {digest!r}
BUNDLE_BASE64 = {base64.b64encode(archive).decode()!r}
CODE_ROOT = pathlib.Path('/content/knee_small_code')
blob = base64.b64decode(BUNDLE_BASE64)
assert hashlib.sha256(blob).hexdigest() == BUNDLE_SHA256
CODE_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as bundle:
    for name in bundle.namelist():
        target = (CODE_ROOT / name).resolve()
        if not target.is_relative_to(CODE_ROOT.resolve()):
            raise ValueError('Invalid bundle path')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bundle.read(name))
manifest = json.loads((CODE_ROOT / 'bundle_manifest.json').read_text())
for name, expected_hash in manifest.items():
    assert hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() == expected_hash, name
sys.path.insert(0, str(CODE_ROOT))
for name in list(sys.modules):
    if name in ('src', 'tools') or name.startswith(('src.', 'tools.')):
        del sys.modules[name]
importlib.invalidate_caches()
from tools.colab_small_runtime import stage_inputs, run_small, run_process
print('Verified bundle:', BUNDLE_SHA256)
''')
    add('markdown', '## 3. Copy and verify your existing shards\nImages are copied to fast Colab local storage. Correct existing local copies are reused. Full SHA-256 checks reject thinned caches, mismatched versions and incomplete copies.')
    add('code', RESTORE_PATHS + f'''EXPECTED_INPUTS = {expected!r}
if 'stage_inputs' not in globals() or 'CODE_ROOT' not in globals():
    raise RuntimeError('Run section 2 (Restore the frozen code and fold) before this cell.')
INPUTS = stage_inputs(DRIVE_CACHE, LOCAL_INPUTS, CODE_ROOT, EXPECTED_INPUTS, SIDE_FILE)
# Actual data/role/provenance preflight before any optimizer step.
run_process([sys.executable, '-u', str(CODE_ROOT / 'tools/train_small.py'),
             '--design', str(CODE_ROOT / 'design/candidate'), '--inputs', str(INPUTS)],
            CODE_ROOT, pathlib.Path(LOCAL_INPUTS) / 'preflight.log')
''')
    add('markdown', '''## 4. Smoke test, then train Small once
Selected and last training checkpoints, metrics, predictions and source receipts save to Drive.
The smoke uses transient weights; full training resets the seed and starts from original pretraining.
If the smoke fails or the GPU runs out of memory, execution stops without changing the recipe.

Completed runs are hash-verified and skipped on rerun. A partial run **stops for recovery**,
preserving `last.ckpt` and the selected checkpoint; it does not silently restart fitting.
Do not run two sessions against the same output folder.
''')
    add('code', RESTORE_PATHS + '''if 'INPUTS' not in globals() or 'run_small' not in globals():
    raise RuntimeError('Run sections 2 and 3 (restore code, then verify shards) before training.')
RUN_DIR = run_small(CODE_ROOT, INPUTS, DRIVE_OUT, BUNDLE_SHA256)
''')
    add('markdown', '''## 5. Results
The exploratory screen requires macro delta ≥0.005, no finding delta below −0.020,
and a positive lower paired-bootstrap bound. A pass is a reason to propose a matched
confirmation, not model promotion or evidence of independent image accuracy.
For review, retain `complete.json`, `comparison.json`, `screen_decision.json`,
`candidate/run_contract.json`, `candidate/outer_predictions.parquet`, training metrics,
and `candidate/selected.pt`. All are saved under DRIVE_OUT.
''')
    add('code', '''print('Saved to:', RUN_DIR)
print((RUN_DIR / 'comparison.json').read_text())
print((RUN_DIR / 'screen_decision.json').read_text())
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 4, 'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    path = Path(destination) if destination else root / 'notebooks/22_convnext_small_224_fold0_colab.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    print(path)
    return path


if __name__ == '__main__':
    build()
