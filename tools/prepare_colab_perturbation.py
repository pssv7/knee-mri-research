"""Generate the self-contained Colab notebook for the perturbation probe.

Inference only, on the historical Tiny control and its 691 outer0 studies.
The frozen design, runtime source and reference predictions are bundled
byte-identical and hash-verified; the checkpoint is read from Drive and
verified against the control's run contract. See src/model/perturb.py.
"""
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

NOTEBOOK = 'notebooks/23_perturbation_probe_colab.ipynb'

RESTORE_PATHS = '''# Restore missing settings without overwriting your custom paths.
for name, default in {
    'DRIVE_CACHE': '/content/drive/MyDrive/kagglehub',
    'SIDE_FILE': '',
    'CHECKPOINT': '/content/drive/MyDrive/knee_runs/tiny_control_outer0/selected.pt',
    'DRIVE_OUT': '/content/drive/MyDrive/knee_runs/perturbation_probe_tiny_outer0_v1',
    'LOCAL_INPUTS': '/content/knee_small_inputs',
}.items():
    globals().setdefault(name, default)
'''

# Runs in the notebook kernel; the probe itself runs in a child process.
PHASES = r'''
def probe_command(limit=None):
    command = [sys.executable, '-u', str(CODE_ROOT / 'tools/perturbation_probe.py'),
               '--design', str(CODE_ROOT / 'design/control'), '--inputs', str(INPUTS),
               '--checkpoint', str(CHECKPOINT), '--reference', str(CODE_ROOT / 'reference')]
    return command + (['--limit', str(limit)] if limit else [])


def run_phase(phase, limit=None):
    """One output folder per phase. Complete phases are reused, partial ones stop."""
    out = pathlib.Path(DRIVE_OUT) / phase
    receipt = out / 'run_contract.json'
    if receipt.is_file():
        contract = json.loads(receipt.read_text())
        print(f'{phase}: already complete, reusing', out)
        return contract
    if out.exists():
        raise RuntimeError(f'{phase}: partial output at {out}. Nothing in it is needed; rename it '
                           '(keep the log) and rerun this cell.')
    pathlib.Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
    run_process(probe_command(limit) + ['--execute', '--out', str(out)], CODE_ROOT,
                pathlib.Path(DRIVE_OUT) / f'{phase}.log')
    contract = json.loads(receipt.read_text())
    if not contract['reproduction']['passed']:
        raise RuntimeError(f'{phase}: identity pass does not reproduce the saved predictions')
    return contract
'''


def payload(root=ROOT, include_series=True):
    root = Path(root)
    frozen = root / 'outputs/distinct_sampling_pair_v1'
    results = root / 'outputs/kaggle_distinct_training_pair_v1_results'
    spec_bytes = (frozen / 'control/spec.json').read_bytes()
    spec = json.loads(spec_bytes)
    files = {}
    for name, digest in spec['runtime_code_sha256'].items():
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Frozen runtime changed: {name}')
        files[name] = content
    for name in ['src/__init__.py', 'src/data/__init__.py', 'src/labels/__init__.py',
                 'src/validation/__init__.py', 'src/model/perturb.py',
                 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']:
        files[name] = (root / name).read_bytes()
    # The frozen control design, unmodified: preflight checks it is the Tiny recipe.
    files['design/control/spec.json'] = spec_bytes
    for name in ['roles.csv', 'proxy_labels.parquet', 'class_counts.csv']:
        files['design/control/' + name] = (frozen / 'control' / name).read_bytes()
    for name in ['exclusions.csv', 'raw_counts.parquet']:
        files['design/' + name] = (frozen / name).read_bytes()
    control = json.loads((results / 'control/run_contract.json').read_text())
    for name, key in [('selected.pt', 'checkpoint_sha256'), ('outer_predictions.parquet', 'outer_predictions_sha256')]:
        if colab_small_runtime.sha(results / 'control' / name) != control[key]:
            raise ValueError('Historical control artifact changed: ' + name)
    for name in ['run_contract.json', 'outer_predictions.parquet']:
        files['reference/' + name] = (results / 'control' / name).read_bytes()
    files['reference/inputs.json'] = (results / 'pair_source/inputs.json').read_bytes()
    expected = json.loads(files['reference/inputs.json'])
    series = (root / 'data/meta/train_series.csv').read_bytes()
    if hashlib.sha256(series).hexdigest() != expected['series_meta']['sha256']:
        raise ValueError('Series metadata differs from control')
    # Kaggle mounts the competition's own copy; the kernel hash-checks it there.
    if include_series:
        files['metadata/train_series.csv'] = series
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files['bundle_manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return buffer.getvalue(), expected, control['checkpoint_sha256']


def build(root=ROOT, destination=None):
    root = Path(root)
    archive, expected, checkpoint_hash = payload(root)
    digest = hashlib.sha256(archive).hexdigest()
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': (source.strip() + '\n').splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# Perturbation probe · Tiny control · outer0 · Colab
Measure what the model reacts to: each of the **691 outer0 studies** is scored unmodified
and under 14 controlled variants of its own input, and the predictions are compared.
**Variants are not labels and not medical truth**; no label is read, nothing is trained,
and nothing is submitted.

Declared grid (`src/model/perturb.py`): identity; intensity shift ±5/±10/±20 levels;
Gaussian noise σ 2/5/10; slice removal (even 12, alternate 8, central 12 of 16);
correct left/right flip (outputs re-permuted medial↔lateral); side-covariate toggle.

Model: the historical Kaggle Tiny control (`convnext_tiny.fb_in22k_ft_in1k`, 224 px,
16 slices × five series). Upload its `selected.pt` to Drive once and set `CHECKPOINT`;
the notebook verifies SHA-256 `''' + checkpoint_hash[:16] + '''…` before anything runs.
The identity pass must reproduce the saved T4 predictions within 0.01, or the run stops.

A 5-study smoke runs first and prints the per-study time; the full run follows.
Needs the same three 224 shards as notebook 22 (reused if already staged locally).
''')
    add('code', '''DRIVE_CACHE = "/content/drive/MyDrive/kagglehub"  # Folder containing all three original shards
SIDE_FILE = ""  # Optional full Drive path to side_predictions.parquet if outside DRIVE_CACHE
CHECKPOINT = "/content/drive/MyDrive/knee_runs/tiny_control_outer0/selected.pt"  # Uploaded Tiny control weights
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/perturbation_probe_tiny_outer0_v1"
LOCAL_INPUTS = "/content/knee_small_inputs"
''')
    add('markdown', '## 1. Setup and mount Drive\nThe import check runs in a child process, exactly as the probe will, so a missing package fails here rather than after staging.')
    add('code', '''%pip install -q "timm==1.0.26" "lightning==2.6.5" kagglehub pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" numpy pandas tqdm
import shutil, subprocess, sys, torch
check = subprocess.run([sys.executable, "-c",
    "import sys, torch, timm, lightning.pytorch as pl, pyarrow, sklearn, cv2, pydicom, pandas, kagglehub; "
    "print('child python', sys.version.split()[0], '| torch', torch.__version__, '| timm', timm.__version__, "
    "'| lightning', pl.__version__, '| cuda', torch.cuda.is_available())"],
    capture_output=True, text=True)
print(check.stdout, check.stderr)
if check.returncode != 0:
    raise RuntimeError("Packages do not import in a child process; see the error above")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Choose a Colab runtime with exactly one GPU")
from google.colab import drive
drive.mount("/content/drive")
print("GPU:", torch.cuda.get_device_name(0))
print("Free local disk GiB:", round(shutil.disk_usage('/content').free / 2**30, 1))
''')
    add('markdown', '## 2. Restore the frozen code and design\nThe bundle holds source, the frozen outer0 design and the control\'s saved predictions; no MRI images, report text or weights.')
    add('code', f'''import base64, hashlib, io, json, pathlib, sys, zipfile, importlib
BUNDLE_SHA256 = {digest!r}
BUNDLE_BASE64 = {base64.b64encode(archive).decode()!r}
CODE_ROOT = pathlib.Path('/content/knee_probe_code')
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
from tools.colab_small_runtime import stage_inputs, run_process
print('Verified bundle:', BUNDLE_SHA256)
''')
    add('markdown', '## 3. Stage shards, then preflight\nShards are copied to local disk and fully hash-checked (existing verified copies are reused). The preflight checks the design, inputs, checkpoint hash and reference predictions without touching the GPU.')
    add('code', RESTORE_PATHS + f'''EXPECTED_INPUTS = {expected!r}
if 'stage_inputs' not in globals() or 'CODE_ROOT' not in globals():
    raise RuntimeError('Run section 2 (Restore the frozen code and design) before this cell.')
if not pathlib.Path(CHECKPOINT).is_file():
    raise RuntimeError(f'Upload the Tiny control selected.pt to Drive and set CHECKPOINT (now {{CHECKPOINT}})')
if not pathlib.Path(DRIVE_OUT).resolve().is_relative_to(pathlib.Path('/content/drive/MyDrive').resolve()):
    raise ValueError('Choose DRIVE_OUT inside MyDrive')
INPUTS = stage_inputs(DRIVE_CACHE, LOCAL_INPUTS, CODE_ROOT, EXPECTED_INPUTS, SIDE_FILE)
run_process([sys.executable, '-u', str(CODE_ROOT / 'tools/perturbation_probe.py'),
             '--design', str(CODE_ROOT / 'design/control'), '--inputs', str(INPUTS),
             '--checkpoint', str(CHECKPOINT), '--reference', str(CODE_ROOT / 'reference')],
            CODE_ROOT, pathlib.Path(LOCAL_INPUTS) / 'probe_preflight.log')
''')
    add('markdown', '''## 4. Smoke on 5 studies, then the full probe
Each line of progress prints the time per study and the projected total.
The smoke must reproduce the saved predictions before the full run starts.
Outputs save to `DRIVE_OUT/smoke` and `DRIVE_OUT/full`; a completed phase is reused on rerun.
''')
    add('code', RESTORE_PATHS + PHASES + '''
if 'INPUTS' not in globals() or 'run_process' not in globals():
    raise RuntimeError('Run sections 2 and 3 (restore code, then stage and preflight) first.')
SMOKE = run_phase('smoke', limit=5)
print('Smoke reproduction:', SMOKE['reproduction'])
FULL = run_phase('full')
print('Full reproduction:', FULL['reproduction'], '| minutes:', round(FULL['elapsed_seconds'] / 60, 1))
''')
    add('markdown', '''## 5. Results
p95 |Δp| per variant × finding (identity pass as the base; flipped outputs re-permuted).
"Sensitive" is the declared descriptive flag: p95 |Δp| > max(0.05, 3 × the identity-vs-saved p95).
Nothing is gated on it. For MCL under the flip the expectation is undefined (no lateral counterpart label).
Keep `full/summary.json`, `full/perturbation_predictions.parquet` and `full/run_contract.json` for review.
''')
    add('code', '''import pandas as pd
summary = json.loads((pathlib.Path(DRIVE_OUT) / 'full/summary.json').read_text())
table = pd.DataFrame(summary['rows'])
pd.set_option('display.width', 200)
print(table.pivot(index='variant', columns='finding', values='p95_abs_dp').round(3).to_string())
print()
print(table[table.sensitive][['variant', 'finding', 'p95_abs_dp', 'mean_dlogit', 'spearman', 'note']].to_string(index=False))
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 4, 'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    path = Path(destination) if destination else root / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    print(path)
    return path


if __name__ == '__main__':
    build()
