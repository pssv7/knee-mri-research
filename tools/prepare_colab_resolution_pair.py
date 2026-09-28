"""Stage matched single-GPU 224/384 Colab screening, after spatial screening."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_resolution_training import resolution_source


def build(root=ROOT, fold=1):
    root = Path(root)
    if fold not in (1, 3):
        raise ValueError('Screen fold 1, then fold 3 only if fold 1 passes')
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    hashes = {n: hashlib.sha256((root / 'data/meta' / n).read_bytes()).hexdigest()
              for n in ('train_labels.parquet', 'train.csv', 'train_series.csv')}
    sources = {size: {phase: resolution_source(notebook, hashes['train_labels.parquet'], size, fold, smoke)
                     for phase, smoke in (('smoke', True), ('full', False))} for size in (224, 384)}
    tree = ast.parse(sources[224]['full'])
    files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'FILES')
    files.update({name: (root / name).read_text() for name in (
        'tools/colab_protocol_runtime.py', 'tools/colab_slice_runtime.py',
        'tools/colab_resolution_runtime.py', 'tools/compare_oof.py', 'tools/compare_resolution.py')})
    files['tools/__init__.py'] = ''
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': source.strip() + '\n'}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', f'''# Matched resolution screening — fold {fold}, Colab A100

Run after the spatial screening sequence and cache verification. This is the
single-GPU alternative when the projected Kaggle 384 arm exceeds eight hours.
Both resolutions use ConvNeXt V1, protocol selection, 16 selected slices from
24-slice raw-derived caches, five series, seed 0, 12 epochs, gradient
checkpointing and effective batch eight (microbatch 1, accumulation 8).

Requires your **original three 224 shards** and **six 384 shards**, manifests,
original 384 receipts, labels and side predictions. No resizing one cache to
create the other. Allow about 120 GiB free local disk; outputs are saved to Drive.
Fold 3 is a separate notebook generated only after reviewing the fold-1 gate.
Reruns preserve previous attempts and restart training at epoch zero.
No leaderboard improvement is guaranteed. Gold is diagnostic only.
''')
    add('code', '''DRIVE_CACHE_224 = "/content/drive/MyDrive/kagglehub"
DRIVE_CACHE_384 = "/content/drive/MyDrive/knee_cache_384_v1"
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/resolution_pair"
LOCAL_224 = "/content/knee_resolution224_inputs"
LOCAL_384 = "/content/knee_resolution384_inputs"''')
    add('code', '''%pip install -q timm==1.0.26 lightning==2.6.5 kagglehub pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" numpy pandas tqdm
from google.colab import drive
drive.mount("/content/drive")
import torch
assert torch.cuda.device_count() == 1, "Select one GPU"
print(torch.cuda.get_device_name())''')
    add('code', f'''from pathlib import Path
import sys
FILES = {files!r}
CODE_ROOT = Path("/content/knee_campaign_code")
for name, content in FILES.items():
    path = CODE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
sys.path.insert(0, str(CODE_ROOT))
HASHES = {hashes!r}
SOURCES = {sources!r}
FOLD = {fold}
from tools.colab_slice_runtime import stage_drive_inputs
from tools.colab_resolution_runtime import adopt_drive_inputs, copy_inputs_to_colab, run_training
from tools.compare_resolution import run as compare_resolution''')
    add('code', '''stage_drive_inputs(DRIVE_CACHE_224, LOCAL_224, HASHES['train_labels.parquet'])
# The 224 runtime writes a hash map; mark its verified state for the shared runner.
import json
path = Path(LOCAL_224) / 'input_receipt.json'
record = json.loads(path.read_text())
record['state'] = 'verified'
path.write_text(json.dumps(record, indent=2))
adopt_drive_inputs(DRIVE_CACHE_384, HASHES)
copy_inputs_to_colab(DRIVE_CACHE_384, LOCAL_384)''')
    add('markdown', '''## Train the matched pair
Each arm runs its actual training smoke before the 12-epoch run. Timing and
training/validation memory are saved in `training_measurement.json`.
The comparison requires complete full-run receipts and matching normalized
contracts, initial weights, labels, splits and effective batch. The standalone
gate is ≥ +0.005 with no finding below −0.020.''')
    add('code', '''CONTROL = run_training(SOURCES[224], Path(DRIVE_OUT) / f"fold{FOLD}/224", LOCAL_224, FOLD, 224)
CANDIDATE = run_training(SOURCES[384], Path(DRIVE_OUT) / f"fold{FOLD}/384", LOCAL_384, FOLD, 384)
COMPARISON = Path(CANDIDATE).parents[1] / 'comparison'
compare_resolution(CONTROL, CANDIDATE, Path(LOCAL_384) / 'auxiliary/train_labels.parquet', COMPARISON)
print('Saved comparison:', COMPARISON)''')
    path = root / f'notebooks/20_resolution_pair_fold{fold}_colab.ipynb'
    path.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 4, 'cells': cells,
        'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}, indent=1) + '\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=(1, 3), default=1)
    print(build(fold=parser.parse_args().fold))
