"""Generate a standalone Drive-first 384-pixel Colab training notebook."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.prepare_resolution_training import resolution_source
from tools import colab_resolution_runtime as runtime
from tools import colab_protocol_runtime as common
from tools import colab_slice_runtime as slice_runtime


def build(root=ROOT, fold=1, remaining=False):
    root = Path(root)
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    hashes = {name: hashlib.sha256((root / 'data/meta' / name).read_bytes()).hexdigest()
              for name in ('train_labels.parquet', 'train.csv', 'train_series.csv')}
    sources = {screening_fold: {
        name: resolution_source(notebook, hashes['train_labels.parquet'], 384,
                                screening_fold, smoke,
                                campaign_completion=screening_fold in (0, 2, 4))
        for name, smoke in (('smoke', True), ('full', False))}
        for screening_fold in ((0, 2, 3, 4) if remaining else (1, 3))}
    if remaining:
        from tools.resolution_completion import clean_training_source
        sources = {f: {phase: clean_training_source(s) for phase, s in phases.items()}
                   for f, phases in sources.items()}
    parsed = ast.parse(sources[0 if remaining else fold]['full'])
    files = next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    helpers = '\n'.join(inspect.getsource(obj) for obj in (
        common.ColabPaths, common.relocate, common.run_process,
        slice_runtime.ColabInputPaths, slice_runtime.relocate_inputs))
    runtime_source = inspect.getsource(runtime).replace(
        'from tools.colab_protocol_runtime import relocate, run_process\nfrom tools.colab_slice_runtime import relocate_inputs',
        'import subprocess\n' + helpers)
    ast.parse(runtime_source)
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'source': (source.strip() + '\n').splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# 384-pixel ConvNeXt — fold 1 diagnostic, Colab

**Workflow: existing Google Drive cache → Colab local disk → train.**

Uses the six private `knee-cache-384-shard-*` outputs already in Drive,
verifies checksums, then copies them to VM-local storage. Training saves
checkpoints and logs directly to Drive.

Allow about **90 GiB of free Colab local disk**. Choose a single GPU for
training (A100 if available). On a new GPU runtime, rerun setup and the
local-copy cell; the Drive cache remains untouched.

Trains the **384-pixel candidate only**: the incumbent ConvNeXt-Tiny V1
(`convnext_tiny.fb_in22k_ft_in1k`) with protocol selection,
16 slices × five series, fold 1 only, 12 epochs, effective batch eight,
and gradient checkpointing. A short smoke test runs before each full run. Gold
labels are monitoring only.
The single-GPU result is not a matched comparison with Kaggle DDP.
Use notebook 20 for the campaign's matched single-GPU 224/384 gate;
do not launch fold 3 until fold 1 passes that gate.

An interrupted download resumes from its `.partial` file. A training rerun
creates a new attempt from epoch zero; it does **not** resume optimizer/EMA
state. Existing attempts are preserved. Colab session limits still apply.
''')
    add('code', '''DRIVE_CACHE = "/content/drive/MyDrive/knee_cache_384_v1"
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/resolution384_v1"
LOCAL_INPUTS = "/content/knee_resolution_inputs"
FOLDS_TO_TRAIN = [1]
''')
    add('markdown', '## 1. Install dependencies and mount Drive\nRun this in both the download session and the training session.')
    add('code', '''%pip install -q "kaggle==2.2.4" "timm==1.0.26" "lightning==2.6.5" pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" numpy pandas tqdm requests
import importlib
for package in ("pydicom", "cv2", "numpy", "pandas", "pyarrow", "sklearn", "tqdm", "timm", "lightning.pytorch"):
    importlib.import_module(package)
from google.colab import drive
drive.mount("/content/drive")
import torch, shutil
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU — downloads/copy only")
print("Free local disk (GiB):", round(shutil.disk_usage("/content").free / 2**30, 1))
''')
    add('markdown', '## 2. Load the self-contained training code\nNo repository upload is needed. Run this cell after every runtime restart.')
    add('code', f'''import pathlib, sys, importlib
FILES = {files!r}
CODE_ROOT = pathlib.Path("/content/knee_resolution_code")
for name, content in FILES.items():
    path = CODE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
sys.path.insert(0, str(CODE_ROOT))
for name in list(sys.modules):
    if name == "src" or name.startswith("src."):
        del sys.modules[name]
importlib.invalidate_caches()
''')
    add('code', runtime_source)
    add('code', f'TRAINING_SOURCES_BY_FOLD = {sources!r}')
    add('markdown', '''## 3. Verify Drive cache and copy to Colab local disk
This contacts neither Kaggle nor the network. It verifies the six cache receipts,
their image/manifest hashes and pinned metadata, then copies the verified files
to the Colab VM for training.''')
    add('code', f'''# Auxiliary artifact: latest COMPLETE SideNet output, not side_model.zip
SIDE_CLASSIFIER = "pssv27/knee-side-classifier"
SIDE_PREDICTIONS_FILE = "side_predictions.parquet"
HASHES = {hashes!r}
DRIVE_RECEIPT = adopt_drive_inputs(DRIVE_CACHE, HASHES)
LOCAL_RECEIPT = copy_inputs_to_colab(DRIVE_CACHE, LOCAL_INPUTS)
''')
    add('markdown', '## 4. Smoke test, then train fold 1 only\nCandidate-only diagnostic: use notebook 20 for the matched resolution gate. Do not launch fold 3 before fold 1 passes. Progress, selected checkpoints, EMA/SWA exports and predictions are saved to Drive.')
    add('code', 'RUN_DIRS = run_resolution_folds(TRAINING_SOURCES_BY_FOLD, DRIVE_OUT, LOCAL_INPUTS, FOLDS_TO_TRAIN)')
    add('markdown', '## 5. Saved results')
    add('code', '''from pathlib import Path
for fold, run_dir in RUN_DIRS.items():
    print(f"Fold {fold}: {run_dir}")
    for name in (f"convnext_fold{fold}.pt", f"convnext_fold{fold}.pt.json", f"oof_fold{fold}.parquet", "run_contract.json"):
        path = Path(run_dir) / name
        print(" ", name, "—", round(path.stat().st_size / 2**20, 2), "MiB")
''')
    if remaining:
        from tools.resolution_completion import configure_notebook
        configure_notebook(cells)
    result = {'nbformat': 4, 'nbformat_minor': 4, 'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    path = root / ('notebooks/21_resolution384_remaining_folds_colab.ipynb' if remaining
                   else f'notebooks/17_resolution384_fold{fold}_colab.ipynb')
    path.write_text(json.dumps(result, indent=1) + '\n')
    print(path)
    return path


if __name__ == '__main__':
    build()
