"""Stage the spatial-encoder screening pair as an A100 Colab notebook.

The plan puts development on an A100 and commits to a smoke plus one paired
fold before anything else is spent. This writes that notebook and nothing
else: no session is started, no GPU is consumed, and the arms are the two
presets -- a 24-slice protocol control and the same model with the 3D blocks.

Drive staging, input verification and execution are the slice-transformer
runtime unchanged, because the inputs are the same 224-pixel, 24-slice cache
and reusing it keeps one staging path under test rather than two.
"""

import argparse
import ast
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.prepare_kaggle_training_pair import prepare
from tools import colab_protocol_runtime as common
from tools import colab_slice_runtime as runtime


def build(root=ROOT, fold=1, smoke_only=False):
    root = Path(root)
    prepare(root=root, candidate="spatial", fold=fold, no_timeout=True,
            smoke_only=smoke_only)
    folder = root / (f"outputs/kaggle_spatial_fold{fold}_smoke" if smoke_only
                     else f"outputs/kaggle_spatial_fold{fold}")
    pair_source = (folder / "train_pair.py").read_text()
    receipt = json.loads((folder / "staging_receipt.json").read_text())
    runtime_source = inspect.getsource(runtime).replace(
        "from tools.colab_protocol_runtime import relocate, run_process",
        "import subprocess\n" + inspect.getsource(common.ColabPaths) + "\n" +
        inspect.getsource(common.relocate) + "\n" + inspect.getsource(common.run_process))
    ast.parse(runtime_source)
    pair_tree = ast.parse(pair_source)
    scripts = next(ast.literal_eval(n.value) for n in pair_tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'TRAINING_SCRIPTS')
    files = next(ast.literal_eval(n.value) for n in ast.parse(scripts['spatial']).body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'FILES')
    # The probe runs with its cwd on Google Drive. The finder for that directory
    # was cached when `pathlib` was imported, and Drive does not bump a directory's
    # mtime when `src/` is created under it, so without invalidation the import
    # below fails with "No module named 'src'" -- the first live Colab run did.
    representative_source = f'''import importlib, pathlib, sys
FILES = {files!r}
for name, content in FILES.items():
    target = pathlib.Path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
sys.path.insert(0, str(pathlib.Path.cwd()))
importlib.invalidate_caches()
''' + (root / 'tools/spatial_smoke.py').read_text()
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"""# Spatial volume encoder — fold {fold}, Colab A100

Matched pair: a 24-slice protocol ConvNeXt-Tiny control, and the same model
with residual 3D convolution blocks inserted between the trunk's feature maps
and its pooling. Both arms use the same cache, split, seed, schedule and
series selection; the 3D path is the only difference, and at initialisation
the candidate reproduces the control exactly.

{'This notebook runs the smoke only.' if smoke_only else
 'Representative geometry smoke, actual training smokes for both arms, then 12 control epochs and 12 candidate epochs, then the comparison.'}
No parent timeout; Colab session limits still apply. A rerun starts a new
attempt from epoch 0 and does not resume optimizer or EMA state.

Before the first full run, read the smoke's step time and peak memory: the
3D blocks add about 2.0M parameters over 38.7M and their cost on an A100 is
not known until measured. `tools/spatial_smoke.py` is the same measurement
standalone, and is worth running first on a fresh runtime.

Select one A100, set `DRIVE_CACHE` to the existing full-shard folder, then run
the cells in order and authorize Drive. The 224-pixel, 24-slice shards are
copied to VM-local storage and verified; no image shards are downloaded.
This requires the **three original 224-pixel shards**. Your six 384-pixel
shards are for the later resolution experiment; they cannot be used here.

The gate is pre-declared: candidate macro at least +0.005 over its control on
this fold, with no finding below -0.020. `comparison.json` carries the mined
OOF numbers; gold remains diagnostic and selects nothing.
""")
    add("code", f'''DRIVE_CACHE = "/content/drive/MyDrive/kagglehub"
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/spatial_fold{fold}_v1"
LOCAL_INPUTS = "/content/knee_spatial_inputs"
''')
    add("code", '''%pip install -q timm==1.0.26 lightning==2.6.5 kagglehub pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" numpy pandas tqdm
import importlib
# Exercise imports before copying 30 GB or creating a training attempt.
for package in ("pydicom", "cv2", "numpy", "pandas", "pyarrow", "sklearn",
                "tqdm", "timm", "lightning.pytorch", "kagglehub"):
    importlib.import_module(package)
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Choose a runtime with exactly one GPU")
print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__)
from google.colab import drive
drive.mount("/content/drive")
''')
    add("code", runtime_source)
    add("code", f"PAIR_SOURCE = {pair_source!r}\nLABELS_SHA256 = {receipt['labels_sha256']!r}")
    add("code", "stage_drive_inputs(DRIVE_CACHE, LOCAL_INPUTS, LABELS_SHA256)")
    add('markdown', '''## Representative GPU correctness and memory gate
One study × five unmasked series × 24 slices, mixed precision, eight
microbatches per optimizer update: effective batch eight. Check forward/backward,
parameter movement, initial identity, export/reload, validation and inference.
Training and validation memory are recorded separately; training must fit 36 GiB.
The actual-data smokes also record timing; CPU measurements cannot establish GPU fit.''')
    add('code', f'''REPRESENTATIVE_SOURCE = {representative_source!r}
from pathlib import Path
import uuid, json
PROBE = Path(DRIVE_OUT) / ("representative_" + uuid.uuid4().hex[:12])
PROBE.mkdir(parents=True)
script = PROBE / 'spatial_smoke.py'
script.write_text(REPRESENTATIVE_SOURCE)
# run_process streams and logs stdout; set argv in a small launcher.
launcher = PROBE / 'run_probe.py'
launcher.write_text("import runpy, sys\\nsys.argv = " + repr([str(script), '--device', 'cuda', '--precision', '16-mixed', '--accumulation', '8', '--full-series', '--out', str(PROBE)]) + "\\nrunpy.run_path(" + repr(str(script)) + ", run_name='__main__')\\n")
run_process(launcher, PROBE)
GPU_SMOKE = json.loads((PROBE / 'smoke.json').read_text())
print(json.dumps(GPU_SMOKE, indent=2))
assert GPU_SMOKE['identical_at_initialisation']
assert all(arm['training_memory_gate_passed'] for arm in GPU_SMOKE['arms'])''')
    add("code", "ATTEMPT = run_pair_on_drive(PAIR_SOURCE, DRIVE_OUT, LOCAL_INPUTS)")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    name = f"19_spatial_fold{fold}{'_smoke' if smoke_only else ''}_colab.ipynb"
    destination = root / "notebooks" / name
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Prepared {destination}; no Colab session started")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 3), default=1)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    build(fold=args.fold, smoke_only=args.smoke_only)
