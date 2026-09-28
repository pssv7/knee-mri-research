"""Build a fold-specific Colab slice-transformer comparison using Drive shards."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.prepare_kaggle_training_pair import prepare
from tools import colab_protocol_runtime as common
from tools import colab_slice_runtime as runtime


def build(root=ROOT, fold=2):
    root = Path(root)
    prepare(root=root, candidate="slice_transformer", fold=fold, no_timeout=True)
    folder = root / f"outputs/kaggle_slice_transformer_fold{fold}"
    pair_source = (folder / "train_pair.py").read_text()
    receipt = json.loads((folder / "staging_receipt.json").read_text())
    runtime_source = inspect.getsource(runtime).replace(
        "from tools.colab_protocol_runtime import relocate, run_process",
        "import subprocess\n" + inspect.getsource(common.ColabPaths) + "\n" +
        inspect.getsource(common.relocate) + "\n" + inspect.getsource(common.run_process))
    ast.parse(runtime_source)
    # Small self-contained repair module can be uploaded into an existing
    # Colab runtime without replacing the notebook or discarding copied shards.
    repair = root / "notebooks/colab_slice_staging_fix.py"
    repair.write_text(runtime_source)
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"""# Slice transformer — fold {fold}, Colab
Additional matched comparison alongside Kaggle folds 1 and 3. Uses the original
ConvNeXt-Tiny V1 backbone and protocol selection in both arms. One short smoke
test, then 12 control epochs and 12 slice-transformer epochs. No parent training
or smoke timeout; Colab session limits still apply.

Select one GPU (A100 if available), set `DRIVE_CACHE` to your existing full-shard
folder, then run in order and authorize Drive. The same 224-pixel, 24-slice cache
is copied to VM-local storage and verified; no image shards are downloaded.
Only missing small labels, side predictions or competition metadata may require
Kaggle access (`KAGGLE_API_TOKEN` Colab secret or a hidden prompt).

Best checkpoints and logs are saved to Drive throughout training. A rerun creates
a new attempt from epoch 0; it does not resume an interrupted optimizer/EMA state.
Single-GPU results must be compared with this notebook's matched control.
Gold is diagnostic; the final `comparison.json` contains the mined OOF gate.
""")
    add("code", '''DRIVE_CACHE = "/content/drive/MyDrive/kagglehub"
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/slice_transformer_fold{fold}_v1"
LOCAL_INPUTS = "/content/knee_slice_inputs"
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
    add("code", "ATTEMPT = run_pair_on_drive(PAIR_SOURCE, DRIVE_OUT, LOCAL_INPUTS)")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    destination = root / f"notebooks/16_slice_transformer_fold{fold}_colab.ipynb"
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Prepared {destination}; no Colab session started")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), default=2)
    args = parser.parse_args()
    build(fold=args.fold)
