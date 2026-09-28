"""Generate a standalone Drive-backed Colab notebook; never launch training."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import colab_protocol_runtime as runtime
from tools.prepare_kaggle_protocol_completion import prepare


STAGE = '''
import getpass
import os
from pathlib import Path
import shutil

import kagglehub
from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token
from google.colab import drive

drive.mount("/content/drive")
if get_kaggle_credentials() is None:
    token = getpass.getpass("Kaggle API token (hidden): ").strip()
    if not token:
        raise RuntimeError("Provide a Kaggle token or enable the KAGGLE_API_TOKEN Colab secret")
    os.environ["KAGGLE_API_TOKEN"] = token
    set_kaggle_api_token(token)
    del token
print("Kaggle account:", kagglehub.whoami()["username"])

# Cache images on the VM, not on Drive's remote random-access filesystem.
os.environ["KAGGLEHUB_CACHE"] = "/content/kagglehub"
inputs = Path("/kaggle/input")
inputs.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage("/content").free < 40 * 10**9 and not any(inputs.iterdir()):
    raise RuntimeError("Need at least 40 GB free local disk for cache downloads and training")

def stage(path, name):
    target = inputs / name
    source = Path(path).resolve()
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise RuntimeError(f"Conflicting input mount: {target}; use a fresh runtime")
    else:
        target.symlink_to(source, target_is_directory=True)

def download_shard(number):
    dataset = f"pssv27/knee-cache-shard-{number}"
    notebook = f"pssv27/knee-cache-224-01-shard-{number}"
    try:
        return kagglehub.dataset_download(dataset)
    except Exception as exc:
        print(f"{dataset}: {type(exc).__name__}; trying original notebook output")
    for attempt in range(10):
        try:
            return kagglehub.notebook_output_download(notebook)
        except Exception as exc:
            if attempt == 9:
                raise
            print(f"Download attempt {attempt + 1} failed: {type(exc).__name__}; retrying")

# Fetch each shard explicitly, including when a previous download was partial.
for number in (1, 2, 3):
    stage(download_shard(number), f"colab-cache-{number}")
stage(kagglehub.dataset_download("pssv27/knee-labels"), "colab-labels")
stage(kagglehub.notebook_output_download("pssv27/knee-side-classifier"), "colab-side")
metadata = inputs / "colab-metadata"
metadata.mkdir(exist_ok=True)
for name in ("train.csv", "train_series.csv"):
    source = kagglehub.competition_download("rsna-knee-abnormality-detection", path=name)
    shutil.copy2(source, metadata / name)

import glob
for pattern, expected in (("cache_train_series_*.npy", 3),
                          ("manifest_train_series_*.parquet", 3),
                          ("train_labels.parquet", 1), ("side_predictions.parquet", 1)):
    paths = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} {pattern}; found {paths}. Use a fresh runtime.")
print("All three image shards and metadata staged locally; Drive outputs:", DRIVE_ROOT)
'''

VALIDATE = '''
from src.model import artifacts
_root = pathlib.Path.cwd()
_folded = pd.read_parquet(_root / "folds.parquet")
_expected = set(_folded.loc[_folded.fold == FOLD, config.ID_COL])
_training = set(_folded.loc[_folded.fold != FOLD, config.ID_COL])
assert len(_folded) == 4349 and _folded[config.ID_COL].is_unique
assert len(_expected) == EXPECTED_VAL and not (_expected & _training)
for _name in (f"convnext_fold{FOLD}.pt", f"avg_fold{FOLD}_ema.pt", f"avg_fold{FOLD}_swa.pt"):
    _metadata = artifacts.load_metadata(_root / _name)
    assert _metadata["fold"] == FOLD and _metadata["experiment"] == RUN.to_dict()
    assert set(_metadata["val_uids"]) == _expected
    assert set(_metadata["train_uids"]) == _training
for _name, _ids in ((f"oof_fold{FOLD}.parquet", _expected),
                    (f"gold_predictions_fold{FOLD}.parquet", set(gold_index.studies))):
    _predictions = pd.read_parquet(_root / _name)
    assert _predictions[config.ID_COL].is_unique and set(_predictions[config.ID_COL]) == _ids
    _values = _predictions[config.LABELS].to_numpy()
    assert np.isfinite(_values).all() and ((_values >= 0) & (_values <= 1)).all()
    if _name.startswith("oof"):
        assert _predictions.fold.eq(FOLD).all()
print("COLAB_ARTIFACT_VALIDATION_PASSED", flush=True)
'''


def adapt(source, smoke=False):
    tree = ast.parse(source)
    guards = [node for node in tree.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "'REFERENCE_CONTRACT' in globals()"]
    if len(guards) != 1:
        raise ValueError("Screened contract guard changed; review Colab adapter")
    index = tree.body.index(guards[0])
    guard = '''
_changes = check_colab_contract(_contract, REFERENCE_CONTRACT)
pathlib.Path("colab_execution.json").write_text(json.dumps({
    "platform": "colab", "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda, "changes_from_screened_fold3": _changes,
    "note": "Single-GPU execution; not bitwise equivalent to the two-T4 screens"
}, indent=2))
'''
    tree.body[index:index + 1] = ast.parse(guard).body
    output = inspect.getsource(runtime.check_colab_contract) + "\n" + ast.unparse(tree)
    if not smoke:
        output += "\n" + VALIDATE
    ast.parse(output)
    return output + "\n"


def build(root=ROOT):
    root = Path(root)
    # Reuse all screen/label/notebook/preset gates, without launching Kaggle jobs.
    prepare(root)
    scripts = {}
    for fold in (0, 2, 4):
        tree = ast.parse((root / f"outputs/kaggle_protocol_fold{fold}_completion/train.py").read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "SCRIPTS")
        scripts[fold] = {name: adapt(source, smoke=name == "ddp_smoke")
                         for name, source in ast.literal_eval(assignment.value).items()}
    runtime_source = (ROOT / "tools/colab_protocol_runtime.py").read_text()
    recipe_hash = hashlib.sha256((json.dumps(scripts, sort_keys=True) + runtime_source).encode()).hexdigest()
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Protocol folds on Colab
Run remaining folds **0, 2 and 4**, sequentially, on one GPU. Folds 1 and 3 are already finished.
Select a GPU runtime (A100 if available), then run the cells in order. Use your Kaggle account
with access to the private inputs and accepted competition rules. Add a `KAGGLE_API_TOKEN`
Colab secret with notebook access, or use the hidden token prompt. Do not paste credentials into code.

The screened recipe stays at 16 slices, 5 series, 224 pixels, 12 epochs, seed 0 and effective
batch 8. Each fold starts with a short smoke test. The runtime and GPU change is recorded;
initial weights, timm and Lightning must still match the screened run. No submissions are made.

Best-validation checkpoints are written to Drive during training; weights, metadata, OOF,
gold predictions and logs stay there. Completed folds are checksum-verified and skipped on rerun.
**An interrupted fold starts a new attempt from epoch 0.** Exact mid-fold resume is not supported
because the screened EMA/SWA callback does not serialize its running averages. Partial attempts
are preserved. Do not run two sessions on the same fold/output folder at the same time.
""")
    add("code", '''FOLDS = [0, 2, 4]  # Use [0], [2] or [4] for separate sessions.
DRIVE_ROOT = "/content/drive/MyDrive/knee_runs/protocol_completion_colab_v1"
''')
    add("code", '''%pip install -q timm==1.0.26 lightning==2.6.5 kagglehub pyarrow scikit-learn
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Select a Colab runtime with exactly one GPU")
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
''')
    add("code", STAGE)
    add("code", runtime_source)
    add("code", f"RECIPE_SHA256 = {recipe_hash!r}\nSCRIPTS = {scripts!r}")
    add("code", "run_folds(FOLDS, DRIVE_ROOT, SCRIPTS, RECIPE_SHA256)")
    add("markdown", """## After completion
Share each fold's `complete.json`, `progress.json`, `run_contract.json`, `colab_execution.json`,
metrics and prediction files for review. Keep the three `.pt` files and their `.pt.json`
sidecars together on Drive. We will compare all five protocol folds with the incumbent
16/24-slice ensemble before deciding whether to build a new submission.
""")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    destination = root / "notebooks/15_protocol_completion_colab.ipynb"
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Prepared {destination}; no Colab runtime started")
    return destination


if __name__ == "__main__":
    build()
