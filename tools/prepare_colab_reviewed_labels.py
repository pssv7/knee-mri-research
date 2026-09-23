"""Generate a Drive-first Colab fold-3 reviewed-label replication notebook."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_reviewed_labels import stage_source


def build(root=ROOT):
    root = Path(root)
    reviewed = root / "outputs/reviewed_labels_v1"
    corrections = json.loads((reviewed / "corrections.json").read_text())
    label_hash = hashlib.sha256((root / "data/meta/train_labels.parquet").read_bytes()).hexdigest()
    source_notebook = json.loads((root / "notebooks/08_finetune.ipynb").read_text())
    sources = {arm: stage_source(source_notebook, label_hash, corrections, arm, fold=3)
               for arm in ("control", "candidate")}
    compare = (root / "tools/compare_oof.py").read_text()
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Reviewed labels — fold 3 replication on Colab

Run this after fold 1 passes. It repeats the matched 224-pixel control/candidate
comparison on fold 3. Existing Drive image shards are copied to the VM before
training; only small labels, side predictions, and metadata CSVs are fetched.
No raw RSNA image archive or submission is used.
""")
    add("code", """DRIVE_INPUTS = "/content/drive/MyDrive/knee_cache_224_reviewed_v1"
DRIVE_SHARD_ROOT = "/content/drive/MyDrive/kagglehub/notebooks/pssv27"
DRIVE_OUTPUT = "/content/drive/MyDrive/knee_runs/reviewed_labels_colab_v1"
LOCAL_INPUTS = "/content/knee_reviewed_inputs"
INPUT_MOUNT = "/content/knee_reviewed_mounts"
FOLD = 3
""")
    add("markdown", "## 1. Install and mount Drive")
    add("code", """%pip install -q kagglehub timm==1.0.26 lightning==2.6.5 pyarrow scikit-learn opencv-python-headless "pydicom>=3.0" tqdm
from google.colab import drive
drive.mount("/content/drive")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Select exactly one GPU runtime")
print("GPU:", torch.cuda.get_device_name(0))
""")
    add("markdown", "## 2. Kaggle login")
    add("code", """import kagglehub
kagglehub.login()
""")
    add("markdown", "## 3. Reuse Drive shards; download only small metadata")
    add("code", """import os, shutil
from pathlib import Path
DRIVE = Path(DRIVE_INPUTS)
SHARD_ROOT = Path(DRIVE_SHARD_ROOT)
DRIVE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(DRIVE / "kagglehub")

def cached(name, fetch):
    marker = DRIVE / f"{name}.path"
    if marker.exists() and Path(marker.read_text().strip()).exists():
        print("Reusing Drive download:", name)
        return Path(marker.read_text().strip())
    value = Path(fetch())
    marker.write_text(str(value))
    return value

shards = []
for n in (1, 2, 3):
    shard = SHARD_ROOT / f"knee-cache-224-01-shard-{n}"
    if not shard.exists():
        raise FileNotFoundError(f"Missing existing Drive shard: {shard}")
    shards.append(shard)
print("Reusing existing Drive image shards:", shards)
labels = cached("labels", lambda: kagglehub.dataset_download("pssv27/knee-labels"))
side = cached("side", lambda: kagglehub.notebook_output_download("pssv27/knee-side-classifier"))
metadata = DRIVE / "metadata"; metadata.mkdir(exist_ok=True)
for name in ("train.csv", "train_series.csv"):
    target = metadata / name
    if not target.exists():
        downloaded = Path(kagglehub.competition_download("rsna-knee-abnormality-detection", path=name))
        candidates = [downloaded] if downloaded.name == name else list(downloaded.rglob(name))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one downloaded {name}, found {candidates}")
        shutil.copy2(candidates[0], target)
print("Small metadata and labels are cached on Drive; no full competition download was requested")
""")
    add("markdown", "## 4. Copy inputs from Drive to the Colab VM")
    add("code", """from tqdm.auto import tqdm
import shutil

def copy_tree(source, destination):
    source, destination = Path(source), Path(destination)
    files = [p for p in source.rglob("*") if p.is_file()]
    with tqdm(total=sum(p.stat().st_size for p in files), desc=f"Copy {source.name}", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for item in files:
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            with item.open("rb") as incoming, target.open("wb") as outgoing:
                while block := incoming.read(8 * 1024**2):
                    outgoing.write(block); bar.update(len(block))

local = Path(LOCAL_INPUTS); local.mkdir(parents=True, exist_ok=True)
downloaded = [Path(DRIVE_SHARD_ROOT) / f"knee-cache-224-01-shard-{n}" for n in (1, 2, 3)]
downloaded += [Path((Path(DRIVE_INPUTS) / "labels.path").read_text().strip()), Path((Path(DRIVE_INPUTS) / "side.path").read_text().strip()), Path(DRIVE_INPUTS) / "metadata"]
names = ["cache1", "cache2", "cache3", "labels", "side", "metadata"]
for source, name in zip(downloaded, names):
    target = local / name
    if target.exists(): print("Reusing local input:", name)
    else: copy_tree(source, target)

inputs = Path(INPUT_MOUNT); inputs.mkdir(parents=True, exist_ok=True)
for name in names:
    target = inputs / f"reviewed-{name}"
    if not target.exists(): target.symlink_to((local / name).resolve(), target_is_directory=True)
print("Local inputs ready")
""")
    runtime = """import ast, hashlib, json, os, pathlib, subprocess, sys, uuid
from pathlib import Path
def relocate(source, output):
    class Rewrite(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith('/kaggle/working'):
                return ast.copy_location(ast.Constant(str(output) + node.value[len('/kaggle/working'):]), node)
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace('/kaggle/input', INPUT_MOUNT)), node)
            return node
    return ast.unparse(ast.fix_missing_locations(Rewrite().visit(ast.parse(source)))) + '\\n'
def run_process(script, directory):
    with (directory / 'train.log').open('w', buffering=1) as log:
        p = subprocess.Popen([sys.executable, '-u', str(script)], cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout: print(line, end='', flush=True); log.write(line)
        if p.wait() != 0: raise RuntimeError(f'Training failed: {directory / "train.log"}')
"""
    add("markdown", "## 5. Run the matched fold-3 pair")
    add("code", runtime + f"""
FOLD = 3
LABELS_SHA256 = {label_hash!r}
CORRECTIONS_SHA256 = {hashlib.sha256(json.dumps(corrections, sort_keys=True).encode()).hexdigest()!r}
SOURCES = {sources!r}
COMPARE_SOURCE = {compare!r}
root = Path(DRIVE_OUTPUT) / f"fold{{FOLD}}"; root.mkdir(parents=True, exist_ok=True)
attempt = root / f"attempt_{{uuid.uuid4().hex[:12]}}"; attempt.mkdir()
runs = {{}}
for arm in ("control", "candidate"):
    arm_root = attempt / arm; arm_root.mkdir()
    script = arm_root / "train.py"; script.write_text(relocate(SOURCES[arm], arm_root))
    run_process(script, arm_root)
    runs[arm] = arm_root / f"reviewed_{{arm}}_fold{{FOLD}}"

control = json.loads((runs["control"] / "label_review_contract.json").read_text())
candidate = json.loads((runs["candidate"] / "label_review_contract.json").read_text())
assert control["correction_count"] == 0 and candidate["correction_count"] == 11
assert control["corrections_sha256"] == candidate["corrections_sha256"] == CORRECTIONS_SHA256
assert not control["validation_labels_changed"] and not candidate["validation_labels_changed"]
assert json.loads((runs["control"] / "run_contract.json").read_text()) == json.loads((runs["candidate"] / "run_contract.json").read_text())
namespace = {{"__name__": "pair_comparison", "__file__": str(attempt / "compare_oof.py")}}
exec(compile(COMPARE_SOURCE, namespace["__file__"], "exec"), namespace)
import pandas as pd
folds = pd.read_parquet(runs["control"] / "folds.parquet")
frames = [namespace["read_predictions"]([runs[arm] / f"oof_fold{{FOLD}}.parquet"], folds) for arm in ("control", "candidate")]
labels = pd.read_parquet(next(Path("/kaggle/input").rglob("train_labels.parquet")))
table, summary = namespace["compare"](labels, *frames, draws=1000, seed=0)
delta = summary["macro"]["candidate"] - summary["macro"]["baseline"]
passed = delta >= .003 and bool((table.candidate_delta >= -.02).all())
summary["screening_gate"] = {{"passed": passed, "macro_delta": delta, "required_macro_delta": .003, "maximum_finding_drop": .02}}
table.to_csv(attempt / "per_finding.csv", index_label="finding")
(attempt / "comparison.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(table.round(5).to_string())
print(json.dumps(summary["screening_gate"], indent=2))
""")
    add("markdown", "Continue only if fold 3 gains at least +0.003 macro AUC and no finding falls more than 0.02. Share `comparison.json` and `per_finding.csv` before training further folds.")
    result = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    path = root / "notebooks/18_reviewed_labels_fold3_colab.ipynb"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(path)
    return path


if __name__ == "__main__":
    build()
