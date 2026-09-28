"""Self-contained orchestration embedded in the protocol Colab notebook."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_colab_contract(current, reference):
    # GPU count/runtime may change; model initialization and recipe may not.
    allowed = {"fold", "steps", "warmup_steps", "devices", "torch"}
    changes = {key: {"screened": reference.get(key), "colab": current.get(key)}
               for key in current.keys() | reference.keys()
               if current.get(key) != reference.get(key)}
    unexpected = set(changes) - allowed
    if unexpected or current["devices"] != 1:
        raise ValueError(f"Colab contract mismatch: {changes}; require one GPU")
    return changes


class ColabPaths(ast.NodeTransformer):
    def __init__(self, output):
        self.output = str(output)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/kaggle/working"):
            return ast.copy_location(ast.Constant(
                self.output + node.value[len("/kaggle/working"):]), node)
        return node


def relocate(source, output):
    tree = ColabPaths(output).visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def required_files(fold):
    weights = [f"convnext_fold{fold}.pt", f"avg_fold{fold}_ema.pt", f"avg_fold{fold}_swa.pt"]
    return weights + [name + ".json" for name in weights] + [
        "folds.parquet", "run_contract.json", "colab_execution.json",
        f"oof_fold{fold}.parquet", f"gold_predictions_fold{fold}.parquet",
        f"lit_fold{fold}/metrics/metrics.csv", "train.py"]


def completed_run(folder, fold, recipe_hash):
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record["fold"] != fold or record["recipe_sha256"] != recipe_hash:
        raise ValueError(f"Completion receipt belongs to another recipe: {marker}")
    run = folder / record["attempt"] / f"protocol_fold{fold}"
    if set(record["sha256"]) != set(required_files(fold)):
        raise ValueError(f"Incomplete artifact receipt: {marker}")
    for name, digest in record["sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Saved artifact changed: {run / name}")
    return run


def run_process(script, directory):
    with (directory / "train.log").open("w", buffering=1) as log:
        process = subprocess.Popen([sys.executable, "-u", str(script)], cwd=directory,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait() != 0:
                raise RuntimeError(f"Training failed; inspect {directory / 'train.log'}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def run_folds(folds, drive_root, scripts, recipe_hash):
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {0, 2, 4}:
        raise ValueError("Choose unique remaining folds from 0, 2, 4")
    root = Path(drive_root)
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive before training")
    if not root.resolve().is_relative_to(Path("/content/drive/MyDrive").resolve()):
        raise ValueError("Outputs must be inside mounted MyDrive")
    root.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        folder = root / f"fold{fold}"
        folder.mkdir(exist_ok=True)
        complete = completed_run(folder, fold, recipe_hash)
        if complete:
            print(f"Fold {fold} already complete and checksums verified: {complete}")
            continue
        # Never mix partial runs. Averaging callbacks do not support exact resume.
        attempt = folder / f"attempt_{uuid.uuid4().hex[:12]}"
        attempt.mkdir()
        progress = {"fold": fold, "state": "running", "recipe_sha256": recipe_hash}
        try:
            for name, source in scripts[fold].items():
                directory = attempt / name
                directory.mkdir()
                script = directory / "train.py"
                script.write_text(relocate(source, attempt))
                progress["active"] = name
                (attempt / "progress.json").write_text(json.dumps(progress, indent=2))
                run_process(script, directory)
            run = attempt / f"protocol_fold{fold}"
            record = {"fold": fold, "attempt": attempt.name, "recipe_sha256": recipe_hash,
                      "sha256": {name: sha256(run / name) for name in required_files(fold)}}
            temporary = folder / "complete.json.tmp"
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(folder / "complete.json")
            progress.update(state="complete", active=None)
            print(f"FOLD {fold} COMPLETE: {run}", flush=True)
        except BaseException as exc:
            progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (attempt / "progress.json").write_text(json.dumps(progress, indent=2))
