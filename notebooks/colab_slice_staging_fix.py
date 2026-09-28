"""Drive staging and execution for the standalone slice-transformer pair."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import subprocess
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



def copy_verified(source, target):
    """Stream-copy from Drive; verify bytes locally before exposing the input."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        # Adopt copies left by the original staging cell, which wrote no
        # receipt until every metadata file had been resolved. Size alone is
        # insufficient: verify both contents before reusing the local copy.
        print(f"Verifying existing local copy: {target.name}", flush=True)
        with source.open("rb") as handle:
            expected = hashlib.file_digest(handle, "sha256").hexdigest()
        with target.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual == expected:
            print(f"Reusing {target.name}; no recopy", flush=True)
            return actual
    if shutil.disk_usage(target.parent).free < source.stat().st_size:
        raise RuntimeError(f"Insufficient local disk to copy {source.name}")
    temporary = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        for chunk in iter(lambda: incoming.read(8 << 20), b""):
            digest.update(chunk)
            outgoing.write(chunk)
    with temporary.open("rb") as copied:
        actual = hashlib.file_digest(copied, "sha256").hexdigest()
    if temporary.stat().st_size != source.stat().st_size or actual != digest.hexdigest():
        raise RuntimeError(f"Copy verification failed: {source}")
    temporary.replace(target)
    return actual


def choose_metadata(hits, name, expected_hash=None):
    """Collapse byte-identical copies; otherwise fetch the authoritative file.

    Never pick an arbitrary Drive version. The labels have a pinned hash, so
    select the matching copy even when older label matrices coexist.
    """
    groups = {}
    for path in sorted(map(Path, hits)):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        groups.setdefault(digest, []).append(path)
    if expected_hash is not None:
        matches = groups.get(expected_hash, [])
        if matches:
            return matches[0]
    elif len(groups) == 1:
        copies = next(iter(groups.values()))
        if len(copies) > 1:
            print(f"{name}: {len(copies)} identical Drive copies; using {copies[0]}")
        return copies[0]
    if groups:
        print(f"{name}: Drive versions differ or do not match the required hash; "
              "fetching official metadata from Kaggle", flush=True)
    return None


def stage_drive_inputs(drive_cache, local_root, labels_hash):
    import numpy as np
    import pandas as pd

    root, local = Path(drive_cache), Path(local_root)
    if not root.is_dir():
        raise RuntimeError(f"Set DRIVE_CACHE to the folder containing your shards: {root}")
    local.mkdir(parents=True, exist_ok=True)
    cache_local = local / "cache"
    cache_local.mkdir(exist_ok=True)

    def unique(pattern, expected):
        matches = sorted(root.rglob(pattern))
        if len(matches) != expected:
            raise RuntimeError(f"Expected {expected} {pattern} in DRIVE_CACHE; found {matches}. "
                               "Choose one complete cache folder, without duplicate versions.")
        return matches

    shards = unique("cache_train_series_*.npy", 3)
    manifests = unique("manifest_train_series_*.parquet", 3)
    by_name = {path.name: path for path in manifests}
    pairs = []
    for shard in shards:
        name = shard.name.replace("cache_", "manifest_", 1).replace(".npy", ".parquet")
        if name not in by_name:
            raise RuntimeError(f"Missing matching manifest: {name}")
        pairs.append((shard, by_name[name]))
    needed = sum(path.stat().st_size for pair in pairs for path in pair
                 if not (cache_local / path.name).is_file()
                 or (cache_local / path.name).stat().st_size != path.stat().st_size)
    # Existing completed copies do not need another 30 GB of free space.
    if shutil.disk_usage(local).free < needed + 5 * 10**9:
        raise RuntimeError("Not enough local disk for three full shards and checkpoints")
    receipt, frames = {}, []
    for shard, manifest in pairs:
        for source in (shard, manifest):
            print(f"Staging {source.name}: {source.stat().st_size / 1e9:.2f} GB", flush=True)
            receipt[source.name] = copy_verified(source, cache_local / source.name)
        array = np.load(cache_local / shard.name, mmap_mode="r")
        frame = pd.read_parquet(cache_local / manifest.name)
        if array.dtype != np.uint8 or array.shape != (len(frame), 24, 224, 224):
            raise RuntimeError(f"Wrong cache geometry or manifest length: {shard.name} {array.shape}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 24371 or not combined.SeriesInstanceUID.is_unique:
        raise RuntimeError("Cache must cover exactly 24,371 unique series")

    # Only fetch small metadata if Drive does not already contain it. Images
    # always come from the user's existing Drive shards.
    for name in ("train_labels.parquet", "side_predictions.parquet", "train.csv", "train_series.csv"):
        source = choose_metadata(root.rglob(name), name,
                                 labels_hash if name == "train_labels.parquet" else None)
        if source is None:
            import kagglehub
            from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token
            if get_kaggle_credentials() is None:
                from getpass import getpass
                token = getpass("Kaggle API token for missing metadata (hidden): ").strip()
                if not token:
                    raise RuntimeError("Enable KAGGLE_API_TOKEN in Colab secrets or provide the token")
                set_kaggle_api_token(token)
                del token
            if name == "train_labels.parquet":
                directory = kagglehub.dataset_download("pssv27/knee-labels")
                candidates = list(Path(directory).rglob(name))
                if len(candidates) != 1:
                    raise RuntimeError("Ambiguous downloaded labels")
                source = candidates[0]
            elif name == "side_predictions.parquet":
                directory = kagglehub.notebook_output_download("pssv27/knee-side-classifier")
                candidates = list(Path(directory).rglob(name))
                if len(candidates) != 1:
                    raise RuntimeError("Ambiguous downloaded side predictions")
                source = candidates[0]
            else:
                source = Path(kagglehub.competition_download(
                    "rsna-knee-abnormality-detection", path=name))
        destination = local / "metadata" if name.endswith(".csv") else cache_local
        receipt[name] = copy_verified(source, destination / name)
    if receipt["train_labels.parquet"] != labels_hash:
        raise RuntimeError("Drive label file differs from the Kaggle experiment")
    (local / "input_receipt.json").write_text(json.dumps(receipt, indent=2))
    mount_inputs(local)
    print("All three unchanged full shards and metadata are staged locally", flush=True)


def mount_inputs(local_root, inputs_root="/content/knee_slice_mounts"):
    # The corrected training finder skips roots containing train.csv to avoid
    # walking raw DICOM. Keep competition CSVs in a separate mount.
    import glob
    local, inputs = Path(local_root), Path(inputs_root)
    inputs.mkdir(parents=True, exist_ok=True)
    for name in ("cache", "metadata"):
        target, source = inputs / f"slice-colab-{name}", local / name
        if not source.is_dir():
            raise RuntimeError(f"Missing staged inputs: {source}")
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError("Conflicting input mount; use a fresh runtime")
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
    for pattern, count in (("cache_train_series_*.npy", 3),
                           ("manifest_train_series_*.parquet", 3),
                           ("train_labels.parquet", 1), ("side_predictions.parquet", 1),
                           ("train.csv", 1), ("train_series.csv", 1)):
        if len(glob.glob(f"{inputs}/**/{pattern}", recursive=True)) != count:
            raise RuntimeError(f"Duplicate or incomplete mounted {pattern}; use a fresh runtime")


class ColabInputPaths(ast.NodeTransformer):
    def __init__(self, inputs_root):
        self.inputs_root = str(inputs_root)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and (node.value == "/kaggle/input"
                                            or node.value.startswith("/kaggle/input/")):
            return ast.copy_location(ast.Constant(
                self.inputs_root + node.value[len("/kaggle/input"):]), node)
        return node


def relocate_inputs(source, inputs_root):
    tree = ColabInputPaths(inputs_root).visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def relocated_pair(source, attempt, inputs_root="/content/knee_slice_mounts"):
    """Relocate both outer runner and embedded training sources to one attempt."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "TRAINING_SCRIPTS":
            scripts = ast.literal_eval(node.value)
            node.value = ast.parse(repr({k: relocate_inputs(relocate(v, attempt), inputs_root)
                                        for k, v in scripts.items()}), mode="eval").body
        elif name == "SMOKE_SOURCE":
            node.value = ast.Constant(relocate_inputs(
                relocate(ast.literal_eval(node.value), attempt), inputs_root))
    return relocate_inputs(relocate(ast.unparse(ast.fix_missing_locations(tree)), attempt), inputs_root)


def run_pair_on_drive(pair_source, drive_root, local_root):
    import importlib
    for package in ("pydicom", "cv2", "numpy", "pandas", "pyarrow", "sklearn",
                    "tqdm", "timm", "lightning.pytorch"):
        try:
            importlib.import_module(package)
        except ImportError as exc:
            raise RuntimeError(f"Training dependency {package} failed to import; "
                               "rerun the notebook's dependency setup cell") from exc
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one Colab GPU before running")
    mount_inputs(local_root)
    root = Path(drive_root)
    mount = Path("/content/drive/MyDrive")
    if not mount.is_dir() or not root.resolve().is_relative_to(mount.resolve()):
        raise RuntimeError("Outputs must be inside mounted MyDrive")
    attempt = root / f"attempt_{uuid.uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    source = relocated_pair(pair_source, attempt)
    (attempt / "train_pair.py").write_text(source)
    shutil.copy2(Path(local_root) / "input_receipt.json", attempt / "input_receipt.json")
    (attempt / "colab_execution.json").write_text(json.dumps({
        "platform": "colab", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "pair_sha256": hashlib.sha256(pair_source.encode()).hexdigest(),
        "runtime_script_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "note": "Single GPU, effective batch 8; not bitwise equivalent to Kaggle DDP",
        "parent_timeout": None,
        "inputs_root": "/content/knee_slice_mounts",
    }, indent=2))
    print("Checkpoints, metrics, progress and logs:", attempt, flush=True)
    # Streaming wait has no training timeout. Interrupting this cell terminates
    # the child; an interrupted attempt stays on Drive and is not overwritten.
    run_process(attempt / "train_pair.py", attempt)
    progress = json.loads((attempt / "pair_progress.json").read_text())
    if progress["state"] != "complete":
        raise RuntimeError(f"Pair incomplete: {progress}")
    print("FOLD 2 PAIR COMPLETE:", attempt, flush=True)
    print((attempt / "comparison.json").read_text())
    return str(attempt)
