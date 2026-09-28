"""Package and stage the fixed 15-model protocol benchmark; never upload or launch."""

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.model import artifacts
from tools.compare_protocol_incumbent import completed_protocol_directory

DATASET = "pssv27/knee-ensemble-protocol-private"
KERNEL = "pssv27/knee-protocol-ensemble-benchmark"

INPUT_SEARCH = '''
def attached_files(pattern):
    found = []
    for root in sorted(pathlib.Path("/kaggle/input").iterdir()):
        if (root.name in ("competitions", "rsna-knee-abnormality-detection")
                or (root / "train.csv").exists() or (root / "train_series.csv").exists()):
            continue
        found.extend(root.rglob(pattern))
    return sorted(found)
'''


def link_verified(source, dest):
    """Reuse immutable weights without duplicating storage or accepting stale files."""
    if dest.exists():
        if artifacts.sha256(source) != artifacts.sha256(dest):
            raise ValueError(f"Existing package file changed: {dest}")
    else:
        dest.hardlink_to(source)


def prepare(root=ROOT):
    root = Path(root)
    incumbent = root / "outputs/ensemble_16_24"
    report_path = root / "data/meta/compare_protocol_incumbent_all_folds/summary.json"
    report = json.loads(report_path.read_text())
    if (report["folds"] != list(range(5)) or report["shared_studies"] != 4349
            or report["baseline_only"] or report["candidate_only"]
            or any(v["candidate"] <= v["baseline"] for v in report["per_fold_macro"].values())
            or report["paired_macro_delta_95ci"]["candidate"][0] <= 0):
        raise ValueError("Full five-fold fixed-family comparison must pass")
    members = json.loads((incumbent / "ensemble.json").read_text())["members"]
    if len(members) != 10 or any(m["weight"] != .1 for m in members):
        raise ValueError("Incumbent must contain ten equal selected models")
    entries = []
    for member in members:
        source = incumbent / member["checkpoint"]
        meta = artifacts.load_metadata(source)
        entries.append((source, meta, "slices_" + str(meta["experiment"]["slices"])))
    for fold in range(5):
        source = completed_protocol_directory(root, fold) / f"convnext_fold{fold}.pt"
        entries.append((source, artifacts.load_metadata(source), "protocol"))
    expected = {(family, fold) for family in ("slices_16", "slices_24", "protocol") for fold in range(5)}
    if len(entries) != 15 or {(family, meta["fold"]) for _, meta, family in entries} != expected:
        raise ValueError("Expected five selected folds per family")
    receipts = {(r["family"], r["fold"]): r for r in report["source_receipts"]}
    # Validate every input before constructing the package; the comparison
    # binds checkpoint and OOF hashes to the exact equal-family result.
    for source, meta, family in entries:
        receipt = receipts[family, meta["fold"]]
        if meta["sha256"] != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from full OOF comparison")
        if artifacts.sha256(root / receipt["oof"]) != receipt["oof_sha256"]:
            raise ValueError("OOF predictions changed since comparison")
    output = root / "outputs/ensemble_16_24_protocol"
    upload = root / "outputs/kaggle_protocol_ensemble_upload"
    kernel = root / "outputs/kaggle_protocol_ensemble_benchmark"
    for path in (output, upload, kernel):
        path.mkdir(parents=True, exist_ok=True)
    manifest, provenance, expected_hashes = {"members": []}, [], {}
    for source, meta, family in entries:
        filename = f"{family}_fold{meta['fold']}.pt"
        reference = f"reference_{family}_fold{meta['fold']}.parquet"
        for suffix in ("", ".json"):
            link_verified(Path(str(source) + suffix), output / (filename + suffix))
            link_verified(output / (filename + suffix), upload / (filename + suffix))
        receipt = receipts[family, meta["fold"]]
        link_verified(root / receipt["oof"], upload / reference)
        manifest["members"].append({"checkpoint": filename, "weight": 1 / 15})
        identity = f"{meta['experiment']['name']}:{meta['fold']}"
        if identity in expected_hashes:
            raise ValueError("Duplicate model identity")
        expected_hashes[identity] = {"sha256": meta["sha256"], "reference": reference}
        provenance.append({**receipt, "source": str(source.relative_to(root)), "checkpoint": filename})
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    for directory in (output, upload):
        (directory / "ensemble.json").write_bytes(manifest_bytes)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        shutil.copy2(incumbent / "reconstructed_folds.parquet", directory / "reconstructed_folds.parquet")
        shutil.copy2(report_path, directory / "oof_comparison.json")
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET, "title": "Knee Ensemble Protocol Private", "licenses": [{"name": "other"}],
        "description": "Fixed selected 16/24/protocol ensemble and held-out reference predictions for private inference verification."
    }, indent=2) + "\n")
    notebook_path = root / "notebooks/10_submit_ensemble.ipynb"
    notebook = json.loads(notebook_path.read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if len(sources) != 5:
        raise ValueError("Submission notebook structure changed")
    bootstrap, setup, models, inference, _ = sources
    setup = "\n".join(line for line in setup.splitlines() if not line.startswith("!"))
    tree = ast.parse(setup)
    finders = [i for i, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef) and node.name == "find_inputs"]
    if len(finders) != 1:
        raise ValueError("Submission input finder changed")
    tree.body[finders[0]] = ast.parse('def find_inputs(pattern):\n    return [str(p) for p in attached_files(pattern)]').body[0]
    setup = ast.unparse(tree)
    marker = "ids, frozen_rows, tuned_rows, failures = [], [], [], 0"
    if inference.count(marker) != 1:
        raise ValueError("Submission inference boundary changed")
    inference = inference.split(marker)[0]
    header = '''import hashlib, json, os, time, threading, pathlib
RUN_STARTED = time.monotonic()
def hard_stop():
    print("Benchmark wall-time limit reached", flush=True)
    os._exit(124)
watchdog = threading.Timer(3500, hard_stop)
watchdog.daemon = True
watchdog.start()
'''
    header += INPUT_SEARCH
    header += f'''
manifests = attached_files("ensemble.json")
assert len(manifests) == 1, "Expected exactly one ensemble manifest"
assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == {hashlib.sha256(manifest_bytes).hexdigest()!r}
os.environ["KNEE_ENSEMBLE_JSON"] = str(manifests[0])
EXPECTED_MODELS = {expected_hashes!r}
'''
    source = "\n\n".join((header, bootstrap, setup,
        'assert device == "cuda", "GPU required"\ntorch.set_num_threads(2)', models, inference,
        (root / "tools/protocol_benchmark_body.py").read_text()))
    ast.parse(source)
    (kernel / "benchmark.py").write_text(source)
    base = json.loads((root / "outputs/kaggle_sources/knee-finetune/kernel-metadata.json").read_text())
    metadata = {key: base[key] for key in ("competition_sources", "kernel_sources", "docker_image")}
    metadata.update(id=KERNEL, title="Knee Protocol Ensemble Benchmark", code_file="benchmark.py",
        language="python", kernel_type="script", is_private=True, enable_gpu=True, enable_internet=False,
        machine_shape="NvidiaTeslaT4", dataset_sources=["pssv27/knee-labels", DATASET])
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"state": "staged_not_launched", "members": expected_hashes,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "notebook_sha256": artifacts.sha256(notebook_path), "dataset": DATASET, "kernel": KERNEL}
    (kernel / "staging_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Packaged {len(entries)} selected models: {output}\nUpload: {upload}\nBenchmark: {kernel}")


if __name__ == "__main__":
    prepare()
