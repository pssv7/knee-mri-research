"""Stage a private, offline, inference-only Kaggle benchmark. Does not upload."""

import ast
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(root=ROOT):
    package = root / "outputs/ensemble_16_24"
    upload = root / "outputs/kaggle_ensemble_upload"
    kernel = root / "outputs/kaggle_benchmark"
    upload.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "ensemble.json").read_text())
    for member in manifest["members"]:
        source = package / member["checkpoint"]
        filename = "_".join(source.relative_to(package).parts)
        for suffix in ["", ".json"]:
            dest = upload / (filename + suffix)
            if not dest.exists():
                os.link(str(source) + suffix, dest)
        member["checkpoint"] = filename
    (upload / "ensemble.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for filename in ["provenance.json", "reconstructed_folds.parquet"]:
        shutil.copy2(package / filename, upload / filename)
    shutil.copy2(root / "experiment /slice_24/fold0/oof_fold0.parquet", upload / "reference_24_fold0.parquet")
    shutil.copy2(root / "experiment /oof_fold1.parquet", upload / "candidate_oof_fold1.parquet")
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": "pssv27/knee-ensemble-16-24-private", "title": "Knee Ensemble 16 24 Private",
        "licenses": [{"name": "other"}],
        "description": "Private selected-checkpoint ensemble for inference benchmarking. No new competition score."
    }, indent=2) + "\n")
    notebook = json.loads((root / "notebooks/10_submit_ensemble.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    bootstrap, setup, models, inference, _ = sources
    setup = "\n".join(line for line in setup.splitlines() if not line.startswith("!"))
    inference = inference.split("ids, frozen_rows, tuned_rows, failures = [], [], [], 0")[0]
    header = '''import json, os, time, threading, pathlib
RUN_STARTED = time.monotonic()
def hard_stop():
    print("Benchmark watchdog: stopping before the one-hour limit", flush=True)
    os._exit(124)
watchdog = threading.Timer(3500, hard_stop)
watchdog.daemon = True
watchdog.start()
matches = list(pathlib.Path("/kaggle/input").rglob("ensemble.json"))
assert len(matches) == 1, f"Expected one explicit ensemble, found {matches}"
os.environ["KNEE_ENSEMBLE_JSON"] = str(matches[0])
'''
    source = "\n\n".join([header, bootstrap, setup,
        'assert device == "cuda", "A GPU is required; do not spend the hour on CPU"\ntorch.set_num_threads(2)',
        models, inference, (root / "tools/kaggle_benchmark_body.py").read_text()])
    ast.parse(source)
    (kernel / "benchmark.py").write_text(source)
    metadata = json.loads((root / "outputs/kaggle_sources/knee-finetune/kernel-metadata.json").read_text())
    metadata = {key: metadata[key] for key in ["competition_sources", "kernel_sources", "docker_image"]}
    metadata.update({"id": "pssv27/knee-ensemble-private-benchmark", "title": "Knee Ensemble Private Benchmark",
        "code_file": "benchmark.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["pssv27/knee-labels", "pssv27/knee-ensemble-16-24-private"]})
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Upload folder: {upload}\nKernel folder: {kernel}")


if __name__ == "__main__":
    prepare()
