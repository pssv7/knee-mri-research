"""Prepare the approved fixed blend for Kaggle; does not launch or submit."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "pssv27/knee-16-24-fixed-blend"


def prepare(root=ROOT):
    benchmark = json.loads((root / "outputs/kaggle_benchmark_results/benchmark_report.json").read_text())
    if not (benchmark["raw_complete"] and benchmark["oof_complete"] and not benchmark["errors"]):
        raise ValueError("The private benchmark must pass before staging a submission")
    upload = root / "outputs/kaggle_ensemble_upload"
    manifest_bytes = (upload / "ensemble.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {}
    for entry in manifest["members"]:
        meta = json.loads((upload / (entry["checkpoint"] + ".json")).read_text())
        if entry["weight"] != .1:
            raise ValueError("The approved blend uses ten equal 0.1 weights")
        identity = f'{meta["experiment"]["slices"]}:{meta["fold"]}'
        if identity in expected:
            raise ValueError("Duplicate slice/fold member")
        expected[identity] = meta["sha256"]
    if set(expected) != {f"{s}:{f}" for s in [16, 24] for f in range(5)}:
        raise ValueError("Expected five folds each at 16 and 24 slices")
    notebook = json.loads((root / "notebooks/10_submit_ensemble.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if len(sources) != 5:
        raise ValueError("Submission template structure changed; review before staging")
    sources[1] = "\n".join(line for line in sources[1].splitlines() if not line.startswith("!"))
    sources[1] += '\nassert device == "cuda", "GPU required"\ntorch.set_num_threads(2)\n'
    header = f'''import hashlib, json, os, pathlib, time
SUBMISSION_STARTED = time.monotonic()
manifests = list(pathlib.Path("/kaggle/input").rglob("ensemble.json"))
assert len(manifests) == 1, "Expected exactly one explicit ensemble manifest"
assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == {hashlib.sha256(manifest_bytes).hexdigest()!r}, "Ensemble manifest changed"
os.environ["KNEE_ENSEMBLE_JSON"] = str(manifests[0])
EXPECTED_CHECKPOINTS = {expected!r}
'''
    sources[2] += '''
assert len(explicit_members) == 10 and len(tuned_models) == 10
assert BLEND == 1.0 and not FLIP_TTA and not MEMBER_RANK_AVERAGE
assert np.array_equal(np.asarray(member_weights), np.full(10, .1))
actual_checkpoints = {f'{m["spec"]["slices"]}:{m["metadata"]["fold"]}': m["metadata"]["sha256"] for m in explicit_members}
assert actual_checkpoints == EXPECTED_CHECKPOINTS, "Checkpoint identities changed"
'''
    validation = '''
assert not FAST_DECODE
assert submission[config.ID_COL].tolist() == test_csv[config.ID_COL].tolist()
assert submission.columns.tolist() == config.SUBMISSION_COLUMNS
assert np.isfinite(submission[config.LABELS].to_numpy()).all()
assert ((submission[config.LABELS] >= 0) & (submission[config.LABELS] <= 1)).all().all()
report = {"rows": len(submission), "members": len(explicit_members), "fallbacks": failures,
          "checkpoints": actual_checkpoints, "weights": list(map(float, member_weights)),
          "elapsed_seconds": time.monotonic() - SUBMISSION_STARTED,
          "gpu": torch.cuda.get_device_name(), "columns": submission.columns.tolist(),
          "submission_sha256": hashlib.sha256(pathlib.Path("/kaggle/working/submission.csv").read_bytes()).hexdigest()}
pathlib.Path("/kaggle/working/submission_report.json").write_text(json.dumps(report, indent=2) + "\\n")
print(json.dumps(report, indent=2))
'''
    code = [header, *sources, validation]
    for source in code:
        ast.parse(source)
    notebook["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Fixed 16/24-slice ensemble\n", "Five selected folds per family, equal probability weights. No training or weight tuning.\n"]}]
    notebook["cells"] += [{"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": source.splitlines(keepends=True)} for source in code]
    output = root / "outputs/kaggle_submission"
    output.mkdir(parents=True, exist_ok=True)
    (output / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    receipt = json.loads((root / "outputs/kaggle_benchmark_receipt/kernel-metadata.json").read_text())
    metadata = {"id": KERNEL, "title": "Knee 16 24 Fixed Blend", "code_file": "submission.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "docker_image": receipt["docker_image"], "dataset_sources": receipt["dataset_sources"],
                "kernel_sources": ["pssv27/knee-side-classifier"],
                "competition_sources": receipt["competition_sources"]}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Staged {output}; no remote run or submission was started")


if __name__ == "__main__":
    prepare()
