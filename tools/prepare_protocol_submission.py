"""Stage the benchmarked fixed protocol ensemble; never launch or submit."""

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import artifacts
from tools.prepare_protocol_benchmark import INPUT_SEARCH


def validate_benchmark(report, manifest_hash, expected):
    if (report.get("fatal_error") or report.get("errors") or report.get("members") != 15
            or not report.get("references_complete") or not report.get("raw_complete")
            or not report.get("runtime_gate_passed")
            or not 0 < report.get("projected_1300_study_hours", 0) < 6.3):
        raise ValueError("A complete passing 15-model benchmark is required")
    if report.get("manifest_sha256") != manifest_hash or report.get("checkpoints") != expected:
        raise ValueError("Benchmark did not use this exact ensemble")
    checks = report.get("reference_checks", [])
    if (len(checks) != 15 or {c["identity"] for c in checks} != set(expected)
            or any(len(c["studies"]) != 8 or not 0 <= c["max_abs_delta"] < .01 for c in checks)):
        raise ValueError("Incomplete or failed reference reproduction")
    raw = report.get("raw_studies", [])
    if len(raw) != 100 or len({r["uid"] for r in raw}) != 100:
        raise ValueError("Expected 100 unique timed studies")


def prepare(root=ROOT):
    root = Path(root)
    upload = root / "outputs/kaggle_protocol_ensemble_upload"
    staging = root / "outputs/kaggle_protocol_ensemble_benchmark"
    staged = json.loads((staging / "staging_receipt.json").read_text())
    manifest_bytes = (upload / "ensemble.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected = {}
    members = json.loads(manifest_bytes)["members"]
    if len(members) != 15 or any(m["weight"] != 1 / 15 for m in members):
        raise ValueError("Expected fifteen equal selected members")
    for member in members:
        meta = artifacts.load_metadata(upload / member["checkpoint"])
        identity = f"{meta['experiment']['name']}:{meta['fold']}"
        if identity in expected:
            raise ValueError("Duplicate family/fold")
        expected[identity] = meta["sha256"]
    if manifest_hash != staged["manifest_sha256"] or expected != {k: v["sha256"] for k, v in staged["members"].items()}:
        raise ValueError("Package changed since benchmark staging")
    report = json.loads((root / "outputs/kaggle_protocol_ensemble_benchmark_results/benchmark_report.json").read_text())
    validate_benchmark(report, manifest_hash, expected)
    notebook_path = root / "notebooks/10_submit_ensemble.ipynb"
    if artifacts.sha256(notebook_path) != staged["notebook_sha256"]:
        raise ValueError("Inference source changed since benchmark")
    notebook = json.loads(notebook_path.read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if len(sources) != 5:
        raise ValueError("Submission template changed")
    setup = ast.parse("\n".join(line for line in sources[1].splitlines() if not line.startswith("!")))
    finders = [i for i, node in enumerate(setup.body) if isinstance(node, ast.FunctionDef) and node.name == "find_inputs"]
    if len(finders) != 1:
        raise ValueError("Input finder changed")
    setup.body[finders[0]] = ast.parse('def find_inputs(pattern):\n    return [str(p) for p in attached_files(pattern)]').body[0]
    sources[1] = ast.unparse(setup) + '\nassert device == "cuda"\ntorch.set_num_threads(2)\n'
    header = 'import hashlib, json, os, pathlib, time\nSUBMISSION_STARTED = time.monotonic()\n' + INPUT_SEARCH
    header += f'''
manifests = attached_files("ensemble.json")
assert len(manifests) == 1
assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == {manifest_hash!r}
os.environ["KNEE_ENSEMBLE_JSON"] = str(manifests[0])
EXPECTED_CHECKPOINTS = {expected!r}
'''
    sources[2] += '''
assert len(explicit_members) == len(tuned_models) == 15
assert BLEND == 1.0 and not FLIP_TTA and not SOUP and not MEMBER_RANK_AVERAGE
assert np.array_equal(np.asarray(member_weights), np.full(15, 1 / 15))
actual_checkpoints = {f"{m['spec']['name']}:{m['metadata']['fold']}": m['metadata']['sha256'] for m in explicit_members}
assert actual_checkpoints == EXPECTED_CHECKPOINTS
'''
    validation = '''
assert not FAST_DECODE
assert submission[config.ID_COL].tolist() == test_csv[config.ID_COL].tolist()
assert submission.columns.tolist() == config.SUBMISSION_COLUMNS
assert np.isfinite(submission[config.LABELS].to_numpy()).all()
report = {"rows": len(submission), "members": len(explicit_members), "fallbacks": failures,
          "checkpoints": actual_checkpoints, "weights": list(map(float, member_weights)),
          "elapsed_seconds": time.monotonic() - SUBMISSION_STARTED,
          "gpu": torch.cuda.get_device_name(), "columns": submission.columns.tolist(),
          "submission_sha256": hashlib.sha256(pathlib.Path("/kaggle/working/submission.csv").read_bytes()).hexdigest()}
pathlib.Path("/kaggle/working/submission_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''
    code = [header, *sources, validation]
    for source in code:
        ast.parse(source)
    notebook["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Fixed 16/24/protocol ensemble\n", "Five selected models per family, equal probability weights.\n"]}]
    notebook["cells"] += [{"cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": source.splitlines(keepends=True)} for source in code]
    output = root / "outputs/kaggle_protocol_submission"
    output.mkdir(parents=True, exist_ok=True)
    (output / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    receipt = json.loads((root / "outputs/kaggle_protocol_ensemble_benchmark_receipt/kernel-metadata.json").read_text())
    metadata = {"id": "pssv27/knee-16-24-protocol-fixed-blend", "title": "Knee 16 24 Protocol Fixed Blend",
        "code_file": "submission.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": receipt["docker_image"], "dataset_sources": receipt["dataset_sources"],
        "kernel_sources": ["pssv27/knee-side-classifier"], "competition_sources": receipt["competition_sources"]}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Staged {output}; no remote run or competition submission")


if __name__ == "__main__":
    prepare()
