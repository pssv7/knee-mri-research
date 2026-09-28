"""Do not promote incomplete benchmarks or different model packages."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_protocol_submission import validate_benchmark


@pytest.fixture
def benchmark():
    expected = {f"model:{i}": f"sha{i}" for i in range(15)}
    return {"members": 15, "errors": [], "references_complete": True, "raw_complete": True,
            "runtime_gate_passed": True, "projected_1300_study_hours": 2.,
            "manifest_sha256": "manifest", "checkpoints": expected,
            "reference_checks": [{"identity": key, "studies": list(range(8)), "max_abs_delta": .0005}
                                 for key in expected], "raw_studies": [{"uid": str(i)} for i in range(100)]}, expected


def test_complete_matching_benchmark_passes(benchmark):
    report, expected = benchmark
    validate_benchmark(report, "manifest", expected)


@pytest.mark.parametrize("problem", ["fatal", "errors", "unfinished", "slow", "manifest", "checkpoint",
                                    "references", "delta", "nan", "raw_duplicate"])
def test_rejects_failed_incomplete_or_different_benchmark(benchmark, problem):
    report, expected = benchmark
    if problem == "fatal":
        report["fatal_error"] = "failure"
    elif problem == "errors":
        report["errors"] = ["failure"]
    elif problem == "unfinished":
        report["raw_complete"] = False
    elif problem == "slow":
        report["projected_1300_study_hours"] = 6.4
    elif problem == "manifest":
        report["manifest_sha256"] = "other"
    elif problem == "checkpoint":
        report["checkpoints"] = {**expected, "model:0": "other"}
    elif problem == "references":
        report["reference_checks"].pop()
    elif problem in ("delta", "nan"):
        report["reference_checks"][0]["max_abs_delta"] = .1 if problem == "delta" else float("nan")
    else:
        report["raw_studies"][0]["uid"] = "1"
    with pytest.raises(ValueError):
        validate_benchmark(report, "manifest", expected)
