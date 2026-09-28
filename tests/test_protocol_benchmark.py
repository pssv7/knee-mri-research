"""Exercise the fixed package, guarded input discovery and GPU benchmark orchestration."""

import ast
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.model import artifacts, dataset
from tools.prepare_protocol_benchmark import INPUT_SEARCH, prepare


def test_package_preserves_incumbent_and_binds_all_fifteen_members(tmp_path):
    incumbent = tmp_path / "outputs/ensemble_16_24"
    incumbent.mkdir(parents=True)
    members, receipts = [], []
    for family in ("slices_16", "slices_24", "protocol"):
        for fold in range(5):
            if family == "protocol":
                suffix = "results" if fold in (1, 3) else "comparison_download"
                result = tmp_path / f"outputs/kaggle_protocol_fold{fold}_{suffix}"
                directory = result / f"protocol_fold{fold}"
                directory.mkdir(parents=True)
                name = "pair_progress.json" if fold in (1, 3) else "completion_progress.json"
                (result / name).write_text(json.dumps({"state": "complete", "fold": fold,
                    "completed": [f"protocol_fold{fold}"]}))
                source = directory / f"convnext_fold{fold}.pt"
            else:
                source = incumbent / f"{family}_fold{fold}.pt"
                members.append({"checkpoint": source.name, "weight": .1})
            source.write_bytes(f"{family}:{fold}".encode())
            spec = {"name": family, "slices": 24 if family == "slices_24" else 16,
                    "max_series": 5, "series_selection": "protocol" if family == "protocol" else "uniform"}
            artifacts.write_metadata(source, spec, fold, [], [])
            oof = source.with_suffix(".parquet")
            oof.write_bytes(b"fixture OOF")
            receipts.append({"family": family, "fold": fold, "checkpoint_sha256": artifacts.sha256(source),
                "oof": str(oof.relative_to(tmp_path)), "oof_sha256": artifacts.sha256(oof)})
    manifest_bytes = json.dumps({"members": members}).encode()
    (incumbent / "ensemble.json").write_bytes(manifest_bytes)
    (incumbent / "reconstructed_folds.parquet").write_bytes(b"fixture folds")
    report = tmp_path / "data/meta/compare_protocol_incumbent_all_folds/summary.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"folds": list(range(5)), "shared_studies": 4349,
        "baseline_only": 0, "candidate_only": 0,
        "per_fold_macro": {str(f): {"baseline": .8, "candidate": .81} for f in range(5)},
        "paired_macro_delta_95ci": {"candidate": [.005, .015]}, "source_receipts": receipts}))
    for name in ("notebooks/10_submit_ensemble.ipynb", "tools/protocol_benchmark_body.py",
                 "outputs/kaggle_sources/knee-finetune/kernel-metadata.json"):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    prepare(tmp_path)
    prepare(tmp_path)
    assert (incumbent / "ensemble.json").read_bytes() == manifest_bytes
    upload = tmp_path / "outputs/kaggle_protocol_ensemble_upload"
    manifest = json.loads((upload / "ensemble.json").read_text())
    assert len(manifest["members"]) == 15
    assert all(m["weight"] == 1 / 15 for m in manifest["members"])
    assert len(list(upload.glob("*.pt"))) == 15 and len(list(upload.glob("reference_*.parquet"))) == 15
    kernel = tmp_path / "outputs/kaggle_protocol_ensemble_benchmark"
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"]
    source = (kernel / "benchmark.py").read_text()
    ast.parse(source)
    assert "competition_submit(" not in source and "submission.to_csv" not in source
    (tmp_path / receipts[0]["oof"]).write_bytes(b"changed OOF")
    with pytest.raises(ValueError, match="OOF predictions changed"):
        prepare(tmp_path)


def test_search_never_recurses_into_raw_competition(tmp_path, monkeypatch):
    for name in ("competitions", "rsna-knee-abnormality-detection", "unusual-mount", "datasets/user/package"):
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "ensemble.json").touch()
    (tmp_path / "unusual-mount/train_series.csv").touch()
    original = Path.rglob
    def guarded(path, pattern):
        assert path.name not in {"competitions", "rsna-knee-abnormality-detection", "unusual-mount"}
        return original(path, pattern)
    monkeypatch.setattr(Path, "rglob", guarded)
    namespace = {"pathlib": SimpleNamespace(Path=lambda p: tmp_path)}
    exec(INPUT_SEARCH, namespace)
    assert namespace["attached_files"]("ensemble.json") == [tmp_path / "datasets/user/package/ensemble.json"]


@pytest.mark.parametrize("bad_reference", [False, True])
def test_benchmark_checks_every_member_before_timing_raw_studies(tmp_path, monkeypatch, bad_reference):
    ids = [f"study{i:04}" for i in range(4349)]
    series = pd.DataFrame({config.ID_COL: ids, "SeriesInstanceUID": ids,
        "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"})
    labels = pd.DataFrame({config.ID_COL: ids, **{name: .5 for name in config.LABELS}})
    folded = pd.DataFrame({config.ID_COL: ids, "fold": np.arange(len(ids)) % 5})
    members, expected = [], {}
    files = {"side_predictions.parquet": series[["SeriesInstanceUID"]].assign(side_predicted="R"),
             "reconstructed_folds.parquet": folded}
    for family in ("legacy_16slice", "legacy_24slice", "protocol"):
        for fold in range(5):
            identity = f"{family}:{fold}"
            expected[identity] = {"sha256": identity, "reference": identity}
            members.append({"spec": {"name": family}, "metadata": {"fold": fold, "sha256": identity}})
            files[identity] = labels.loc[folded.fold == fold].copy()
    if bad_reference:
        files["protocol:4"][config.LABELS] = .9
    for i in range(3):
        files[f"manifest_train_series_{i}.parquet"] = series.iloc[i::3][[config.ID_COL, "SeriesInstanceUID"]].assign(position=0)
    original = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p: files[str(p)].copy() if str(p) in files else original(p))
    monkeypatch.setattr(pd, "read_csv", lambda p: series.copy())
    monkeypatch.setattr(np, "load", lambda *a, **kw: np.zeros((1, 2, 8, 8), dtype=np.uint8))
    cuda = SimpleNamespace(get_device_name=lambda: "fixture", max_memory_allocated=lambda: 1024,
        max_memory_reserved=lambda: 2048, reset_peak_memory_stats=lambda: None, synchronize=lambda: None)
    manifest = tmp_path / "ensemble.json"
    manifest.write_text("{}")
    raw = []
    def build(uid, fast):
        raw.append(uid)
        return [np.zeros((2, 8, 8))], [uid], [0]
    namespace = {"pathlib": SimpleNamespace(Path=lambda p: tmp_path if p == "/kaggle/working" else Path(p)),
        "torch": SimpleNamespace(cuda=cuda), "json": json, "hashlib": hashlib, "time": time, "pd": pd, "np": np,
        "RUN_STARTED": time.monotonic(), "ENSEMBLE_JSON": manifest, "explicit_members": members,
        "EXPECTED_MODELS": expected, "member_weights": [1 / 15] * 15,
        "knee_dataset": dataset, "train_labels": labels, "config": config, "device": "cuda",
        "find_one": lambda p: p, "find_competition_file": lambda p: p,
        "find_inputs": lambda p: [f"cache_train_series_{i}.npy" for i in range(3)],
        "splits": SimpleNamespace(load_gold_uids=lambda p: set(), assert_no_gold_leakage=lambda *a: None),
        "artifacts": SimpleNamespace(predict_members=lambda *a: np.full((1, 12), .5)),
        "build_volumes": build, "predict_side": lambda *a, **kw: "R",
        "tuned_predict": lambda *a, **kw: np.full((15, 12), .5), "watchdog": SimpleNamespace(cancel=lambda: None)}
    source = (ROOT / "tools/protocol_benchmark_body.py").read_text()
    if bad_reference:
        with pytest.raises(AssertionError, match="OOF reproduction failed for protocol:4"):
            exec(compile(source, "benchmark", "exec"), namespace)
        assert not raw
    else:
        exec(compile(source, "benchmark", "exec"), namespace)
        report = json.loads((tmp_path / "benchmark_report.json").read_text())
        assert report["references_complete"] and report["raw_complete"] and report["runtime_gate_passed"]
        assert len(report["reference_checks"]) == 15 and len(raw) == len(set(raw)) == 100
        assert not list(tmp_path.glob("*submission*"))
