"""Check the private benchmark package without contacting Kaggle."""

import ast
import json
import shutil
import sys
import time
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_kaggle_benchmark import ROOT, prepare


def test_staging_is_private_offline_and_does_not_submit(tmp_path):
    package = tmp_path / "outputs/ensemble_16_24"
    (package / "slices_16").mkdir(parents=True)
    (package / "slices_16/model.pt").write_bytes(b"fixture")
    (package / "slices_16/model.pt.json").write_text("{}")
    (package / "ensemble.json").write_text(json.dumps({
        "members": [{"checkpoint": "slices_16/model.pt", "weight": 1}]}))
    for name in ["provenance.json", "reconstructed_folds.parquet"]:
        (package / name).write_bytes(b"fixture")
    for name in ["experiment /slice_24/fold0/oof_fold0.parquet", "experiment /oof_fold1.parquet"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    for name in ["notebooks/10_submit_ensemble.ipynb", "tools/kaggle_benchmark_body.py",
                 "outputs/kaggle_sources/knee-finetune/kernel-metadata.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("kernel-metadata.json"):
            path.write_text(json.dumps({"competition_sources": ["competition"],
                "kernel_sources": ["cache"], "docker_image": "image"}))
        else:
            shutil.copy2(ROOT / name, path)
    prepare(tmp_path)
    prepare(tmp_path)
    kernel = tmp_path / "outputs/kaggle_benchmark"
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    source = (kernel / "benchmark.py").read_text()
    ast.parse(source)
    assert "threading.Timer(3500, hard_stop)" in source
    assert "competition_submit(" not in source and "to_csv(" not in source
    assert "for study in tqdm(studies" not in source
    upload = tmp_path / "outputs/kaggle_ensemble_upload"
    manifest = json.loads((upload / "ensemble.json").read_text())
    assert manifest["members"][0]["checkpoint"] == "slices_16_model.pt"
    assert (upload / "slices_16_model.pt.json").is_file()


def test_benchmark_body_recovery_and_raw_loop(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from src import config
    from src.model import dataset

    ids = [f"study{i:03}" for i in range(100)]
    series = pd.DataFrame({config.ID_COL: ids, "SeriesInstanceUID": [f"s{i}" for i in range(100)],
        "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"})
    labels = pd.DataFrame({config.ID_COL: ids, **{name: .5 for name in config.LABELS}})
    folds = pd.DataFrame({config.ID_COL: ids, "fold": [i % 2 for i in range(100)]})
    files = {"side_predictions.parquet": series[["SeriesInstanceUID"]].assign(side_predicted="R"),
        "reconstructed_folds.parquet": folds,
        "reference_24_fold0.parquet": labels.iloc[::2],
        "candidate_oof_fold1.parquet": labels.iloc[1::2].assign(fold=1)}
    for i in range(3):
        files[f"manifest_train_series_{i}.parquet"] = series.iloc[i::3][[config.ID_COL, "SeriesInstanceUID"]].assign(position=0)
    original_read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p: files[str(p)].copy() if str(p) in files else original_read(p))
    monkeypatch.setattr(pd, "read_csv", lambda p: series.copy())
    monkeypatch.setattr(np, "load", lambda *a, **kw: np.zeros((1, 2, 8, 8), dtype=np.uint8))
    members = [{"spec": {"slices": slices}, "metadata": {"fold": fold}}
               for slices in [16, 24] for fold in range(5)]
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: "fixture GPU",
        max_memory_allocated=lambda: 1024, reset_peak_memory_stats=lambda: None, synchronize=lambda: None))
    namespace = {"pathlib": SimpleNamespace(Path=lambda p: tmp_path if p == "/kaggle/working" else Path(p)),
        "torch": fake_torch, "json": json, "time": time, "pd": pd, "np": np,
        "RUN_STARTED": time.monotonic(), "ENSEMBLE_JSON": "fixture.json", "explicit_members": members,
        "knee_dataset": dataset, "train_labels": labels, "config": config, "device": "cuda",
        "find_one": lambda p: p, "find_competition_file": lambda p: p,
        "find_inputs": lambda p: [f"cache_train_series_{i}.npy" for i in range(3)],
        "splits": SimpleNamespace(load_gold_uids=lambda p: set(), assert_no_gold_leakage=lambda *a: None),
        "artifacts": SimpleNamespace(predict_members=lambda *a: np.full((1, 12), .5)),
        "build_volumes": lambda uid, fast: ([np.zeros((2, 8, 8))], [uid], [0]),
        "predict_side": lambda *a, **kw: "R", "tuned_predict": lambda *a, **kw: np.full((10, 12), .5),
        "member_weights": [.1] * 10, "watchdog": SimpleNamespace(cancel=lambda: None)}
    exec(compile((ROOT / "tools/kaggle_benchmark_body.py").read_text(), "benchmark", "exec"), namespace)
    report = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert report["raw_complete"] and report["oof_complete"]
    assert report["candidate_fold1_matches_within_0_01"]
    assert report["oof_studies"] == 50 and len(report["raw_studies"]) == 100
    assert not list(tmp_path.glob("*submission*"))
