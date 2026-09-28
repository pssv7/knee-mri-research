"""Colab changes execution environment, never the screened training recipe."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.model.experiments import get_experiment
from tools.colab_protocol_runtime import (
    check_colab_contract, completed_run, relocate, required_files, sha256,
)
from tools.prepare_colab_protocol_completion import adapt, build, STAGE, VALIDATE
from test_protocol_completion import staged_root


def embedded_files(source):
    return next(ast.literal_eval(node.value) for node in ast.parse(source).body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "FILES")


def test_build_preserves_modules_and_geometry_and_only_changes_platform_guard(staged_root):
    path = build(staged_root)
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("\n".join(line for line in cell["source"].splitlines()
                                if not line.startswith("%")))
    payload = next(cell["source"] for cell in notebook["cells"]
                   if cell["source"].startswith("RECIPE_SHA256"))
    scripts = ast.literal_eval(ast.parse(payload).body[1].value)
    assert set(scripts) == {0, 2, 4}
    original = json.loads((staged_root / "notebooks/08_finetune.ipynb").read_text())
    original_source = "".join([cell for cell in original["cells"]
                               if cell["cell_type"] == "code"][1]["source"])
    for fold, jobs in scripts.items():
        assert list(jobs) == ["ddp_smoke", f"protocol_fold{fold}"]
        for name, source in jobs.items():
            moved = relocate(source, "/content/drive/MyDrive/fixture")
            assert embedded_files(moved) == embedded_files(original_source)
            assert "BATCH_STUDIES * DEVICES * ACCUM_STEPS == 8" in moved
            assert "check_colab_contract(_contract, REFERENCE_CONTRACT)" in moved
            assert moved.index("check_colab_contract(_contract") < moved.index("trainer.fit(")
            assert "Completion settings or initialization differ" not in moved
            constants = [node.value for node in ast.walk(ast.parse(moved))
                         if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            assert not any(value.startswith("/kaggle/working") for value in constants)
            assert ("COLAB_ARTIFACT_VALIDATION_PASSED" in moved) == (name != "ddp_smoke")
            assert "ckpt_path=" not in moved


def test_only_explicit_hardware_and_fold_contract_changes_are_allowed():
    reference = {"devices": 2, "torch": "screened", "fold": 3, "steps": 5220,
                 "warmup_steps": 1566, "effective_batch": 8, "initial_weights_sha256": "same",
                 "timm": "1.0.26", "lightning": "2.6.5", "seed": 0,
                 "experiment": get_experiment("protocol").to_dict()}
    current = dict(reference, devices=1, torch="colab", fold=0)
    assert set(check_colab_contract(current, reference)) == {"devices", "torch", "fold"}
    for key, value in (("effective_batch", 16), ("initial_weights_sha256", "different"),
                       ("timm", "new"), ("lightning", "new"), ("seed", 1),
                       ("devices", 2), ("experiment", get_experiment("baseline").to_dict())):
        with pytest.raises(ValueError, match="contract mismatch"):
            check_colab_contract(dict(current, **{key: value}), reference)


def test_completion_receipt_checks_every_artifact_and_rejects_corruption(tmp_path):
    assert completed_run(tmp_path, 0, "recipe") is None
    run = tmp_path / "attempt_fixture/protocol_fold0"
    run.mkdir(parents=True)
    for name in required_files(0):
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
    record = {"fold": 0, "attempt": "attempt_fixture", "recipe_sha256": "recipe",
              "sha256": {name: sha256(run / name) for name in required_files(0)}}
    (tmp_path / "complete.json").write_text(json.dumps(record))
    assert completed_run(tmp_path, 0, "recipe") == run
    with pytest.raises(ValueError, match="another recipe"):
        completed_run(tmp_path, 0, "changed")
    (run / "avg_fold0_swa.pt.json").write_text("corrupt")
    with pytest.raises(ValueError, match="artifact changed"):
        completed_run(tmp_path, 0, "recipe")


def test_guard_structure_drift_fails_closed():
    with pytest.raises(ValueError, match="guard changed"):
        adapt("x = 1")


def test_stage_fetches_all_shards_and_only_small_competition_metadata():
    ast.parse(STAGE)
    assert "for number in (1, 2, 3):" in STAGE
    assert "if len(paths) != expected:" in STAGE
    assert 'for name in ("train.csv", "train_series.csv"):' in STAGE
    assert 'competition_download("rsna-knee-abnormality-detection", path=name)' in STAGE
    assert 'os.environ["KAGGLEHUB_CACHE"] = "/content/kagglehub"' in STAGE


def test_artifact_validation_accepts_existing_protocol_result(monkeypatch):
    directory = ROOT / "outputs/kaggle_protocol_fold3_results/protocol_fold3"
    if not directory.exists():
        pytest.skip("Downloaded fold-3 artifacts unavailable")
    monkeypatch.chdir(directory)
    gold = pd.read_parquet(directory / "gold_predictions_fold3.parquet")
    namespace = {"pathlib": __import__("pathlib"), "pd": pd, "np": np,
                 "config": config, "FOLD": 3, "EXPECTED_VAL": 869,
                 "RUN": get_experiment("protocol"),
                 "gold_index": SimpleNamespace(studies=list(gold[config.ID_COL]))}
    exec(VALIDATE, namespace)
