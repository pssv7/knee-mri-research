"""The comparison path that the fold-1 pair died on, exercised on fixtures."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from tools import recover_reviewed_comparison as recovery
from tools.compare_oof import find_arm_file

CORRECTIONS = [{"StudyInstanceUID": "s0", "finding": "ACL",
                "old_target": .9, "new_target": 0.}]
CORRECTIONS_SHA = hashlib.sha256(
    json.dumps(CORRECTIONS, sort_keys=True).encode()).hexdigest()
RUN_CONTRACT = {"experiment": {"name": "baseline", "slices": 16, "max_series": 5,
                               "series_selection": "protocol"},
                "fold": 1, "epochs": 12, "seed": 0}


def write_arm(root, arm, *, studies, labels_sha, fold=1, corrections=None,
              run=None, contract_dir=None, artifact_dir=None):
    """Lay an arm out the way the failed run did: contracts and weights apart."""
    contracts = root / (contract_dir or f"{arm}_fold{fold}")
    outputs = root / (artifact_dir or f"reviewed_{arm}_fold{fold}")
    for directory in (contracts, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    review = {"arm": arm, "corrections_sha256": CORRECTIONS_SHA,
              "original_labels_sha256": labels_sha,
              "correction_count": len(CORRECTIONS) if arm == "candidate" else 0,
              "validation_labels_changed": False, "folds_changed": False}
    review.update(corrections or {})
    (contracts / "label_review_contract.json").write_text(json.dumps(review))
    (contracts / "run_contract.json").write_text(
        json.dumps({**RUN_CONTRACT, **(run or {})}))

    checkpoint = outputs / f"convnext_fold{fold}.pt"
    checkpoint.write_bytes(arm.encode())
    (outputs / f"convnext_fold{fold}.pt.json").write_text(json.dumps({
        "schema_version": 1, "labels": config.LABELS,
        "sha256": hashlib.sha256(arm.encode()).hexdigest(),
        "experiment": RUN_CONTRACT["experiment"],
        "cache_size": config.CACHE_SIZE, "cache_slices": config.CACHE_SLICES,
        "fold": fold, "train_uids": ["train0"], "val_uids": sorted(studies)}))

    folds = pd.DataFrame({config.ID_COL: studies, "fold": fold})
    folds.to_parquet(outputs / "folds.parquet", index=False)
    rng = np.random.default_rng(abs(hash(arm)) % 2**32)
    frame = pd.DataFrame({config.ID_COL: studies, "fold": fold})
    for name in config.LABELS:
        frame[name] = rng.uniform(size=len(studies))
    frame.to_parquet(outputs / f"oof_fold{fold}.parquet", index=False)
    return contracts, outputs


@pytest.fixture
def pair(tmp_path):
    """A complete fold-1 pair, plus the staged inputs the recovery reads."""
    studies = [f"s{i}" for i in range(40)]
    rng = np.random.default_rng(0)
    labels = pd.DataFrame({config.ID_COL: studies})
    for name in config.LABELS:
        labels[name] = rng.integers(0, 2, len(studies)).astype(float)
    labels_path = tmp_path / "train_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    labels_sha = recovery.sha256(labels_path)

    root = tmp_path / "run"
    for arm in recovery.ARMS:
        write_arm(root, arm, studies=studies, labels_sha=labels_sha)

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "corrections.json").write_text(json.dumps(CORRECTIONS))
    (staged / "receipt.json").write_text(json.dumps(
        {"original_labels_sha256": labels_sha}))
    return {"root": root, "labels": labels_path, "staged": staged,
            "out": tmp_path / "out", "studies": studies, "labels_sha": labels_sha}


def run(pair, **kwargs):
    return recovery.recover(pair["root"], 1, pair["labels"], pair["staged"],
                            pair["out"], draws=kwargs.pop("draws", 20), **kwargs)


def test_finds_the_arm_outputs_the_failed_run_looked_for_in_the_wrong_place(pair):
    """`control_fold1` holds the contracts; the weights are one directory over."""
    contract = find_arm_file(pair["root"], "control", "run_contract.json")
    checkpoint = find_arm_file(pair["root"], "control", "convnext_fold1.pt")
    assert contract.parent.name == "control_fold1"
    assert checkpoint.parent.name == "reviewed_control_fold1"
    assert "candidate" not in str(checkpoint)


def test_ambiguous_outputs_raise_instead_of_choosing(pair):
    write_arm(pair["root"], "control", studies=pair["studies"],
              labels_sha=pair["labels_sha"], artifact_dir="control_fold1_rerun")
    with pytest.raises(ValueError, match="ambiguous"):
        find_arm_file(pair["root"], "control", "convnext_fold1.pt")


def test_missing_outputs_raise_rather_than_comparing_one_arm(pair):
    find_arm_file(pair["root"], "candidate", "oof_fold1.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        run(pair)


def test_recovers_the_gate_and_the_provenance_it_rests_on(pair):
    table, summary = run(pair)
    assert summary["shared_studies"] == len(pair["studies"])
    assert list(table.index) == config.LABELS
    assert set(summary["screening_gate"]) >= {
        "passed", "macro_delta", "required_macro_delta", "paired_lower_bound"}
    assert summary["screening_gate"]["required_macro_delta"] == 0.003
    assert [c["arm"] for c in summary["reviewed_label_contracts"]] == list(recovery.ARMS)
    assert len(set(summary["checkpoint_sha256"])) == 2
    written = json.loads((pair["out"] / "comparison.json").read_text())
    assert written["macro"] == summary["macro"]


@pytest.mark.parametrize("mutation", ["settings", "corrections", "split",
                                      "validation", "checkpoint"])
def test_rejects_a_pair_that_is_not_matched(pair, mutation):
    if mutation == "settings":
        write_arm(pair["root"], "candidate", studies=pair["studies"],
                  labels_sha=pair["labels_sha"], run={"seed": 1})
    elif mutation == "corrections":
        write_arm(pair["root"], "candidate", studies=pair["studies"],
                  labels_sha=pair["labels_sha"],
                  corrections={"corrections_sha256": "1" * 64})
    elif mutation == "split":
        write_arm(pair["root"], "candidate", studies=pair["studies"][:39],
                  labels_sha=pair["labels_sha"])
    elif mutation == "validation":
        write_arm(pair["root"], "candidate", studies=pair["studies"],
                  labels_sha=pair["labels_sha"],
                  corrections={"validation_labels_changed": True})
    else:
        find_arm_file(pair["root"], "candidate", "convnext_fold1.pt").write_bytes(b"x")
    with pytest.raises(ValueError):
        run(pair)


def test_refuses_labels_that_changed_since_the_pair_was_staged(pair):
    (pair["staged"] / "receipt.json").write_text(
        json.dumps({"original_labels_sha256": "2" * 64}))
    with pytest.raises(ValueError, match="labels changed"):
        run(pair)
