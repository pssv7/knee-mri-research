#!/usr/bin/env python3
"""Validate and reconcile provisional Experiment 22 image labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment22_image_hard_cases_v1/completed_labels"
SOURCE = Path("/Users/pssv/Downloads/experiment22_expert_labels_completed.csv")
SEALED = ROOT / "outputs/experiment22_image_hard_cases_v1/sealed_selection_metadata.parquet"
REQUIRED = ["case_id", "arm", "montage_file", "expert_label", "image_quality",
            "reviewer_confidence", "notes"]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels = pd.read_csv(SOURCE, dtype={"case_id": str})
    sealed = pd.read_parquet(SEALED)
    if labels.columns.tolist() != REQUIRED:
        raise RuntimeError("Completed label columns changed")
    if len(labels) != 80 or labels.case_id.duplicated().any():
        raise RuntimeError("Expected 80 unique completed cases")
    if labels[REQUIRED].isna().any().any():
        raise RuntimeError("Completed label file contains blank fields")
    if set(labels.expert_label) != {0, 1}:
        raise RuntimeError("expert_label must contain both binary classes")
    if not set(labels.image_quality) <= {"diagnostic", "limited", "unreadable"}:
        raise RuntimeError("Unexpected image_quality value")
    if not set(labels.reviewer_confidence) <= {"low", "medium", "high"}:
        raise RuntimeError("Unexpected reviewer_confidence value")
    if set(labels.case_id) != set(sealed.case_id):
        raise RuntimeError("Completed and sealed case identities differ")
    merged = labels.merge(
        sealed[["case_id", "arm", "selection_group", "StudyInstanceUID"]],
        on=["case_id", "arm"], how="left", validate="one_to_one")
    # These labels came from a vision model reviewing reduced montages. Keep
    # only diagnostic medium/high calls, and encode confidence as a weight.
    merged["training_weight"] = 0.0
    merged.loc[(merged.image_quality == "diagnostic") &
               (merged.reviewer_confidence == "medium"), "training_weight"] = 0.65
    merged.loc[(merged.image_quality == "diagnostic") &
               (merged.reviewer_confidence == "high"), "training_weight"] = 1.0
    merged["target_label"] = merged.arm.map({"pfoa": "PF OA", "synovitis": "Synovitis"})
    usable = merged[merged.training_weight > 0].copy()
    counts = usable.groupby(["arm", "expert_label"]).size().unstack(fill_value=0)
    for arm in ("pfoa", "synovitis"):
        if counts.loc[arm].get(0, 0) == 0 or counts.loc[arm].get(1, 0) == 0:
            raise RuntimeError(f"Usable {arm} labels lack both classes")
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT / "experiment22_all_reconciled.parquet", index=False)
    usable[["StudyInstanceUID", "case_id", "arm", "target_label", "expert_label",
            "training_weight", "image_quality", "reviewer_confidence",
            "selection_group", "notes"]].to_parquet(
                OUT / "experiment22_usable_training_labels.parquet", index=False)
    receipt = {
        "status": "complete", "experiment": 22,
        "annotation_type": "provisional AI vision labels from blinded MRI montages",
        "source_rows": len(merged), "usable_rows": len(usable),
        "excluded_rows": int((merged.training_weight == 0).sum()),
        "usable_counts": {
            arm: {str(int(label)): int(value) for label, value in row.items()}
            for arm, row in counts.to_dict(orient="index").items()
        },
        "weight_rule": {"diagnostic/high": 1.0, "diagnostic/medium": 0.65,
                        "limited_or_low_or_unreadable": 0.0},
        "suitable_for_standalone_training": False,
        "recommended_use": "hard-case audit or low-weight auxiliary supervision only",
        "source_sha256": sha256(SOURCE),
    }
    (OUT / "reconciliation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
