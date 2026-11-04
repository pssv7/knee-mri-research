#!/usr/bin/env python3
"""Build a blinded, deterministic PF OA/Synovitis hard-case review cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment22_image_hard_cases_v1"
ID = "StudyInstanceUID"
CONFIDENCE = 0.90
PER_GROUP = 10


def explicit(frame: pd.DataFrame, label: str, value: int):
    return frame[label].eq(float(value)) & frame[f"{label}_confidence"].ge(CONFIDENCE)


def stable_take(frame: pd.DataFrame, arm: str, group: str, count: int):
    ranked = frame.copy()
    ranked["_key"] = ranked[ID].astype(str).map(
        lambda uid: hashlib.sha256(f"experiment22:{arm}:{group}:{uid}".encode()).hexdigest())
    return ranked.sort_values("_key", kind="stable").head(count).drop(columns="_key")


def main():
    labels = pd.read_parquet(ROOT / "data/meta/llm_labels.parquet")
    reports = pd.read_csv(ROOT / "data/meta/train.csv", usecols=[ID, "Report"], dtype={ID: str})
    gold = pd.read_csv(ROOT / "data/meta/gold_truth_58.csv", dtype={ID: str})
    labels[ID] = labels[ID].astype(str)
    labels = labels[~labels[ID].isin(set(gold[ID]))].copy()

    pfp, pfn = explicit(labels, "PF OA", 1), explicit(labels, "PF OA", 0)
    mop, mon = explicit(labels, "Medial OA", 1), explicit(labels, "Medial OA", 0)
    lop, lon = explicit(labels, "Lateral OA", 1), explicit(labels, "Lateral OA", 0)
    syp, syn = explicit(labels, "Synovitis", 1), explicit(labels, "Synovitis", 0)
    efp, efn = explicit(labels, "Effusion", 1), explicit(labels, "Effusion", 0)

    definitions = [
        ("pfoa", "pf_positive_both_tf_oa_negative", pfp & mon & lon),
        ("pfoa", "pf_negative_any_tf_oa_positive", pfn & (mop | lop)),
        ("pfoa", "pf_positive_any_tf_oa_positive_control", pfp & (mop | lop)),
        ("pfoa", "pf_negative_both_tf_oa_negative_control", pfn & mon & lon),
        ("synovitis", "syn_negative_effusion_positive", syn & efp),
        ("synovitis", "syn_positive_effusion_negative", syp & efn),
        ("synovitis", "syn_positive_effusion_positive_control", syp & efp),
        ("synovitis", "syn_negative_effusion_negative_control", syn & efn),
    ]
    selected, inventory = [], []
    for arm, group, mask in definitions:
        pool = labels.loc[mask].copy()
        if len(pool) < PER_GROUP:
            raise RuntimeError(f"{arm}/{group} has {len(pool)} studies; need {PER_GROUP}")
        chosen = stable_take(pool, arm, group, PER_GROUP)
        chosen["arm"] = arm
        chosen["selection_group"] = group
        selected.append(chosen)
        inventory.append({"arm": arm, "selection_group": group,
                          "available": len(pool), "selected": len(chosen)})
    cohort = pd.concat(selected, ignore_index=True)
    if len(cohort) != 8 * PER_GROUP or cohort.duplicated(["arm", ID]).any():
        raise RuntimeError("Experiment 22 selection identity contract failed")
    cohort = cohort.merge(reports, on=ID, how="left", validate="many_to_one")
    cohort["case_id"] = [f"E22-{i:03d}" for i in range(1, len(cohort) + 1)]

    OUT.mkdir(parents=True, exist_ok=True)
    sealed_columns = ["case_id", "arm", "selection_group", ID, "Report"]
    evidence_columns = [column for column in cohort.columns
                        if column.endswith("_evidence") or column.endswith("_confidence")]
    cohort[sealed_columns + evidence_columns].to_parquet(
        OUT / "sealed_selection_metadata.parquet", index=False)
    review = cohort[["case_id", "arm"]].copy()
    review["expert_label"] = pd.Series([pd.NA] * len(review), dtype="Int64")
    review["image_quality"] = ""
    review["reviewer_confidence"] = ""
    review["notes"] = ""
    review.to_csv(OUT / "expert_label_template.csv", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "selection_inventory.csv", index=False)
    receipt = {
        "status": "selected_not_rendered", "experiment": 22,
        "cases": len(cohort), "pfoa_cases": int((cohort.arm == "pfoa").sum()),
        "synovitis_cases": int((cohort.arm == "synovitis").sum()),
        "cases_per_group": PER_GROUP, "gold58_excluded": True,
        "review_is_blinded": True,
        "label_schema": {"expert_label": "0=absent, 1=present",
                         "image_quality": "diagnostic, limited, or unreadable",
                         "reviewer_confidence": "low, medium, or high"},
    }
    (OUT / "selection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
