"""Run the reviewed-label pair comparison on artifacts that already exist.

Both fold-1 arms trained to completion on Kaggle; the comparison that was
supposed to follow them died on `FileNotFoundError:
/kaggle/working/control_fold1/convnext_fold1.pt.json`. The weights, the OOF
predictions, the folds and both contracts were written and downloaded, so the
comparison is a local computation over saved files -- nothing here retrains
anything, and this refuses to run if an input is missing rather than falling
back to a partial answer.

The checks are the ones the on-Kaggle runner would have made, in the same
order: arm contracts, identical settings and initialisation, identical split
membership recorded in the checkpoint sidecars, identical folds, the label
file the arms were scored against, then `compare_oof` and the pre-declared
screening gate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import artifacts
from tools.compare_oof import compare, find_arm_file, read_predictions

ARMS = ("control", "candidate")
# Pre-declared in docs/multiplane_reset_plan.md and in the launcher that staged
# this pair. Recorded here so the recovered numbers are read against the gate
# that was set before the run, not one chosen after seeing them.
REQUIRED_MACRO_DELTA = 0.003
MAXIMUM_FINDING_DROP = 0.02


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    """Repo-relative where possible; absolute for anything outside it."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def locate(root, fold):
    """Every file the comparison needs, per arm, found by content not by name."""
    names = ("label_review_contract.json", "run_contract.json",
             f"convnext_fold{fold}.pt", f"oof_fold{fold}.parquet", "folds.parquet")
    return {arm: {name: find_arm_file(root, arm, name) for name in names}
            for arm in ARMS}


def verify(found, corrections_sha256, labels_sha256, correction_count):
    """Assert the two arms differ in the eleven corrections and nothing else."""
    reviews = {arm: json.loads(found[arm]["label_review_contract.json"].read_text())
               for arm in ARMS}
    expected = {"control": 0, "candidate": correction_count}
    for arm in ARMS:
        review = reviews[arm]
        if review["arm"] != arm or review["correction_count"] != expected[arm]:
            raise ValueError(f"{arm} contract does not describe the {arm} arm")
        if review["corrections_sha256"] != corrections_sha256:
            raise ValueError(f"{arm} trained against different corrections")
        if review["original_labels_sha256"] != labels_sha256:
            raise ValueError(f"{arm} trained against different original labels")
        if review["validation_labels_changed"] or review["folds_changed"]:
            raise ValueError(f"{arm} changed validation labels or folds")

    runs = [json.loads(found[arm]["run_contract.json"].read_text()) for arm in ARMS]
    if runs[0] != runs[1]:
        differing = sorted(k for k in set(runs[0]) | set(runs[1])
                           if runs[0].get(k) != runs[1].get(k))
        raise ValueError(f"settings or initialisation differ: {differing}")

    # Recomputes each checkpoint's sha256 and rejects a geometry the current
    # cache cannot serve, so a stale or truncated download cannot be compared.
    meta = [artifacts.load_metadata(found[arm][f"convnext_fold{runs[0]['fold']}.pt"])
            for arm in ARMS]
    for key in ("train_uids", "val_uids", "fold", "labels"):
        if meta[0][key] != meta[1][key]:
            raise ValueError(f"provenance differs between arms: {key}")
    if set(meta[0]["train_uids"]) & set(meta[0]["val_uids"]):
        raise ValueError("training and validation studies overlap")

    folds = [pd.read_parquet(found[arm]["folds.parquet"]) for arm in ARMS]
    pd.testing.assert_frame_equal(folds[0], folds[1])
    return reviews, runs[0], meta, folds[0]


def recover(run_dir, fold, labels_path, staged, out_dir, draws=1000, seed=0):
    corrections = json.loads((staged / "corrections.json").read_text())
    receipt = json.loads((staged / "receipt.json").read_text())
    labels_sha256 = sha256(labels_path)
    if labels_sha256 != receipt["original_labels_sha256"]:
        raise ValueError("original labels changed since the pair was staged")
    corrections_sha256 = hashlib.sha256(
        json.dumps(corrections, sort_keys=True).encode()).hexdigest()

    found = locate(run_dir, fold)
    reviews, run, meta, folds = verify(
        found, corrections_sha256, labels_sha256, len(corrections))
    if run["fold"] != fold:
        raise ValueError(f"run contract is fold {run['fold']}, not {fold}")

    labels = pd.read_parquet(labels_path)
    frames = [read_predictions([found[arm][f"oof_fold{fold}.parquet"]], folds)
              for arm in ARMS]
    validation = set(meta[0]["val_uids"])
    for arm, frame in zip(ARMS, frames):
        if set(frame.index) != validation:
            raise ValueError(f"{arm} OOF is not exactly the validation fold")

    table, summary = compare(labels, *frames, draws=draws, seed=seed)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("comparison produced non-finite AUCs")

    delta = summary["macro"]["candidate"] - summary["macro"]["baseline"]
    lower = summary["paired_macro_delta_95ci"]["candidate"][0]
    passed = bool(delta >= REQUIRED_MACRO_DELTA
                  and (table.candidate_delta >= -MAXIMUM_FINDING_DROP).all())
    summary["screening_gate"] = {
        "passed": passed, "macro_delta": float(delta),
        "required_macro_delta": REQUIRED_MACRO_DELTA,
        "maximum_finding_drop": MAXIMUM_FINDING_DROP,
        "paired_lower_bound": float(lower),
        "next_step": "repeat on fold 3" if passed else "review before more training"}
    summary["reviewed_label_contracts"] = [reviews[arm] for arm in ARMS]
    summary["run_contract"] = run
    summary["checkpoint_sha256"] = [m["sha256"] for m in meta]
    summary["recovered_from"] = {
        arm: {name: relative(path) for name, path in files.items()}
        for arm, files in found.items()}
    summary["recovery"] = (
        "Computed locally from the downloaded fold-%d artifacts after the "
        "on-Kaggle comparison failed on arm output discovery. No arm was "
        "retrained; both checkpoints match the sha256 they were exported with."
        % fold)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "per_finding.csv", index_label="finding")
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return table, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/kaggle_reviewed_labels_fold1/v3_final")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--staged", default="outputs/reviewed_labels_v1")
    parser.add_argument("--labels", default="data/meta/train_labels.parquet")
    parser.add_argument("--out", default="outputs/reviewed_labels_fold1_comparison")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    table, summary = recover(ROOT / args.run_dir, args.fold, ROOT / args.labels,
                             ROOT / args.staged, ROOT / args.out, args.draws)
    print(table.round(5).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
