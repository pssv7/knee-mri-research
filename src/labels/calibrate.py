"""Score a report extractor against the 58 radiologist-labelled studies.

Those 58 labels are the only trustworthy signal in the dataset, and this is
their highest-value use: measuring how much a mined label can be believed,
per finding. They are never used for training.

The per-finding AUC produced here drives the kill criterion -- a finding whose
mined labels score below KILL_THRESHOLD is noise, and the model should predict
the training prior for it rather than learn from corrupted targets.
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src import config

# Below this AUC against gold, a finding's mined labels are not worth training
# on. Chosen to sit meaningfully above chance while tolerating the noise of a
# 58-sample estimate.
KILL_THRESHOLD = 0.70


def load_gold(train_csv=None) -> pd.DataFrame:
    """The 58 studies carrying radiologist labels."""
    frame = pd.read_csv(train_csv or config.TRAIN_CSV)
    labelled = frame[frame[config.LABELS[0]].notna()].copy()
    for name in config.LABELS:
        labelled[name] = labelled[name].astype(int)
    return labelled.reset_index(drop=True)


def load_unlabelled(train_csv=None) -> pd.DataFrame:
    """The 4,349 studies carrying only a report."""
    frame = pd.read_csv(train_csv or config.TRAIN_CSV)
    return frame[frame[config.LABELS[0]].isna()].reset_index(drop=True)


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via rank statistic, so sklearn stays an optional dependency."""
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks within ties, otherwise a constant score would not score 0.5.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sums = np.zeros(len(unique))
    np.add.at(tie_sums, inverse, ranks)
    ranks = (tie_sums / counts)[inverse]

    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_extractor(
    extract_fn: Callable[[str, str], Dict],
    gold: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-finding reliability of an extractor against the gold studies.

    `extract_fn(report, finding)` must return a dict with `present` (1/0/None)
    and `severity` (float or None), matching the lexicon module's contract.
    """
    gold = gold if gold is not None else load_gold()
    reports = gold["Report"].tolist()
    records = []

    for name in config.LABELS:
        truth = gold[name].to_numpy()
        results = [extract_fn(report, name) for report in reports]

        present = np.array([
            r["present"] if r["present"] is not None else -1 for r in results
        ])
        severity = np.array([
            r["severity"] if r["severity"] is not None else 0.0 for r in results
        ])

        predicted = (present == 1).astype(int)
        true_pos = int(((predicted == 1) & (truth == 1)).sum())
        false_pos = int(((predicted == 1) & (truth == 0)).sum())
        false_neg = int(((predicted == 0) & (truth == 1)).sum())

        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else np.nan
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else np.nan
        auc = _auc(truth, severity)

        # How often a finding goes unmentioned, and how often it is truly
        # positive when unmentioned. The second number is what fusion should
        # use as the soft target for silence -- treating it as a hard zero is
        # what injects systematic noise.
        unmentioned = present == -1
        silence_rate = float(unmentioned.mean())
        prior_if_silent = (
            float(truth[unmentioned].mean()) if unmentioned.any() else np.nan
        )

        records.append({
            "finding": name,
            "n_positive": int(truth.sum()),
            "mention_rate": 1.0 - silence_rate,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "prior_if_silent": prior_if_silent,
            "usable": bool(auc >= KILL_THRESHOLD) if np.isfinite(auc) else False,
        })

    return pd.DataFrame(records).set_index("finding")


def format_report(table: pd.DataFrame) -> str:
    """Human-readable reliability table."""
    lines = [
        f"{'finding':18s} {'n+':>3s} {'ment':>5s} {'prec':>5s} {'rec':>5s} "
        f"{'AUC':>5s} {'P(1|silent)':>11s} {'use':>4s}",
        "-" * 66,
    ]
    for name, row in table.iterrows():
        lines.append(
            f"{name:18s} {int(row['n_positive']):3d} "
            f"{row['mention_rate']:4.0%} {row['precision']:5.2f} "
            f"{row['recall']:5.2f} {row['auc']:5.3f} "
            f"{row['prior_if_silent']:11.2f} {'yes' if row['usable'] else 'NO':>4s}"
        )
    lines.append("-" * 66)
    lines.append(f"macro AUC: {table['auc'].mean():.3f}   "
                 f"usable findings: {int(table['usable'].sum())}/{len(table)}")
    return "\n".join(lines)


def main():
    from src.labels import lexicon

    table = score_extractor(lexicon.extract_finding)
    print(format_report(table))


if __name__ == "__main__":
    main()
