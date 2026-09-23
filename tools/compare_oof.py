"""Compare aligned OOF predictions and a fixed 50/50 probability ensemble."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.labels.calibrate import _auc
from src.model import splits


ARM_NAME_PATTERN = "(?:^|_){arm}(?:_|$)"


def find_arm_file(root, arm, name):
    """Locate one arm's output file without assuming which directory holds it.

    The fold-1 reviewed-label pair wrote its checkpoints to
    `reviewed_control_fold1` (the training script's own output directory) while
    the comparison that followed read `control_fold1` (the directory it had
    created to run that script in). Both existed, so the failure was a
    FileNotFoundError after two completed 12-epoch arms rather than anything
    the runs themselves did wrong.

    Discovery is therefore by content: the unique file of this name under a
    directory naming this arm. Ambiguity raises rather than picking one --
    averaging or guessing between two arms' artifacts is the error this is
    here to prevent.
    """
    root = Path(root)
    pattern = re.compile(ARM_NAME_PATTERN.format(arm=re.escape(arm)))
    found = sorted({p for p in root.rglob(name) if pattern.search(p.parent.name)})
    if not found:
        raise FileNotFoundError(f"no {name} for arm {arm!r} under {root}")
    if len(found) > 1:
        listed = ", ".join(str(p) for p in found)
        raise ValueError(f"ambiguous {name} for arm {arm!r}: {listed}")
    return found[0]


def read_predictions(paths, folded):
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if frame[config.ID_COL].duplicated().any():
        raise ValueError("duplicate studies in OOF predictions")
    if "fold" not in frame:
        raise ValueError("OOF predictions must carry fold assignments")
    indexed = frame.set_index(config.ID_COL)
    expected = folded.set_index(config.ID_COL)["fold"]
    if not indexed.index.isin(expected.index).all():
        raise ValueError("OOF includes gold, unknown, or unlabelled studies")
    if not indexed["fold"].eq(expected.loc[indexed.index]).all():
        raise ValueError("OOF folds disagree with the declared split")
    probs = indexed[config.LABELS].to_numpy(float)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("OOF probabilities must be finite and in [0, 1]")
    return indexed


def aucs(truth, probs):
    return np.array([_auc(truth[:, i], probs[:, i]) for i in range(truth.shape[1])])


def compare(labels, baseline, candidate, draws=1000, seed=0):
    shared = baseline.index.intersection(candidate.index).sort_values()
    if not len(shared):
        raise ValueError("no shared OOF studies")
    truth = (labels.set_index(config.ID_COL).loc[shared, config.LABELS].to_numpy() >= .5).astype(int)
    a, b = [frame.loc[shared, config.LABELS].to_numpy(float) for frame in (baseline, candidate)]
    predictions = {"baseline": a, "candidate": b, "blend_50_50": (a + b) / 2}
    results = {name: aucs(truth, pred) for name, pred in predictions.items()}
    table = pd.DataFrame(results, index=config.LABELS)
    table["candidate_delta"] = table.candidate - table.baseline
    table["blend_delta"] = table.blend_50_50 - table.baseline
    rng = np.random.default_rng(seed)
    fold = baseline.loc[shared, "fold"].to_numpy()
    blocks = [np.flatnonzero(fold == f) for f in np.unique(fold)]
    per_fold = {str(int(f)): {name: float(np.nanmean(aucs(truth[fold == f], pred[fold == f])))
                             for name, pred in predictions.items()} for f in np.unique(fold)}
    deltas = {"candidate": [], "blend_50_50": []}
    for _ in range(draws):
        chosen = np.concatenate([rng.choice(block, len(block), replace=True) for block in blocks])
        base = aucs(truth[chosen], a[chosen])
        for name in deltas:
            values = aucs(truth[chosen], predictions[name][chosen])
            valid = np.isfinite(base) & np.isfinite(values)
            if valid.any():
                deltas[name].append(float((values[valid] - base[valid]).mean()))
    summary = {"shared_studies": len(shared), "folds": sorted(set(map(int, fold))),
               "baseline_only": len(baseline.index.difference(shared)),
               "candidate_only": len(candidate.index.difference(shared)),
               "macro": {k: float(np.nanmean(v)) for k, v in results.items()},
               "per_fold_macro": per_fold,
               "paired_macro_delta_95ci": {
                   k: np.quantile(v, [.025, .975]).tolist() for k, v in deltas.items()},
               "limitation": "Mined-label OOF proxy; fold provenance asserted by files, not proof of training exclusion. Bootstrap holds fitted models fixed; it excludes retraining variability."}
    return table, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--labels", default="data/meta/train_labels.parquet")
    parser.add_argument("--folds", help="Saved folds.parquet; default is the legacy study split")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    labels = pd.read_parquet(args.labels)
    folded = pd.read_parquet(args.folds) if args.folds else splits.make_folds(labels)
    a, b = [read_predictions(paths, folded) for paths in (args.baseline, args.candidate)]
    table, summary = compare(labels, a, b, args.draws)
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    table.to_csv(path / "per_finding.csv", index_label="finding")
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.round(4).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
