"""Which studies is the model getting wrong, and do they have anything in common?

Seven consecutive changes have measured null against a single macro number.
That number cannot say *why* a model scores 0.877, and it cannot distinguish
"the model is uniformly mediocre" from "the model is fine except on studies
missing a fluid-sensitive series". This script asks the second question.

**Two modes, with opposite trade-offs, and it is worth running both:**

- `--truth gold` scores against the 58 radiologist-labelled studies. Real
  labels, no power. A subgroup of 20 studies carries a per-finding standard
  error near 0.15, so almost nothing here is conclusive on its own.
- `--truth mined` scores out-of-fold predictions against the mined labels over
  thousands of studies. Real power, contaminated labels -- it measures
  agreement with the extractor, not with the truth. A subgroup effect that
  appears in *both* is worth acting on; one that appears only in gold is
  probably noise, and one that appears only in mined may be an artefact of how
  reports are written.

**Every AUC is printed with its Hanley-McNeil standard error**, because this
project has repeatedly acted on differences inside the noise band. If the SE
column is larger than the difference you are looking at, there is no finding.

Usage:

    python tools/error_analysis.py --predictions oof_fold1.parquet --truth mined
    python tools/error_analysis.py --predictions gold_predictions_fold1.parquet \\
        --truth gold --worst 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.model import conditional  # noqa: E402
from src.model import inspect as knee_inspect  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# One definition of the AUC and its standard error, imported rather than
# copied: `src/model/conditional.py` is torch-free for exactly this reason, and
# two implementations of the number every decision here rests on is how they
# drift apart. The names are re-exported so this module's callers -- and its
# tests -- keep reading them from here.
auc = conditional.auc
auc_se = conditional.auc_se


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def study_features(series_csv: Path) -> pd.DataFrame:
    """Per-study descriptors from the competition's series table.

    Only columns given in BOTH train and test, so any effect found here is one
    the submission could actually act on.
    """
    series = pd.read_csv(series_csv)
    plane = series["Anatomical_Plane"].astype(str)
    series = series.assign(
        _sag=(plane == "Sagittal").astype(int),
        _cor=(plane == "Coronal").astype(int),
        _ax=(plane == "Axial").astype(int),
        _fluid=series["Fluid_Sensitive"].fillna(0).astype(int),
        _fatsat=series["Fat_Suppression"].fillna(0).astype(int))
    grouped = series.groupby(config.ID_COL)
    return pd.DataFrame({
        "n_series": grouped.size(),
        "n_sagittal": grouped["_sag"].sum(),
        "n_coronal": grouped["_cor"].sum(),
        "n_axial": grouped["_ax"].sum(),
        "n_fluid_sensitive": grouped["_fluid"].sum(),
        "n_fat_suppressed": grouped["_fatsat"].sum(),
    })


def load_truth(kind: str, meta_dir: Path) -> pd.DataFrame:
    """Gold (58 radiologist labels) or mined (thousands, from reports)."""
    if kind == "gold":
        frame = pd.read_csv(meta_dir / "train.csv")
        frame = frame[frame[config.LABELS[0]].notna()]
        return frame.set_index(config.ID_COL)[config.LABELS].astype(float)
    mined = pd.read_parquet(meta_dir / "train_labels.parquet")
    # Mined targets are soft; binarise at 0.5, as the training loop's val does.
    return (mined.set_index(config.ID_COL)[config.LABELS] >= 0.5).astype(float)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def per_finding(truth: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in config.LABELS:
        y, s = truth[name].to_numpy(), preds[name].to_numpy()
        value = auc(y, s)
        n_pos, n_neg = int((y == 1).sum()), int((y != 1).sum())
        rows.append({"finding": name, "auc": value, "se": auc_se(value, n_pos, n_neg),
                     "n_pos": n_pos, "n_neg": n_neg})
    return pd.DataFrame(rows).set_index("finding")


def subgroup_table(truth, preds, features, column, edges) -> pd.DataFrame:
    """Macro AUC within buckets of one study descriptor.

    This is the question the aggregate number cannot answer: is the model
    uniformly mediocre, or is it losing a definable slice of studies?
    """
    values = features.reindex(truth.index)[column]
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        if mask.sum() < 8:
            rows.append({"bucket": f"[{low},{high})", "n": int(mask.sum()),
                         "macro": float("nan"), "worst_se": float("nan")})
            continue
        table = per_finding(truth[mask.to_numpy()], preds[mask.to_numpy()])
        rows.append({"bucket": f"[{low},{high})", "n": int(mask.sum()),
                     "macro": float(table["auc"].mean(skipna=True)),
                     "worst_se": float(table["se"].max(skipna=True))})
    return pd.DataFrame(rows).set_index("bucket")


def worst_studies(truth, preds, features, limit) -> pd.DataFrame:
    """Studies contributing most to the loss, by summed per-finding rank error.

    For each finding a study's error is how far its prediction sits from where
    its label says it should rank: a positive scored below the median of the
    negatives, or the reverse. Summed over the twelve findings, that is how
    much this one study is dragging the macro down.
    """
    # One implementation, in src/model/inspect.py, because the notebook that
    # renders the failing images ranks them the same way and the two drifting
    # apart would mean looking at different studies than the table names.
    matrix = knee_inspect.per_finding_errors(
        truth[config.LABELS].to_numpy(), preds[config.LABELS].to_numpy())
    detail = pd.DataFrame(matrix, index=truth.index, columns=config.LABELS)

    out = pd.DataFrame({"rank_error": matrix.sum(axis=1)}, index=truth.index)
    out = out.join(features, how="left")
    worst = out.sort_values("rank_error", ascending=False).head(limit)
    return worst.assign(worst_finding=detail.loc[worst.index].idxmax(axis=1),
                        worst_error=detail.loc[worst.index].max(axis=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="parquet/csv with StudyInstanceUID + the 12 findings")
    parser.add_argument("--truth", choices=("gold", "mined"), default="gold")
    parser.add_argument("--meta-dir", default=str(config.META_DIR))
    parser.add_argument("--series-csv", default=None)
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args(argv)

    meta_dir = Path(args.meta_dir)
    path = Path(args.predictions)
    preds = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    preds = preds.set_index(config.ID_COL)[config.LABELS].astype(float)

    truth = load_truth(args.truth, meta_dir)
    shared = truth.index.intersection(preds.index)
    if len(shared) == 0:
        raise SystemExit(
            f"no studies in common between {path.name} and the {args.truth} "
            "labels -- wrong predictions file, or wrong --truth")
    truth, preds = truth.loc[shared], preds.loc[shared]

    series_csv = Path(args.series_csv) if args.series_csv else \
        meta_dir / "train_series.csv"
    features = study_features(series_csv)

    print(f"predictions : {path.name}")
    print(f"truth       : {args.truth}   ({len(shared)} studies scored)")
    if args.truth == "mined":
        print("              NB mined labels measure agreement with the "
              "extractor, not with truth")
    print()

    table = per_finding(truth, preds)
    print("per finding")
    print(f"  {'finding':18s} {'auc':>7s} {'+/- se':>8s} {'pos':>5s} {'neg':>5s}")
    for name, row in table.iterrows():
        print(f"  {name:18s} {row['auc']:7.3f} {row['se']:8.3f} "
              f"{int(row['n_pos']):5d} {int(row['n_neg']):5d}")
    print(f"  {'macro':18s} {table['auc'].mean(skipna=True):7.3f}")
    print()

    print("macro by study descriptor  (nan = fewer than 8 studies in the bucket)")
    for column, edges in (("n_series", [0, 4, 5, 6, 8, 99]),
                          ("n_fluid_sensitive", [0, 1, 2, 3, 99]),
                          ("n_sagittal", [0, 1, 2, 3, 99]),
                          ("n_coronal", [0, 1, 2, 99]),
                          ("n_axial", [0, 1, 2, 99])):
        sub = subgroup_table(truth, preds, features, column, edges)
        line = "  ".join(f"{b}:{r['macro']:.3f}(n={int(r['n'])})"
                         if np.isfinite(r["macro"]) else f"{b}:--(n={int(r['n'])})"
                         for b, r in sub.iterrows())
        print(f"  {column:20s} {line}")
    print()

    print(f"worst {args.worst} studies by summed rank error")
    worst = worst_studies(truth, preds, features, args.worst)
    cols = ["rank_error", "worst_finding", "worst_error", "n_series",
            "n_fluid_sensitive", "n_sagittal", "n_coronal", "n_axial"]
    cols = [c for c in cols if c in worst.columns]
    printable = worst[cols].copy()
    printable.index = [str(i)[-12:] for i in printable.index]
    print(printable.to_string(float_format=lambda v: f"{v:.2f}"))
    print()
    print("Read the SE column before believing any gap. On 58 gold studies a")
    print("per-finding SE of 0.05-0.08 is normal, and a subgroup of twenty")
    print("carries roughly double that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
