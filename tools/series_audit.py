"""Stage 0c and 0d: is the input to the synovitis head what we think it is?

Two checks that run before any modelling, because either one firing would
change what the rest of the plan is for.

**0c -- contrast.** Effusion-synovitis is defined in the literature as fluid
distension that cannot be reliably separated from synovitis *without* contrast,
and Hoffa-synovitis is an explicitly non-specific surrogate. So two answers
matter and they point opposite ways: if post-contrast series are present and
being fed to the model as ordinary fluid-sensitive ones, that is a routing bug
worth more than every modelling idea in the plan; if they are genuinely absent,
a hard ceiling applies to the finding and the honest report says so.

The plan's warning is taken literally here: **absence of a string is not
absence of contrast.** Roughly a fifth of series in this collection carry no
usable `SeriesDescription` at all, so the description scan is reported with the
size of its own blind spot, and beside a description-free control -- a study
that acquired the same T1-weighted geometry twice in one plane is the
signature of a pre/post pair whether or not anybody typed "post".

**0d -- series composition.** A study whose protocol included more
fluid-sensitive fat-suppressed sequences presents the same fluid to the
aggregator more than once. If the synovitis score scales with that count, part
of the head is reading protocol rather than anatomy. The check that decides it
is not the raw correlation but the comparison against the *label's* own
correlation with the same count: sicker knees get scanned with more sequences,
so a model tracking protocol is only an artefact if it tracks it harder than
the truth does.

Usage:

    python tools/series_audit.py --index ~/Downloads/index_train.parquet \\
        --predictions failures/gold_predictions.parquet

`--index` is the header scan built by `src/data/index.py` on Kaggle. Without
it 0c cannot run and the tool says so rather than reporting a clean bill.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

# Post-contrast markers as they are actually typed at a console. "+C" is the
# common one and the only one present in this collection; the rest are here
# because a scan that looks for one spelling and reports "absent" is worthless.
CONTRAST_PATTERNS = {
    "+c": r"\+\s*c\b",
    "post": r"\bpost(?:\b|[-_ ]?(?:gd|contrast|c\b))",
    "gad": r"\bgad|\bgd\b",
    "contrast": r"contrast",
    "enhanced": r"enhanc",
    "dynamic": r"\bdyn\b",
    "subtraction": r"subtract",
    "arthrogram": r"arthro",
}

# Descriptions that exist but say nothing. Anonymisation replaced some with a
# placeholder; others are empty. Both are blind spots, not negatives.
EMPTY_DESCRIPTIONS = ("", "dummyseriesdesc!", "none", "nan")


def series_table(index_path: Path) -> pd.DataFrame:
    """One row per series from the instance-level header index."""
    columns = ["StudyInstanceUID", "SeriesInstanceUID", "SeriesDescription",
               "ProtocolName", "EchoTime", "RepetitionTime"]
    frame = pd.read_parquet(index_path, columns=columns)
    return frame.drop_duplicates("SeriesInstanceUID").reset_index(drop=True)


def described(series: pd.DataFrame) -> pd.Series:
    """Lower-cased description + protocol, and whether either says anything."""
    text = (series["SeriesDescription"].fillna("").astype(str) + " | "
            + series["ProtocolName"].fillna("").astype(str)).str.lower()
    return text


def blind_spot(series: pd.DataFrame) -> pd.Series:
    description = (series["SeriesDescription"].fillna("").astype(str)
                   .str.strip().str.lower())
    protocol = (series["ProtocolName"].fillna("").astype(str)
                .str.strip().str.lower())
    return description.isin(EMPTY_DESCRIPTIONS) & protocol.isin(EMPTY_DESCRIPTIONS)


def contrast_scan(series: pd.DataFrame, routing: pd.DataFrame) -> dict:
    """Which series name contrast, and how they are routed to the model."""
    text = described(series)
    hits = pd.Series(False, index=series.index)
    per_pattern = {}
    for name, pattern in CONTRAST_PATTERNS.items():
        match = text.str.contains(pattern, regex=True, na=False)
        per_pattern[name] = int(match.sum())
        hits |= match

    flagged = series[hits].merge(
        routing, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")
    misrouted = flagged[(flagged["Fluid_Sensitive"].fillna(0) == 1)]
    unusable = blind_spot(series)
    return {"per_pattern": per_pattern, "flagged": flagged,
            "misrouted": misrouted,
            "studies": int(series.loc[hits, config.ID_COL].nunique()),
            "unusable_series": int(unusable.sum()),
            "unusable_share": float(unusable.mean()),
            "studies_all_unusable": int(
                series.assign(_u=unusable).groupby(config.ID_COL)["_u"].all().sum()),
            "n_series": len(series),
            "n_studies": int(series[config.ID_COL].nunique())}


def repeat_control(series: pd.DataFrame, routing: pd.DataFrame) -> dict:
    """Description-free control: the same T1-ish acquisition, twice, one plane.

    A pre/post pair is two T1-weighted runs of the same geometry minutes
    apart. Short TE and short TR pick out T1 weighting from the parameters
    rather than from a typed string, so this fires on studies whose
    descriptions were anonymised away. It over-counts -- a repeat for motion
    looks identical -- which is the right direction for a control that exists
    to bound how much contrast the description scan could be missing.
    """
    merged = series.merge(routing, on=["StudyInstanceUID", "SeriesInstanceUID"],
                          how="left")
    echo = pd.to_numeric(merged["EchoTime"], errors="coerce")
    repetition = pd.to_numeric(merged["RepetitionTime"], errors="coerce")
    t1_like = (echo < 30) & (repetition < 1000)
    grouped = merged[t1_like].groupby(
        [config.ID_COL, "Anatomical_Plane", "Fat_Suppression"]).size()
    repeated = grouped[grouped >= 2]
    return {"t1_like": int(t1_like.sum()),
            "repeated_groups": int(len(repeated)),
            "studies": int(repeated.index.get_level_values(0).nunique())}


def composition(routing: pd.DataFrame) -> pd.DataFrame:
    """Per-study counts of the series types the aggregator sees."""
    fluid = routing["Fluid_Sensitive"].fillna(0).astype(int)
    fatsat = routing["Fat_Suppression"].fillna(0).astype(int)
    table = routing.assign(_fluid=fluid, _fatsat=fatsat,
                           _both=(fluid & fatsat).astype(int))
    grouped = table.groupby(config.ID_COL)
    return pd.DataFrame({"n_series": grouped.size(),
                         "n_fluid": grouped["_fluid"].sum(),
                         "n_fluid_fatsat": grouped["_both"].sum()})


def logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def partial_correlation(x, y, control):
    """corr(x, y) with the linear effect of `control` removed from both."""
    x, y, control = (np.asarray(v, dtype=float) for v in (x, y, control))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if ok.sum() < 3:
        return float("nan")
    residual = [v[ok] - np.polyval(np.polyfit(control[ok], v[ok], 1), control[ok])
                for v in (x, y)]
    return float(np.corrcoef(residual[0], residual[1])[0, 1])


def composition_effect(counts, scores, coupled, labels=None) -> dict:
    """Does the score scale with the count -- more than the truth does?"""
    counts = np.asarray(counts, dtype=float)
    ok = np.isfinite(counts)
    score_logit, coupled_logit = logit(scores)[ok], logit(coupled)[ok]
    result = {
        "n": int(ok.sum()),
        "r_score": float(np.corrcoef(counts[ok], score_logit)[0, 1]),
        "r_score_partial": partial_correlation(counts[ok], score_logit,
                                               coupled_logit),
        "r_coupled": float(np.corrcoef(counts[ok], coupled_logit)[0, 1]),
        "r_label": float("nan"),
    }
    if labels is not None:
        label = np.asarray(labels, dtype=float)[ok]
        if np.isfinite(label).all() and label.std() > 0:
            result["r_label"] = float(np.corrcoef(counts[ok], label)[0, 1])
    return result


def read_predictions(path: Path) -> pd.DataFrame:
    frame = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    return frame.set_index(config.ID_COL)[config.LABELS].astype(float)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=None,
                        help="index_train.parquet from src/data/index.py")
    parser.add_argument("--predictions", action="append", default=None,
                        help="predictions to test for the 0d effect; repeatable")
    parser.add_argument("--meta-dir", default=str(config.META_DIR))
    parser.add_argument("--finding", default="Synovitis")
    parser.add_argument("--coupled-with", default="Effusion")
    args = parser.parse_args(argv)

    meta_dir = Path(args.meta_dir)
    routing = pd.read_csv(meta_dir / "train_series.csv")

    print("0c -- contrast")
    if args.index is None:
        print("  SKIPPED: no --index. The competition's series table carries no")
        print("  SeriesDescription, so contrast cannot be ruled in or out from")
        print("  it. Run src/data/index.py on Kaggle and pass the parquet.")
    else:
        series = series_table(Path(args.index))
        scan = contrast_scan(series, routing)
        print(f"  {scan['n_series']} series over {scan['n_studies']} studies")
        named = "  ".join(f"{k}:{v}" for k, v in scan["per_pattern"].items() if v)
        print(f"  markers found: {named or 'none'}")
        print(f"  {len(scan['flagged'])} series in {scan['studies']} studies "
              f"name contrast")
        for _, row in scan["flagged"].iterrows():
            print(f"    {str(row[config.ID_COL])[-12:]}  "
                  f"{str(row['SeriesDescription'])[:32]:32s} "
                  f"fluid={row.get('Fluid_Sensitive')} "
                  f"fatsat={row.get('Fat_Suppression')} "
                  f"{row.get('Anatomical_Plane')}")
        print(f"  routed as fluid-sensitive: {len(scan['misrouted'])}"
              + ("   <- ROUTING BUG, fix before anything else"
                 if len(scan["misrouted"]) else "   (none -- no routing bug)"))
        print(f"  blind spot: {scan['unusable_series']} series "
              f"({scan['unusable_share']:.1%}) carry no description or protocol; "
              f"{scan['studies_all_unusable']} studies carry none at all")
        control = repeat_control(series, routing)
        print(f"  description-free control: {control['repeated_groups']} "
              f"study/plane/fat-sat groups hold two or more T1-like "
              f"acquisitions, in {control['studies']} studies -- an upper "
              f"bound on hidden pre/post pairs, since a repeat for motion "
              f"looks the same")
    print()

    print(f"0d -- does the {args.finding} score scale with series composition?")
    counts = composition(routing)
    mined = pd.read_parquet(meta_dir / "train_labels.parquet").set_index(config.ID_COL)
    gold = pd.read_parquet(meta_dir / "gold_truth_58.parquet").set_index(config.ID_COL)
    for spec in args.predictions or []:
        path = Path(spec)
        preds = read_predictions(path)
        shared = preds.index.intersection(counts.index)
        preds, feature = preds.loc[shared], counts.loc[shared, "n_fluid_fatsat"]
        truth = gold if shared.isin(gold.index).all() else mined
        labels = (truth.reindex(shared)[args.finding]
                  if args.finding in truth.columns else None)
        effect = composition_effect(feature, preds[args.finding],
                                    preds[args.coupled_with], labels)
        print(f"  {path.name}  (n={effect['n']}, truth="
              f"{'gold' if truth is gold else 'mined'})")
        print(f"    corr(n_fluid_fatsat, {args.finding} logit) "
              f"{effect['r_score']:+.3f}   "
              f"partial on {args.coupled_with} {effect['r_score_partial']:+.3f}")
        print(f"    the label's own correlation with the same count  "
              f"{effect['r_label']:+.3f}")
        # A correlation on 58 studies has a standard error near 0.13, which
        # is larger than any effect this check could plausibly find. Refusing
        # to read it is the point: the gold set has no power for 0d, and the
        # OOF predictions over thousands of studies do.
        floor = 200
        if effect["n"] < floor:
            verdict = (f"n={effect['n']} -- correlation SE is about "
                       f"{1 / np.sqrt(effect['n'] - 3):.2f} here, so this is "
                       "not readable. Run it on out-of-fold predictions.")
        elif (np.isfinite(effect["r_label"])
                and effect["r_score"] > effect["r_label"] + 0.05):
            verdict = ("the score tracks protocol harder than the truth does "
                       "-- series-type-aware aggregation is worth trying")
        else:
            verdict = ("the score tracks protocol no harder than the truth "
                       "does -- no aggregation artefact visible here")
        print(f"    -> {verdict}")
        table = pd.DataFrame({"n": np.asarray(feature, dtype=float),
                              "score": preds[args.finding].to_numpy()})
        summary = table.groupby("n")["score"].agg(["mean", "size"])
        print("    mean score by count: " + "  ".join(
            f"{int(k)}:{v['mean']:.3f}(n={int(v['size'])})"
            for k, v in summary.iterrows()))
    if not args.predictions:
        print("  SKIPPED: no --predictions given")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
