"""Score synovitis on the metric that can measure it, and read the Stage 0 gate.

The plan's §3 and §4. Pooled AUC on this finding is reproducible from the
effusion label alone (0.698 against the head's 0.701), so it cannot detect
progress until the gain is very large -- which is the mechanism behind seven
consecutive null readings. This tool reports **effusion-stratified conditional
AUC** first, pooled beside it, and never macro.

Three questions, in the order the plan asks them:

1. **Where does the error sit?** The pair decomposition splits every
   discordant positive/negative pair by the effusion status of each member.
   Within-stratum pairs are the entire addressable headroom; cross-stratum
   pairs are ordered by effusion coupling and move for reasons that have
   nothing to do with synovitis.

2. **What would a conditional target be worth in pooled terms?** So a
   conditional gain is not dismissed by a leaderboard number that moves a
   third as fast, and so the ceiling under an unchanged gating is visible.

3. **Does the gate open?** Stage 0a scores the frozen path's predictions and
   0b re-scores the silence-reweighting run's; both are decided on whether
   conditional AUC is clearly above chance, tested by bootstrap. Pass the run
   under test with `--predictions` and the incumbent with `--baseline`, and
   the difference is compared paired, on the same studies.

Usage:

    # the §3 report on one run
    python tools/synovitis_gate.py --predictions failures/gold_predictions.parquet

    # Stage 0a / 0b: a candidate against the fine-tuned incumbent
    python tools/synovitis_gate.py \\
        --predictions frozen_gold_predictions.parquet \\
        --baseline failures/gold_predictions.parquet

    # §6 interim proxy: mined labels, thousands of studies, far more power
    python tools/synovitis_gate.py --truth mined \\
        --predictions models_train/fold1/oof_fold1.parquet

**Pre-registered decision rule.** The plan says "clearly > 0.5", "= 0.5" and
"ambiguous" without numeric cutoffs, so these are fixed here, once, in code,
rather than chosen after seeing a number:

    clear      2.5th percentile of the bootstrap > 0.5, and
               P(conditional AUC <= 0.5) < 0.05
    chance     point estimate <= 0.55 and P(<= 0.5) >= 0.25
    ambiguous  anything else -- the expected outcome at n=58

Note one conflict in the plan itself, left visible rather than resolved: for a
frozen conditional AUC at chance, §4's gate table says Stage 3 (anatomical
targeting) and §10's decision table says Stage 4 (descope). The tool prints
both.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.model import conditional as cond  # noqa: E402
from tools.error_analysis import load_truth  # noqa: E402

CONVERSION_TARGETS = (0.70, 0.75, 0.80, 0.85, 1.00)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

# One definition of the cutoffs and the actions they imply, imported rather
# than restated: `src/model/conditional.py` is bootstrapped into the training
# notebooks, which read the same gate at the end of a frozen-path run, and a
# second copy here is how the two would come to disagree.
gate_reading = cond.gate_reading
GATE_ACTIONS = cond.GATE_ACTIONS


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_predictions(spec: str):
    """`path` or `path:label`, parquet or csv, indexed by study."""
    label = None
    if ":" in spec and not Path(spec).exists():
        spec, label = spec.rsplit(":", 1)
    path = Path(spec)
    frame = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    frame = frame.set_index(config.ID_COL)[config.LABELS].astype(float)
    return frame, label or path.name


def strata_for(kind, truth, preds, finding_by, threshold):
    """Gold effusion label, or a median split of the predicted effusion score.

    Stratifying on the prediction is what makes this metric computable during
    training, where no gold effusion label exists. It costs little: 0.617
    against the gold label's 0.630 on the reference run.
    """
    if kind == "gold":
        return truth[finding_by].to_numpy(), f"gold {finding_by} label"
    values = cond.strata_from_scores(preds[finding_by].to_numpy(), threshold)
    where = "median" if threshold is None else f">= {threshold:g}"
    return values, f"predicted {finding_by} ({where} split)"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_one(name, y, scores, strata, caliper, draws, seed):
    pooled = cond.auc(y, scores)
    n_pos, n_neg = int((y == 1).sum()), int((y != 1).sum())
    stats = cond.bootstrap_conditional(y, scores, strata, caliper, draws, seed)
    counts = cond.pair_counts(y, strata, caliper)

    print(f"  {name}")
    print(f"    conditional  {stats['auc']:.3f}  "
          f"95% CI [{stats['lo']:.3f}, {stats['hi']:.3f}]  "
          f"P(<= 0.5) = {stats['p_at_or_below_chance']:.3f}   "
          f"{counts['within']} within-stratum pairs")
    print(f"    pooled       {pooled:.3f}  +/- {cond.auc_se(pooled, n_pos, n_neg):.3f} "
          f"(Hanley-McNeil, unpaired -- do not judge a difference on it)")
    return {"name": name, "pooled": pooled, **stats, **counts}


def print_decomposition(y, scores, strata):
    rows = cond.pair_decomposition(y, scores, strata)
    print("  where the error sits  (pairs split by each member's stratum)")
    print(f"    {'pos/neg':>9s} {'pairs':>7s} {'misranked':>10s} "
          f"{'share of error':>15s}")
    for row in rows:
        mark = "  <- within" if row["within"] else ""
        print(f"    {row['pos_stratum']:>4d}/{row['neg_stratum']:<4d} "
              f"{row['pairs']:7d} {row['misranked']:9.1%} "
              f"{row['error_share']:14.1%}{mark}")
    within = sum(r["pairs"] for r in rows if r["within"])
    total = sum(r["pairs"] for r in rows)
    share = sum(r["error_share"] for r in rows if r["within"])
    print(f"    within-stratum pairs {within}/{total} ({within / total:.1%}), "
          f"carrying {share:.1%} of the error -- the addressable headroom")


def print_conversion(y, scores, strata):
    print("  what a conditional target is worth in pooled terms, on these studies")
    rows = cond.pooled_from_conditional(y, scores, strata, CONVERSION_TARGETS)
    line = "   ".join(f"{r['conditional']:.2f}->{r['pooled']:.3f}" for r in rows)
    print(f"    {line}")
    print("    the last entry is the ceiling under an unchanged gating -- "
          "arithmetic, not reachable by assumption")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, action="append",
                        help="parquet/csv of the 12 findings; repeatable, "
                             "optionally path:label")
    parser.add_argument("--baseline", default=None,
                        help="incumbent predictions, compared paired against "
                             "each --predictions")
    parser.add_argument("--truth", choices=("gold", "mined"), default="gold")
    parser.add_argument("--meta-dir", default=str(config.META_DIR))
    parser.add_argument("--finding", default="Synovitis")
    parser.add_argument("--stratify-by", default="Effusion")
    parser.add_argument("--stratify-on", choices=("gold", "pred"), default="gold")
    parser.add_argument("--threshold", type=float, default=None,
                        help="split point for --stratify-on pred (default: median)")
    parser.add_argument("--caliper", type=float, default=None,
                        help="treat the stratifier as continuous and pair "
                             "studies within this distance instead")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.stratify_on == "gold" and args.truth != "gold":
        parser.error("--stratify-on gold needs --truth gold; mined labels have "
                     "their own extraction error and would blur the strata")

    truth = load_truth(args.truth, Path(args.meta_dir))
    candidates = [read_predictions(spec) for spec in args.predictions]
    if args.baseline:
        candidates.append(read_predictions(args.baseline))

    shared = truth.index
    for frame, _ in candidates:
        shared = shared.intersection(frame.index)
    if len(shared) == 0:
        raise SystemExit("no studies in common between the truth and the "
                         "predictions -- wrong file, or wrong --truth")
    truth = truth.loc[shared]
    candidates = [(frame.loc[shared], name) for frame, name in candidates]

    reference = candidates[-1][0] if args.baseline else candidates[0][0]
    strata, strata_name = strata_for(args.stratify_on, truth, reference,
                                     args.stratify_by, args.threshold)
    if args.caliper is not None:
        strata = reference[args.stratify_by].to_numpy()
        strata_name = (f"predicted {args.stratify_by} "
                       f"(caliper {args.caliper:g})")

    y = truth[args.finding].to_numpy()
    print(f"finding     : {args.finding}")
    print(f"truth       : {args.truth}   ({len(shared)} studies, "
          f"{int((y == 1).sum())} pos / {int((y != 1).sum())} neg)")
    if args.truth == "mined":
        print("              NB mined labels measure agreement with the "
              "extractor, not with truth")
    print(f"stratified on: {strata_name}")
    print()

    print("scores")
    results = [report_one(name, y, frame[args.finding].to_numpy(), strata,
                          args.caliper, args.draws, args.seed)
               for frame, name in candidates]
    print()

    scored = candidates[0][0][args.finding].to_numpy()
    print_decomposition(y, scored, strata)
    print()
    print_conversion(y, scored, strata)
    print()

    # The claim that motivates the whole metric change: on this finding a
    # model's own effusion head is as good a synovitis classifier as its
    # synovitis head. Reported PER MODEL -- reading one model's gap off
    # another's effusion column is meaningless, and doing exactly that once
    # made a run look more gated than it was when its effusion head had simply
    # got better.
    print(f"  sanity: each model's own {args.stratify_by} head, used directly "
          f"as a {args.finding} classifier")
    print(f"    {'model':28s} {args.stratify_by:>10s} "
          f"{'as ' + args.finding:>14s} {args.finding:>10s} {'gap':>7s}")
    for frame, name in candidates:
        parent = frame[args.stratify_by].to_numpy()
        child = frame[args.finding].to_numpy()
        own = cond.auc(truth[args.stratify_by].to_numpy(), parent)
        print(f"    {name[:28]:28s} {own:10.3f} {cond.auc(y, parent):14.3f} "
              f"{cond.auc(y, child):10.3f} {cond.auc(y, parent) - cond.auc(y, child):+7.3f}")
    print("    A gap that widens because the parent head IMPROVED is not the "
          "same as one that")
    print("    widens because the child got more gated. The parent's own AUC "
          "is there to tell them apart.")
    print()

    if args.baseline:
        base_frame, base_name = candidates[-1]
        base = base_frame[args.finding].to_numpy()
        print(f"paired against {base_name}  (same studies, same resamples)")
        for frame, name in candidates[:-1]:
            scores = frame[args.finding].to_numpy()
            delta_c = cond.paired_bootstrap(y, scores, base, strata,
                                            args.caliper, args.draws, args.seed)
            delta_p = cond.paired_bootstrap(y, scores, base, None, None,
                                            args.draws, args.seed)
            print(f"  {name}")
            print(f"    conditional  {delta_c['delta']:+.3f}  "
                  f"95% CI [{delta_c['lo']:+.3f}, {delta_c['hi']:+.3f}]  "
                  f"P(not better) = {delta_c['p_a_not_better']:.3f}")
            print(f"    pooled       {delta_p['delta']:+.3f}  "
                  f"95% CI [{delta_p['lo']:+.3f}, {delta_p['hi']:+.3f}]  "
                  f"P(not better) = {delta_p['p_a_not_better']:.3f}")
        print()

    print("gate")
    for result in results:
        reading = gate_reading(result)
        why, action = GATE_ACTIONS[reading]
        print(f"  {result['name']}: {reading.upper()} -- {why}")
        print(f"    -> {action}")
    print()
    print("Standing rules: judge this finding on conditional AUC, paired, "
          "never on macro and never on pooled alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
