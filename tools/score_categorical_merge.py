"""How much is left in the categorical extractor, and can the 32B get it?

Two questions, answered locally in seconds:

1. **Ceilings.** If the errors found by `inspect_extraction_errors.py` were
   simply fixed, what would the fused gold macro become? These rows are
   computed *using the gold labels*, so they are upper bounds and nothing
   else -- no rule here is shippable, and reporting one as a result would be
   the 58-study overfit this project has already paid for twice. Their only
   job is to say whether the work is worth attempting at all.

2. **Merges.** Gold-blind rules that combine the 14B and 32B categorical
   frames. These *are* shippable, and are fixed here by structure rather
   than chosen by score:

     fill_silence   where the 14B said nothing and the 32B did, take the 32B.
                    Aimed straight at the diagnosed defect -- in-text
                    silences the 32B caught.
     mean_severity  average the two extractors' scores; the soft-ensemble
                    analogue of what was tried on the probability side.
     both_agree     assert a positive only when both models do. Aimed at the
                    dominant error class instead: 97 false positives.

    python3 tools/score_categorical_merge.py

Read every number against the measured noise floor: an independent redraw of
the same extraction moves the fused macro by ~0.005 (worklog 2026-08-26 h),
so a merge that gains less than about 0.01 has not been shown to gain
anything.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src import config                                    # noqa: E402
from src.labels import calibrate, eval_fusion, fuse, mode_fusion  # noqa: E402
from inspect_extraction_errors import anatomy_hits        # noqa: E402

SHIPPED = config.META_DIR / "llm_labels_gold.parquet"
CANDIDATE = config.META_DIR / "gold_sweep" / "llm_labels_gold__32b.parquet"
NOISE = 0.005           # measured run-to-run sd of the fused macro
TYPICAL_SEVERITY = 0.85  # "moderate", "definite" -- what a clean call scores


def blank_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy()


def set_call(frame, row, finding, present, severity=None, confidence=None):
    frame.loc[frame.index[row], finding] = present
    if severity is not None:
        frame.loc[frame.index[row], f"{finding}_severity"] = severity
    if confidence is not None:
        frame.loc[frame.index[row], f"{finding}_confidence"] = confidence


def get_call(frame, row, finding):
    present = frame[finding].iloc[row]
    severity = frame[f"{finding}_severity"].iloc[row]
    confidence = frame[f"{finding}_confidence"].iloc[row]
    return present, severity, confidence


# --------------------------------------------------------------------------
# Ceilings -- these read the gold labels. Upper bounds only.
# --------------------------------------------------------------------------

def oracle(base, gold, fix_silence=False, fix_said_no=False,
           fix_false_pos=False, only_reachable=False):
    out = blank_like(base)
    for finding in config.LABELS:
        truth = gold[finding].to_numpy().astype(int)
        in_text = [bool(anatomy_hits(r, finding, 1)) for r in gold["Report"]]
        for i in range(len(gold)):
            present, _, _ = get_call(base, i, finding)
            silent = pd.isna(present)
            if only_reachable and silent and not in_text[i]:
                continue          # never written down: not fixable by anyone
            if fix_silence and silent and truth[i] == 1:
                set_call(out, i, finding, 1.0, TYPICAL_SEVERITY, 1.0)
            elif fix_said_no and not silent and present == 0 and truth[i] == 1:
                set_call(out, i, finding, 1.0, TYPICAL_SEVERITY, 1.0)
            elif fix_false_pos and not silent and present == 1 and truth[i] == 0:
                set_call(out, i, finding, 0.0, 0.0, 1.0)
    return out


def perfect(base, gold):
    out = blank_like(base)
    for finding in config.LABELS:
        truth = gold[finding].to_numpy().astype(int)
        for i in range(len(gold)):
            set_call(out, i, finding, float(truth[i]),
                     TYPICAL_SEVERITY if truth[i] else 0.0, 1.0)
    return out


# --------------------------------------------------------------------------
# Merges -- gold-blind, shippable.
# --------------------------------------------------------------------------

def merge_fill_silence(a, b):
    out = blank_like(a)
    for finding in config.LABELS:
        for i in range(len(a)):
            present, _, _ = get_call(a, i, finding)
            if pd.isna(present):
                bp, bs, bc = get_call(b, i, finding)
                if not pd.isna(bp):
                    set_call(out, i, finding, bp, bs, bc)
    return out


def merge_mean_severity(a, b):
    out = blank_like(a)
    for finding in config.LABELS:
        for i in range(len(a)):
            ap, asev, ac = get_call(a, i, finding)
            bp, bsev, bc = get_call(b, i, finding)
            if pd.isna(ap) and pd.isna(bp):
                continue
            scores = [fuse._score(p, s) for p, s in ((ap, asev), (bp, bsev))
                      if not pd.isna(p)]
            mean = float(np.mean(scores))
            confs = [c for c in (ac, bc) if not pd.isna(c)]
            set_call(out, i, finding, 1.0 if mean > 0 else 0.0,
                     mean if mean > 0 else 0.0,
                     float(np.mean(confs)) if confs else 1.0)
    return out


def merge_both_agree(a, b):
    out = blank_like(a)
    for finding in config.LABELS:
        for i in range(len(a)):
            ap, asev, ac = get_call(a, i, finding)
            bp, _, _ = get_call(b, i, finding)
            if ap == 1 and bp != 1:
                # One model asserts, the other does not: demote to a negative
                # call rather than to silence, so the study still ranks.
                set_call(out, i, finding, 0.0, 0.0, ac)
    return out


def main():
    gold = calibrate.load_gold()
    lex = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    prob = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_probs_gold.parquet"), gold)
    cat14 = eval_fusion._align(pd.read_parquet(SHIPPED), gold)
    cat32 = eval_fusion._align(pd.read_parquet(CANDIDATE), gold)

    def fused(cat):
        scores = mode_fusion.combine(cat, prob, "score_average")
        return eval_fusion._score_frame_aucs(gold, fuse.fuse_scores(lex, scores))

    def cat_only(cat):
        return eval_fusion._score_frame_aucs(
            gold, mode_fusion.from_categorical(cat))

    base = float(np.nanmean(fused(cat14).to_numpy()))

    rows = [
        ("14B (shipped)", cat14, "baseline"),
        ("32B", cat32, "measured"),
        ("merge: fill_silence", merge_fill_silence(cat14, cat32), "shippable"),
        ("merge: mean_severity", merge_mean_severity(cat14, cat32), "shippable"),
        ("merge: both_agree", merge_both_agree(cat14, cat32), "shippable"),
        ("CEILING: fix in-text silences",
         oracle(cat14, gold, fix_silence=True, only_reachable=True), "ceiling"),
        ("CEILING: fix said-no",
         oracle(cat14, gold, fix_said_no=True), "ceiling"),
        ("CEILING: fix false positives",
         oracle(cat14, gold, fix_false_pos=True), "ceiling"),
        ("CEILING: fix all reachable",
         oracle(cat14, gold, fix_silence=True, fix_said_no=True,
                fix_false_pos=True, only_reachable=True), "ceiling"),
        ("CEILING: perfect extraction", perfect(cat14, gold), "ceiling"),
    ]

    print("Fused gold macro (the number labels are gated on), vs shipped "
          f"{base:.3f}")
    print(f"Measured noise floor: +-{NOISE:.3f} per redraw, so a gain under "
          f"~{2*NOISE:.2f} is not a demonstrated gain.\n")
    print(f"{'configuration':32s} {'cat-only':>9s} {'fused':>7s} "
          f"{'delta':>7s}   {'kind':>9s}")
    print("-" * 72)
    for name, frame, kind in rows:
        f_macro = float(np.nanmean(fused(frame).to_numpy()))
        c_macro = float(np.nanmean(cat_only(frame).to_numpy()))
        flag = ""
        if kind == "shippable":
            flag = "  <-- real" if f_macro - base > 2 * NOISE else "  (noise)"
        print(f"{name:32s} {c_macro:9.3f} {f_macro:7.3f} "
              f"{f_macro - base:+7.3f}   {kind:>9s}{flag}")
    print("-" * 72)
    print("\nCEILING rows use the gold labels and are upper bounds, not "
          "methods. They answer 'is this worth chasing', nothing more.")


if __name__ == "__main__":
    main()
