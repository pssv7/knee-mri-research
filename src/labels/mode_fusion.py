"""Combine the categorical and probability extraction modes into one score.

The two LLM extraction modes fail in complementary ways. The categorical mode
is precise where reports state a finding crisply (Medial Meniscus 0.944, MCL
0.959 on gold) but collapses more than half of all label slots onto a single
value, and AUC cannot order studies inside a tie block. The probability mode
spreads the silent studies (Fracture 0.885 vs 0.752) but leaves Medial
Meniscus and MCL nearly all tied. Best-of-both per finding scores ~0.869
against 0.854 for the categorical-only fusion -- but picking a winner per
finding on 58 gold studies is the same overfit that sank the fitted fusion
weights (0.860 in-sample, 0.812 leave-one-out).

Every strategy here therefore applies one rule globally, with at most one
global parameter, so in-sample and held-out are the same number. Per-finding
selection is deliberately not offered.

Measured on the 32B gold dump (2026-08-23): categorical alone 0.838,
probability alone 0.860, `score_average` and `rank_average` tied at 0.889 --
above the 0.885 per-finding oracle, because averaging complementary errors
beats choosing between them. The delete-8 jackknife split 0.54/0.47 between
the two, which is a tie, and the tie-break rule is simplicity: freeze
`score_average`. It also keeps a probability-like scale (mean target 0.33
against gold prevalence 0.35, where rank scores are uniform at 0.5 by
construction), which matters because these become BCE targets downstream --
imputing silence priors as targets measured worse than zero every time.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.labels import fuse

# Frozen from the gold-dump comparison recorded in the module docstring.
DEFAULT_STRATEGY = "score_average"

# Tiebreak weight. Correctness only needs it below the smallest gap between
# distinct categorical scores (severity x certainty products; adjacent values
# sit ~0.05 apart), so 0.01 is not a tuned quantity.
TIEBREAK_EPSILON = 0.01

# Global blend weight on the categorical rank for the `global_blend` strategy.
# Must be set from a plateau, never an argmax -- see eval_fusion.py.
GLOBAL_BLEND_ALPHA = 0.5

STRATEGIES = (
    "score_average",
    "rank_average",
    "cat_primary_tiebreak",
    "global_blend",
    "confidence_weighted",
)


def _tied_ranks_01(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks rescaled to [0, 1]; NaN positions come back NaN."""
    result = np.full(len(values), np.nan)
    mask = ~np.isnan(values)
    n = int(mask.sum())
    if n == 0:
        return result
    if n == 1:
        result[mask] = 0.5
        return result

    subset = values[mask]
    order = np.argsort(subset, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    unique, inverse, counts = np.unique(subset, return_inverse=True,
                                        return_counts=True)
    tie_sums = np.zeros(len(unique))
    np.add.at(tie_sums, inverse, ranks)
    ranks = (tie_sums / counts)[inverse]

    result[mask] = (ranks - 1.0) / (n - 1.0)
    return result


def _categorical_scores(cat_frame: pd.DataFrame, name: str) -> np.ndarray:
    """Map one finding's categorical columns through the fusion score."""
    return np.array([
        fuse._score(p, s) for p, s in
        zip(cat_frame[name], cat_frame[f"{name}_severity"])
    ])


def _categorical_informative(cat_frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.array([fuse._informative(p) for p in cat_frame[name]])


def _combine_finding(cat_score, cat_ok, prob, prob_conf, cat_conf,
                     strategy: str) -> np.ndarray:
    """Combine one finding's two score vectors. `prob` may contain NaN."""
    has_prob = ~np.isnan(prob)

    if strategy == "score_average":
        # Plain mean of the two scores. On the gold dump this matched rank
        # averaging's AUC exactly while keeping a probability-like scale --
        # its mean target sits at the gold prevalence, where rank scores are
        # uniform by construction. Scale matters because these become BCE
        # targets downstream.
        return np.where(has_prob, 0.5 * (cat_score + prob), cat_score)

    if strategy == "cat_primary_tiebreak":
        tiebreak = np.where(has_prob, prob, 0.0)
        combined = cat_score + TIEBREAK_EPSILON * tiebreak
        return combined / (1.0 + TIEBREAK_EPSILON)

    if strategy == "rank_average":
        cat_rank = _tied_ranks_01(cat_score)
        prob_rank = _tied_ranks_01(np.where(has_prob, prob, np.nan))
        return np.where(has_prob, 0.5 * (cat_rank + prob_rank), cat_rank)

    if strategy == "global_blend":
        alpha = GLOBAL_BLEND_ALPHA
        cat_rank = _tied_ranks_01(cat_score)
        prob_rank = _tied_ranks_01(np.where(has_prob, prob, np.nan))
        return np.where(has_prob,
                        alpha * cat_rank + (1.0 - alpha) * prob_rank,
                        cat_rank)

    if strategy == "confidence_weighted":
        w_cat = np.where(cat_ok, np.nan_to_num(cat_conf, nan=0.0), 0.0)
        w_prob = np.where(has_prob, np.nan_to_num(prob_conf, nan=0.0), 0.0)
        total = w_cat + w_prob
        # Neither mode confident: fall back to the categorical score, which
        # is zero for silence anyway.
        blended = np.where(
            total > 0,
            (w_cat * cat_score + w_prob * np.where(has_prob, prob, 0.0))
            / np.where(total > 0, total, 1.0),
            cat_score,
        )
        return blended

    raise ValueError(f"unknown strategy {strategy!r}; one of {STRATEGIES}")


def combine(cat_frame: pd.DataFrame, prob_frame: pd.DataFrame,
            strategy: str = DEFAULT_STRATEGY) -> pd.DataFrame:
    """Merge the two extraction modes into one score frame.

    Output has, per finding: `<name>` (score in [0, 1]), `<name>_informative`
    (the *categorical* mode asserted something -- see the comment below, this
    deliberately excludes the probability mode's context-guesses), and
    `<name>_call` (the positive/negative call used for agreement weighting
    downstream -- the categorical call where it exists, else probability
    >= 0.5).
    """
    merged = cat_frame.merge(prob_frame, on=config.ID_COL,
                             suffixes=("", "_prob"))
    if len(merged) != len(cat_frame):
        raise ValueError(
            f"mode merge lost rows: {len(cat_frame)} -> {len(merged)}")

    output = {config.ID_COL: merged[config.ID_COL].to_numpy()}
    for name in config.LABELS:
        cat_score = _categorical_scores(merged, name)
        cat_ok = _categorical_informative(merged, name)
        cat_conf = merged.get(
            f"{name}_confidence",
            pd.Series(1.0, index=merged.index)).to_numpy(dtype=float)
        prob = merged[f"{name}_prob"].to_numpy(dtype=float)
        prob_conf_col = merged.get(f"{name}_confidence_prob")
        prob_conf = (prob_conf_col.to_numpy(dtype=float)
                     if prob_conf_col is not None
                     else np.ones(len(merged)))
        has_prob = ~np.isnan(prob)

        combined = _combine_finding(cat_score, cat_ok, prob, prob_conf,
                                    cat_conf, strategy)
        output[name] = np.clip(combined, 0.0, 1.0)
        # Informative means an extractor *asserted* something, which only the
        # categorical mode tracks. The probability mode is never silent -- its
        # prompt asks it to estimate unmentioned findings from context -- so
        # `cat_ok | has_prob` would be True everywhere and would silently
        # retire the silence down-weighting. That is not a free change: 34.5%
        # of slots are both-silent, and lifting them from WEIGHT_SILENT to
        # WEIGHT_SINGLE puts 2.8x the training mass on the least trustworthy
        # third of the corpus. The probability estimate still ranks studies
        # inside that block, which is the tie-breaking gain worth having --
        # the value is kept, the certainty is not asserted.
        output[f"{name}_informative"] = cat_ok
        output[f"{name}_call"] = np.where(
            cat_ok, cat_score > 0, np.where(has_prob, prob >= 0.5, False)
        ).astype(bool)

    return pd.DataFrame(output)


def from_categorical(cat_frame: pd.DataFrame) -> pd.DataFrame:
    """A score frame from the categorical mode alone.

    The baseline every strategy is compared against, and the shape `combine`
    degrades to when the probability frame has nothing to say.
    """
    output = {config.ID_COL: cat_frame[config.ID_COL].to_numpy()}
    for name in config.LABELS:
        score = _categorical_scores(cat_frame, name)
        ok = _categorical_informative(cat_frame, name)
        output[name] = score
        output[f"{name}_informative"] = ok
        output[f"{name}_call"] = (score > 0) & ok
    return pd.DataFrame(output)
