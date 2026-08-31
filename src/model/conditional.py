"""Effusion-stratified conditional AUC: the metric synovitis is judged on.

Pooled AUC cannot measure this finding. On the 58 gold studies the *gold
effusion label by itself* scores 0.698 as a synovitis classifier, against the
synovitis head's 0.701 -- so a number in that region is obtainable with no
synovitis-specific skill at all, and seven consecutive experiments read as
null partly because they were scored on it.

Conditional AUC is the concordance of positive/negative pairs drawn from
**within the same effusion stratum**. Cross-stratum pairs, where effusion
alone orders the pair correctly, are dropped. What remains is the only
quantity on this gold set that moves when synovitis-specific skill moves:

    pooled       0.701      837 pairs, ~55% of them cross-stratum
    conditional  0.630      376 within-stratum pairs

The two are linked by pair mass, which is why `pooled_from_conditional` can
state the exact pooled number a conditional target implies on this set --
conditional 0.75 is pooled 0.755, and a *perfect* conditional score is pooled
0.867. Anything above that is unreachable without changing the gating.

Deliberately free of torch and of pandas-at-import: this module is imported by
the analysis tools, bootstrapped into the notebooks, and used by the training
loop's validation hook, and `src/model/lightning.py` cannot be imported by any
of the first two without dragging in the Lightning framework.

Two rules this module exists to enforce, from the plan's standing rules:

1. Never judge synovitis on macro, and never on pooled alone.
2. Compare paired. Independent Hanley-McNeil standard errors are +/-0.07 here
   and carry between-study variance that a matched comparison removes; a real
   movement of 0.05 is invisible against them and visible against a paired
   bootstrap of the difference.
"""

from typing import Dict, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pooled AUC, and the noise band around it
# ---------------------------------------------------------------------------

def auc(y_true, scores) -> float:
    """ROC AUC via the rank statistic, ties averaged. Matches the training loop."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def auc_se(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC.

    Printed beside pooled numbers so a difference inside the band is visibly
    not evidence. Never used to judge a *difference* between two models on the
    same studies -- that is what `paired_bootstrap` is for.
    """
    if not np.isfinite(a) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    numerator = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                 + (n_neg - 1) * (q2 - a * a))
    return float(np.sqrt(max(numerator, 0.0) / (n_pos * n_neg)))


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def strata_from_scores(effusion, threshold: Optional[float] = None):
    """Binary effusion strata from a continuous effusion score.

    The plan stratifies on *predicted* effusion rather than on the mined
    effusion label, because the prediction scores AUC 0.888 while the mined
    label carries its own extraction error that would blur the strata. The
    default split is the median, which is what reproduces the plan's 0.617
    against the gold label's 0.630 -- close enough that the metric is
    computable during training, where no gold effusion label exists.

    A median split rather than a fixed probability because the head is not
    calibrated across runs: a 0.70 cut that balances one model's studies
    puts 80% of another model's on one side, and the number would then move
    between runs for a reason that has nothing to do with synovitis.
    """
    effusion = np.asarray(effusion, dtype=float)
    if threshold is None:
        threshold = float(np.median(effusion))
    return (effusion >= threshold).astype(int)


def comparable_pairs(y_true, strata, caliper: Optional[float] = None):
    """(n_pos, n_neg) boolean mask of pairs that share an effusion stratum.

    With `caliper=None`, `strata` is treated as categorical and pairs must
    match exactly -- the gold-label and median-split cases. With a caliper,
    `strata` is a continuous score and a pair is comparable when the two
    studies' effusion scores lie within `caliper` of each other, which is the
    matching rule the stratified training loss uses.
    """
    y_true = np.asarray(y_true)
    strata = np.asarray(strata)
    positives = y_true == 1
    left, right = strata[positives], strata[~positives]
    if caliper is None:
        return left[:, None] == right[None, :]
    return np.abs(left[:, None].astype(float)
                  - right[None, :].astype(float)) <= caliper


def _concordance(y_true, scores):
    """(n_pos, n_neg) matrix of 1 / 0.5 / 0 per pair. Ties count as half."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    difference = scores[positives][:, None] - scores[~positives][None, :]
    return (difference > 0).astype(float) + 0.5 * (difference == 0)


def conditional_auc(y_true, scores, strata,
                    caliper: Optional[float] = None) -> float:
    """Concordance over within-stratum pairs only. NaN if there are none.

    Pair-weighted, not stratum-averaged: the strata are 286 and 90 pairs on
    the gold set, and averaging them would give the 5-positive stratum equal
    say with the 22-positive one. Pair weighting is also what makes the
    conversion to pooled AUC exact.
    """
    mask = comparable_pairs(y_true, strata, caliper)
    total = mask.sum()
    if total == 0:
        return float("nan")
    return float((_concordance(y_true, scores) * mask).sum() / total)


def pair_counts(y_true, strata, caliper: Optional[float] = None) -> Dict[str, int]:
    """How many pairs the conditional number rests on, and the total."""
    mask = comparable_pairs(y_true, strata, caliper)
    return {"within": int(mask.sum()), "cross": int((~mask).sum()),
            "total": int(mask.size)}


# ---------------------------------------------------------------------------
# Where the error sits
# ---------------------------------------------------------------------------

def pair_decomposition(y_true, scores, strata) -> list:
    """Every discordant pair, split by the stratum of each member.

    The table in the plan's §1.4. Reading it is how the addressable headroom
    was located: within-stratum pairs are 44.9% of the total and are the only
    ones a synovitis-specific improvement can move. The cell where a
    stratum-0 positive meets a stratum-1 negative is ranked worse than chance
    -- that is the gating tax, and it is paid on five studies.
    """
    y_true = np.asarray(y_true)
    strata = np.asarray(strata)
    positives = y_true == 1
    concordance = _concordance(y_true, scores)
    error = 1.0 - concordance
    total_error = error.sum()

    rows = []
    for pos_level in sorted(set(strata[positives].tolist()), reverse=True):
        for neg_level in sorted(set(strata[~positives].tolist()), reverse=True):
            mask = ((strata[positives] == pos_level)[:, None]
                    & (strata[~positives] == neg_level)[None, :])
            pairs = int(mask.sum())
            if pairs == 0:
                continue
            cell_error = float((error * mask).sum())
            rows.append({
                "pos_stratum": int(pos_level), "neg_stratum": int(neg_level),
                "within": pos_level == neg_level,
                "pairs": pairs,
                "misranked": cell_error / pairs,
                "error_share": cell_error / total_error if total_error else float("nan"),
            })
    return rows


def pooled_from_conditional(y_true, scores, strata,
                            targets: Sequence[float]) -> list:
    """The pooled AUC each conditional target implies, on these studies.

    Cross-stratum concordance is held at what this model already achieves and
    only the within-stratum mass is moved, so the answer is arithmetic rather
    than a fit. It exists to keep two conversations in the same units: the
    leaderboard and every historical table are pooled, the plan's targets are
    conditional, and quoting one as the other is how a 0.63 -> 0.75 move gets
    dismissed as "+0.05, inside the noise".

    The value at `target=1.0` is the ceiling under an unchanged gating -- 0.867
    here. It is an arithmetic bound, not a reachable target: one model produces
    both halves, and a residualised model can buy within-stratum ordering by
    selling the gating.
    """
    mask = comparable_pairs(y_true, strata)
    concordance = _concordance(y_true, scores)
    within, cross = mask.sum(), (~mask).sum()
    cross_concordant = float((concordance * ~mask).sum())
    total = within + cross
    return [{"conditional": float(t),
             "pooled": float((cross_concordant + t * within) / total)}
            for t in targets]


# ---------------------------------------------------------------------------
# Uncertainty: paired, always
# ---------------------------------------------------------------------------

def _resamples(n: int, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(draws, n))


def bootstrap_conditional(y_true, scores, strata, caliper=None,
                          draws: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Percentile CI for one model's conditional AUC, plus P(AUC <= 0.5).

    Studies are resampled, not pairs: a study appears in many pairs and
    resampling pairs would treat those as independent observations and report
    an interval far narrower than the evidence supports.

    `p_at_or_below_chance` is the fraction of resamples at or under 0.5. It is
    the number the gate is read on, because "is there any synovitis-specific
    signal at all" is a question about which side of chance the estimate sits,
    not about whether it differs from some other model.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    strata = np.asarray(strata)

    values = []
    for index in _resamples(len(y_true), draws, seed):
        value = conditional_auc(y_true[index], scores[index], strata[index],
                                caliper)
        if np.isfinite(value):
            values.append(value)
    values = np.asarray(values)
    point = conditional_auc(y_true, scores, strata, caliper)
    if values.size == 0:
        return {"auc": point, "lo": float("nan"), "hi": float("nan"),
                "sd": float("nan"), "p_at_or_below_chance": float("nan"),
                "draws": 0}
    return {"auc": point,
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
            "sd": float(values.std(ddof=1)),
            "p_at_or_below_chance": float((values <= 0.5).mean()),
            "draws": int(values.size)}


def paired_bootstrap(y_true, scores_a, scores_b, strata=None, caliper=None,
                     draws: int = 4000, seed: int = 0) -> Dict[str, float]:
    """CI for (A - B) on the same studies, conditional or pooled.

    `strata=None` compares pooled AUCs; otherwise both models are scored on
    the same within-stratum pairs. Either way the *same* resample indexes both
    models, so between-study variance -- which is most of the +/-0.07 an
    unpaired standard error reports -- cancels.

    `p_a_not_better` is the fraction of resamples where the difference is at
    or below zero. Read it as the evidence against A, and pre-register the
    threshold before running, not after seeing it.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    strata = None if strata is None else np.asarray(strata)

    def statistic(y, a, b, s):
        if s is None:
            return auc(y, a) - auc(y, b)
        return (conditional_auc(y, a, s, caliper)
                - conditional_auc(y, b, s, caliper))

    values = []
    for index in _resamples(len(y_true), draws, seed):
        value = statistic(y_true[index], scores_a[index], scores_b[index],
                          None if strata is None else strata[index])
        if np.isfinite(value):
            values.append(value)
    values = np.asarray(values)
    point = statistic(y_true, scores_a, scores_b, strata)
    if values.size == 0:
        return {"delta": point, "lo": float("nan"), "hi": float("nan"),
                "p_a_not_better": float("nan"), "draws": 0}
    return {"delta": point,
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
            "p_a_not_better": float((values <= 0).mean()),
            "draws": int(values.size)}


# ---------------------------------------------------------------------------
# Calibrating the residual coupling
# ---------------------------------------------------------------------------

def logit(probability, floor: float = 1e-6):
    """Probabilities to logits, clipped off the asymptotes."""
    clipped = np.clip(np.asarray(probability, dtype=float), floor, 1 - floor)
    return np.log(clipped / (1 - clipped))


def calibrate_coupling(labels, parent_logits, iterations: int = 50) -> dict:
    """Fit `P(child=1) = sigmoid(a + b * parent_logit)` and return b.

    This is the beta for the residual parameterisation
    `child_logit = beta * stopgrad(parent_logit) + g(features)`, and where it
    comes from matters more than its value:

    - **Not** from the model's own +0.843 reconstruction coefficient. That is
      the coupling the current model already has, and reproducing it by
      construction would hand the residual head exactly the term it is
      supposed to replace.
    - **Not** from r_true = 0.403. A rank correlation between two binary
      labels does not map onto a logit coefficient, and the plan says so
      explicitly.

    It is fitted out of fold, on labels, by IRLS -- the coupling the *truth*
    supports between the two findings, measured on whatever label set is
    trustworthy enough to fit it. `separated` flags the degenerate case where
    the fit runs away because the parent perfectly orders the child; the
    coefficient is then meaningless and should not be used.
    """
    y = np.asarray(labels, dtype=float)
    x = np.asarray(parent_logits, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if y.size < 10 or y.std() == 0:
        return {"beta": float("nan"), "intercept": float("nan"),
                "n": int(y.size), "separated": True}

    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.zeros(2)
    for _ in range(iterations):
        # Clipped before the exponential: under separation the coefficients
        # run away and the naive form overflows on the way to being flagged.
        eta = np.clip(design @ coefficients, -700, 700)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(mu * (1 - mu), 1e-9, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return {"beta": float("nan"), "intercept": float("nan"),
                    "n": int(y.size), "separated": True}
        coefficients = coefficients + step
        if np.max(np.abs(step)) < 1e-8:
            break
    separated = bool(np.max(np.abs(coefficients)) > 20)
    return {"beta": float(coefficients[1]), "intercept": float(coefficients[0]),
            "n": int(y.size), "separated": separated}


# ---------------------------------------------------------------------------
# The Stage 0 gate
# ---------------------------------------------------------------------------
#
# The plan says "clearly > 0.5", "= 0.5" and "ambiguous" without giving
# numbers. They are fixed here, once, in code, so they cannot be chosen after
# seeing a result -- and here rather than in the tool because the training
# notebooks read the same gate and two copies would drift.

GATE_ACTIONS = {
    "clear": ("synovitis-specific signal survives conditioning on effusion",
              "Stage 1 -- train the conditional task directly"),
    "chance": ("the score is a better effusion proxy, nothing more",
               "Stage 3 (plan §4) or Stage 4 descope (plan §10) -- the plan "
               "gives both; decide before running either"),
    "ambiguous": ("expected at n=58; the CI straddles chance",
                  "Stage 1 at reduced confidence, and prioritise Stage 2 "
                  "(expand the gold set)"),
}


def gate_reading(stats: dict) -> str:
    """clear | chance | ambiguous, from a `bootstrap_conditional` result.

        clear      2.5th percentile above 0.5, and P(AUC <= 0.5) < 0.05
        chance     point estimate <= 0.55 and P(AUC <= 0.5) >= 0.25
        ambiguous  anything else -- the expected outcome at n=58
    """
    if not np.isfinite(stats.get("auc", float("nan"))):
        return "ambiguous"
    if stats["lo"] > 0.5 and stats["p_at_or_below_chance"] < 0.05:
        return "clear"
    if stats["auc"] <= 0.55 and stats["p_at_or_below_chance"] >= 0.25:
        return "chance"
    return "ambiguous"
