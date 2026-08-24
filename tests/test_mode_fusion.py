"""Tests for combining the categorical and probability extraction modes.

The properties that matter: no strategy fits anything per finding, the
tiebreak strategy can never reorder what the categorical extractor already
separates, and a probability frame with nothing to say degrades every
strategy to the categorical baseline. The last one is what keeps `build()`
working on machines that never downloaded the probability dumps.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import fuse, mode_fusion  # noqa: E402


def cat_frame(present, severity, confidence=None, finding="ACL"):
    """A categorical extractor frame with one finding populated."""
    n = len(present)
    data = {config.ID_COL: [f"s{i}" for i in range(n)]}
    for name in config.LABELS:
        active = name == finding
        data[name] = list(present) if active else [None] * n
        data[f"{name}_severity"] = list(severity) if active else [None] * n
        data[f"{name}_confidence"] = (
            list(confidence) if active and confidence is not None
            else [1.0] * n)
    return pd.DataFrame(data)


def prob_frame(probs, confidence=None, finding="ACL"):
    """A probability extractor frame with one finding populated."""
    n = len(probs)
    data = {config.ID_COL: [f"s{i}" for i in range(n)]}
    for name in config.LABELS:
        active = name == finding
        data[name] = list(probs) if active else [np.nan] * n
        data[f"{name}_confidence"] = (
            list(confidence) if active and confidence is not None
            else [1.0] * n)
    return pd.DataFrame(data)


def tied_rank_order(values):
    """Tie-averaged ranks, for comparing orderings between score vectors."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    unique, inverse, counts = np.unique(values, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


class TestTiebreak:
    def test_categorical_order_is_preserved(self):
        # Distinct categorical scores must stay in their order no matter what
        # the probabilities say -- epsilon is smaller than any score gap.
        cat = cat_frame([1, 1, 0, 1], [1.0, 0.85, None, 0.35])
        prob = prob_frame([0.0, 1.0, 1.0, 1.0])
        combined = mode_fusion.combine(cat, prob, "cat_primary_tiebreak")
        scores = combined["ACL"].to_numpy()
        assert scores[0] > scores[1] > scores[3] > scores[2]

    def test_probability_orders_inside_a_tie_block(self):
        # Four studies the categorical extractor cannot separate.
        cat = cat_frame([0, 0, 0, 0], [None] * 4)
        prob = prob_frame([0.1, 0.9, 0.4, 0.7])
        combined = mode_fusion.combine(cat, prob, "cat_primary_tiebreak")
        scores = combined["ACL"].to_numpy()
        assert list(np.argsort(scores)) == [0, 2, 3, 1]

    def test_nan_probability_contributes_nothing(self):
        cat = cat_frame([1, 1], [1.0, 1.0])
        prob = prob_frame([np.nan, 0.0])
        combined = mode_fusion.combine(cat, prob, "cat_primary_tiebreak")
        assert combined["ACL"].iloc[0] == combined["ACL"].iloc[1]


class TestRankAverage:
    def test_probability_orders_inside_a_tie_block(self):
        cat = cat_frame([0, 0, 0, 1], [None, None, None, 1.0])
        prob = prob_frame([0.8, 0.2, 0.5, 0.9])
        combined = mode_fusion.combine(cat, prob, "rank_average")
        scores = combined["ACL"].to_numpy()
        # Within the tie block, probability decides; the categorical positive
        # still outranks all of it.
        assert scores[3] == max(scores)
        assert scores[1] < scores[2] < scores[0]

    def test_nan_probability_falls_back_to_categorical_rank(self):
        cat = cat_frame([1, 0, 0], [1.0, None, None])
        prob = prob_frame([np.nan, np.nan, np.nan])
        combined = mode_fusion.combine(cat, prob, "rank_average")
        scores = combined["ACL"].to_numpy()
        assert scores[0] == max(scores)
        assert scores[1] == scores[2]


class TestDegradation:
    """An empty probability frame must reproduce categorical-only ordering."""

    @pytest.mark.parametrize("strategy", mode_fusion.STRATEGIES)
    def test_all_nan_probabilities_keep_categorical_order(self, strategy):
        cat = cat_frame([1, 0, 1, 0, 1], [1.0, None, 0.35, 0.0, 0.85])
        prob = prob_frame([np.nan] * 5)
        combined = mode_fusion.combine(cat, prob, strategy)
        cat_scores = mode_fusion.from_categorical(cat)["ACL"].to_numpy()
        assert np.allclose(tied_rank_order(combined["ACL"].to_numpy()),
                           tied_rank_order(cat_scores))


class TestConfidenceWeighted:
    def test_zero_probability_confidence_yields_categorical(self):
        cat = cat_frame([1, 0], [0.85, None])
        prob = prob_frame([0.2, 0.9], confidence=[0.0, 0.0])
        combined = mode_fusion.combine(cat, prob, "confidence_weighted")
        expected = mode_fusion.from_categorical(cat)["ACL"].to_numpy()
        assert np.allclose(combined["ACL"].to_numpy(), expected)

    def test_equal_confidence_averages(self):
        cat = cat_frame([1], [1.0])
        prob = prob_frame([0.5])
        combined = mode_fusion.combine(cat, prob, "confidence_weighted")
        assert combined["ACL"].iloc[0] == pytest.approx(0.75)


class TestFrameContract:
    def test_scores_stay_in_unit_range(self):
        cat = cat_frame([1, 0, None], [1.0, None, None])
        prob = prob_frame([1.0, 0.0, np.nan])
        for strategy in mode_fusion.STRATEGIES:
            scores = mode_fusion.combine(cat, prob, strategy)["ACL"]
            assert ((scores >= 0.0) & (scores <= 1.0)).all()

    def test_informative_is_either_mode(self):
        cat = cat_frame([1, None, None, None], [1.0, None, None, None])
        prob = prob_frame([np.nan, 0.3, np.nan, 0.9])
        combined = mode_fusion.combine(cat, prob)
        assert list(combined["ACL_informative"]) == [True, True, False, True]

    def test_call_prefers_the_categorical_verdict(self):
        # A categorical negative with a high probability stays a negative
        # call; probability only decides where the categorical mode is silent.
        cat = cat_frame([0, None, None], [None, None, None])
        prob = prob_frame([0.9, 0.9, 0.2])
        combined = mode_fusion.combine(cat, prob)
        assert list(combined["ACL_call"]) == [False, True, False]

    def test_unknown_strategy_raises(self):
        cat = cat_frame([1], [1.0])
        prob = prob_frame([0.5])
        with pytest.raises(ValueError, match="unknown strategy"):
            mode_fusion.combine(cat, prob, "argmax_per_finding")

    def test_row_loss_raises(self):
        cat = cat_frame([1], [1.0])
        prob = prob_frame([0.5])
        prob[config.ID_COL] = ["different-uid"]
        with pytest.raises(ValueError, match="merge lost rows"):
            mode_fusion.combine(cat, prob)


class TestFuseScoresEquivalence:
    """`fuse_scores` over a categorical-derived frame must equal `fuse`.

    This is the guarantee behind the `build()` fallback: without probability
    dumps the new path reproduces the old labels bit for bit.
    """

    def _lex_and_llm(self):
        rng = np.random.default_rng(7)
        n = 40
        present_choices = [1, 0, None]
        lex = {config.ID_COL: [f"s{i}" for i in range(n)]}
        llm = {config.ID_COL: [f"s{i}" for i in range(n)]}
        for name in config.LABELS:
            lex[name] = list(rng.choice(present_choices, n))
            lex[f"{name}_severity"] = list(rng.choice([1.0, 0.5, None], n))
            llm[name] = list(rng.choice(present_choices, n))
            llm[f"{name}_severity"] = list(rng.choice([1.0, 0.35, None], n))
        return pd.DataFrame(lex), pd.DataFrame(llm)

    def test_targets_and_weights_match(self):
        lex, llm = self._lex_and_llm()
        old = fuse.fuse(lex, llm)
        new = fuse.fuse_scores(lex, mode_fusion.from_categorical(llm))
        for name in config.LABELS:
            assert np.allclose(old[name], new[name]), name
            assert np.allclose(old[f"{name}_weight"],
                               new[f"{name}_weight"]), name


class TestNoPerFindingFitting:
    def test_strategies_carry_at_most_one_global_parameter(self):
        # The guard against reintroducing the 58-study overfit: nothing in the
        # module may key a constant off a finding name.
        source = (config.PROJECT_ROOT / "src" / "labels"
                  / "mode_fusion.py").read_text()
        for name in config.LABELS:
            assert f'"{name}"' not in source.replace(
                "config.LABELS", ""), name
