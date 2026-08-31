"""Tests for the effusion-stratified conditional AUC.

The metric exists because pooled AUC on synovitis is reproducible from the
effusion label alone, so the properties worth guarding are the ones that make
it a different measurement rather than a rescaling of the old one:

- a score that is a pure function of effusion must read at chance
  conditionally, whatever it scores pooled;
- the conditional-to-pooled conversion must be an identity at the model's own
  conditional value, or the two conversations are not in the same units;
- the bootstrap must resample studies, not pairs -- resampling pairs would
  report an interval far narrower than 58 studies support.

The last class re-derives the plan's published figures from the real gold
files when they are present, so a change to the metric that moves 0.630 or
0.867 fails here rather than in a worklog entry six weeks later.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import conditional as cond  # noqa: E402

META = os.path.join(ROOT, "data", "meta")
GOLD_TRUTH = os.path.join(META, "gold_truth_58.parquet")
GOLD_PREDICTIONS = os.path.join(ROOT, "failures", "gold_predictions.parquet")


def planted(n=200, seed=0, gated=True):
    """A gated model: its synovitis score is a monotone function of effusion.

    The failure this project actually has. Pooled AUC rewards it, because
    effusion and synovitis co-occur; conditional AUC must not.
    """
    rng = np.random.default_rng(seed)
    effusion = rng.integers(0, 2, size=n)
    # Synovitis co-occurs with effusion but is not determined by it.
    synovitis = np.where(rng.random(n) < np.where(effusion == 1, 0.7, 0.3), 1, 0)
    effusion_score = effusion + rng.normal(0, 0.2, size=n)
    if gated:
        scores = effusion_score
    else:
        scores = synovitis + rng.normal(0, 0.4, size=n)
    return synovitis, scores, effusion, effusion_score


class TestAGatedScoreReadsAtChance:
    def test_pooled_rewards_it_and_conditional_does_not(self):
        y, scores, effusion, _ = planted(400)
        assert cond.auc(y, scores) > 0.6
        assert abs(cond.conditional_auc(y, scores, effusion) - 0.5) < 0.05

    def test_a_real_signal_survives_conditioning(self):
        y, scores, effusion, _ = planted(400, gated=False)
        assert cond.conditional_auc(y, scores, effusion) > 0.7

    def test_one_stratum_makes_it_the_pooled_number(self):
        y, scores, _, _ = planted(200, gated=False)
        flat = np.zeros(len(y), dtype=int)
        assert cond.conditional_auc(y, scores, flat) == pytest.approx(
            cond.auc(y, scores))


class TestPairAccounting:
    def test_within_and_cross_partition_every_pair(self):
        y, _, effusion, _ = planted(120)
        counts = cond.pair_counts(y, effusion)
        assert counts["within"] + counts["cross"] == counts["total"]
        assert counts["total"] == int((y == 1).sum()) * int((y != 1).sum())

    def test_decomposition_shares_sum_to_one(self):
        y, scores, effusion, _ = planted(120)
        rows = cond.pair_decomposition(y, scores, effusion)
        assert sum(row["error_share"] for row in rows) == pytest.approx(1.0)
        assert sum(row["pairs"] for row in rows) == \
            int((y == 1).sum()) * int((y != 1).sum())

    def test_conversion_is_an_identity_at_the_model_s_own_value(self):
        y, scores, effusion, _ = planted(150)
        here = cond.conditional_auc(y, scores, effusion)
        converted = cond.pooled_from_conditional(y, scores, effusion, [here])
        assert converted[0]["pooled"] == pytest.approx(cond.auc(y, scores))

    def test_the_ceiling_is_below_one(self):
        # A perfect conditional score still cannot fix cross-stratum pairs the
        # model gets wrong, so the bound is strictly under 1.0.
        y, scores, effusion, _ = planted(150)
        ceiling = cond.pooled_from_conditional(y, scores, effusion, [1.0])[0]
        assert ceiling["pooled"] < 1.0


class TestStrata:
    def test_the_median_split_balances_the_strata(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9])
        assert cond.strata_from_scores(scores).sum() == 2

    def test_a_caliper_matches_on_distance_instead(self):
        y = np.array([1, 0, 0])
        effusion = np.array([0.10, 0.15, 0.90])
        mask = cond.comparable_pairs(y, effusion, caliper=0.1)
        assert mask.tolist() == [[True, False]]


class TestUncertainty:
    def test_the_interval_brackets_the_point_estimate(self):
        y, scores, effusion, _ = planted(120, gated=False)
        stats = cond.bootstrap_conditional(y, scores, effusion, draws=400)
        assert stats["lo"] < stats["auc"] < stats["hi"]

    def test_resampling_studies_not_pairs(self):
        # 837 pairs from 58 studies. A pair-level bootstrap would report an SD
        # near 1/sqrt(837) ~ 0.017; the study-level one is several times wider.
        y, scores, effusion, _ = planted(58, gated=False)
        stats = cond.bootstrap_conditional(y, scores, effusion, draws=600)
        assert stats["sd"] > 0.04

    def test_a_model_against_itself_is_exactly_zero(self):
        y, scores, effusion, _ = planted(80)
        delta = cond.paired_bootstrap(y, scores, scores, effusion, draws=200)
        assert delta["delta"] == 0.0
        assert delta["lo"] == delta["hi"] == 0.0

    def test_pairing_is_tighter_than_an_unpaired_standard_error(self):
        # Two models differing by a small shared-direction nudge. The paired
        # interval on the difference must be far narrower than the
        # Hanley-McNeil SE of either model alone, which is the whole reason
        # the standing rules require paired tests.
        y, scores, effusion, _ = planted(58, gated=False)
        nudged = scores + 0.05
        delta = cond.paired_bootstrap(y, scores, nudged, effusion, draws=400)
        unpaired = cond.auc_se(cond.auc(y, scores), int((y == 1).sum()),
                               int((y != 1).sum()))
        assert (delta["hi"] - delta["lo"]) < unpaired


class TestCalibrateCoupling:
    def test_it_recovers_a_planted_coefficient(self):
        rng = np.random.default_rng(0)
        parent = rng.normal(0, 2, size=8000)
        probability = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * parent)))
        child = (rng.random(8000) < probability).astype(float)
        fitted = cond.calibrate_coupling(child, parent)
        assert fitted["beta"] == pytest.approx(0.8, abs=0.08)
        assert not fitted["separated"]

    def test_perfect_separation_is_flagged_not_returned(self):
        parent = np.linspace(-5, 5, 200)
        child = (parent > 0).astype(float)
        assert cond.calibrate_coupling(child, parent)["separated"]


@pytest.mark.skipif(not (os.path.exists(GOLD_TRUTH)
                         and os.path.exists(GOLD_PREDICTIONS)),
                    reason="gold truth or reference predictions not present")
class TestTheReferenceNumbers:
    """The plan's §1 figures, re-derived. These are the project's constants."""

    @staticmethod
    def load():
        import pandas as pd
        truth = pd.read_parquet(GOLD_TRUTH).set_index("StudyInstanceUID")
        preds = (pd.read_parquet(GOLD_PREDICTIONS)
                 .set_index("StudyInstanceUID").loc[truth.index])
        return (truth["Synovitis"].to_numpy(), preds["Synovitis"].to_numpy(),
                truth["Effusion"].to_numpy(), preds["Effusion"].to_numpy())

    def test_pooled_and_conditional(self):
        y, scores, effusion, _ = self.load()
        assert cond.auc(y, scores) == pytest.approx(0.701, abs=0.001)
        assert cond.conditional_auc(y, scores, effusion) == \
            pytest.approx(0.630, abs=0.001)

    def test_the_effusion_head_scores_synovitis_as_well_as_the_synovitis_head(self):
        y, scores, _, effusion_score = self.load()
        assert cond.auc(y, effusion_score) >= cond.auc(y, scores)

    def test_predicted_strata_track_the_gold_ones(self):
        y, scores, effusion, effusion_score = self.load()
        predicted = cond.strata_from_scores(effusion_score)
        assert cond.conditional_auc(y, scores, predicted) == \
            pytest.approx(0.617, abs=0.002)

    def test_the_pair_decomposition(self):
        y, scores, effusion, _ = self.load()
        rows = {(r["pos_stratum"], r["neg_stratum"]): r
                for r in cond.pair_decomposition(y, scores, effusion)}
        assert rows[(1, 1)]["pairs"] == 286
        assert rows[(1, 1)]["error_share"] == pytest.approx(0.444, abs=0.001)
        # The gating tax: a positive without effusion against a negative with
        # one is ranked worse than a coin toss.
        assert rows[(0, 1)]["misranked"] > 0.5

    def test_the_ceiling_under_an_unchanged_gating(self):
        y, scores, effusion, _ = self.load()
        table = {r["conditional"]: r["pooled"] for r in
                 cond.pooled_from_conditional(y, scores, effusion,
                                              [0.75, 1.00])}
        assert table[0.75] == pytest.approx(0.755, abs=0.001)
        assert table[1.00] == pytest.approx(0.867, abs=0.001)
