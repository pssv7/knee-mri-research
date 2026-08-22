"""Tests for extractor fusion.

The property that matters most here is that nothing is fitted on the 58 gold
studies. An earlier version derived blend weights and silence priors from them
and scored 0.860 in-sample against 0.812 held out -- worse than using the LLM
extractor alone. These tests guard the parameter-free replacement.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import calibrate, fuse  # noqa: E402

MINED = config.META_DIR / "llm_labels.parquet"
GOLD_LLM = config.META_DIR / "llm_labels_gold.parquet"

requires_llm_labels = pytest.mark.skipif(
    not (MINED.exists() and GOLD_LLM.exists()),
    reason="LLM label parquets not present",
)


def frames(lex_present, lex_sev, llm_present, llm_sev, finding="ACL"):
    """Minimal one-row extractor frames for a single finding."""
    lex = {config.ID_COL: ["s1"]}
    llm = {config.ID_COL: ["s1"]}
    for name in config.LABELS:
        lex[name] = [lex_present if name == finding else None]
        lex[f"{name}_severity"] = [lex_sev if name == finding else None]
        llm[name] = [llm_present if name == finding else None]
        llm[f"{name}_severity"] = [llm_sev if name == finding else None]
    return pd.DataFrame(lex), pd.DataFrame(llm)


class TestBlend:
    def test_weights_sum_to_one(self):
        assert fuse.LEXICON_WEIGHT + fuse.LLM_WEIGHT == pytest.approx(1.0)

    def test_target_is_the_weighted_sum(self):
        lex, llm = frames(1, 1.0, 1, 0.5)
        result = fuse.fuse(lex, llm)
        expected = fuse.LEXICON_WEIGHT * 1.0 + fuse.LLM_WEIGHT * 0.5
        assert result["ACL"].iloc[0] == pytest.approx(expected)

    def test_llm_carries_more_than_lexicon(self):
        # The LLM beats the lexicon on eleven of twelve findings.
        assert fuse.LLM_WEIGHT > fuse.LEXICON_WEIGHT

    def test_targets_stay_in_unit_range(self):
        for lex_sev, llm_sev in [(1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]:
            lex, llm = frames(1, lex_sev, 1, llm_sev)
            value = fuse.fuse(lex, llm)["ACL"].iloc[0]
            assert 0.0 <= value <= 1.0


class TestSilence:
    def test_silence_scores_zero(self):
        # Using a measured P(positive | silent) as the target was worse than
        # zero on every variant tested; the weight carries that uncertainty.
        lex, llm = frames(None, None, None, None)
        assert fuse.fuse(lex, llm)["ACL"].iloc[0] == 0.0

    def test_silence_gets_the_lowest_weight(self):
        lex, llm = frames(None, None, None, None)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_SILENT
        assert fuse.WEIGHT_SILENT < fuse.WEIGHT_DISAGREE < fuse.WEIGHT_AGREE

    def test_nan_is_treated_as_silence(self):
        # Parquet round-trips None to NaN, so both must behave identically.
        lex, llm = frames(np.nan, np.nan, np.nan, np.nan)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_SILENT


class TestAgreementWeighting:
    def test_agreement_gets_full_weight(self):
        lex, llm = frames(1, 1.0, 1, 1.0)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_AGREE

    def test_agreed_negative_also_gets_full_weight(self):
        lex, llm = frames(0, 0.0, 0, 0.0)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_AGREE

    def test_disagreement_is_downweighted(self):
        lex, llm = frames(1, 1.0, 0, 0.0)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_DISAGREE

    def test_single_informative_extractor(self):
        lex, llm = frames(None, None, 1, 1.0)
        assert fuse.fuse(lex, llm)["ACL_weight"].iloc[0] == fuse.WEIGHT_SINGLE


class TestMergeSafety:
    def test_row_loss_raises(self):
        lex, llm = frames(1, 1.0, 1, 1.0)
        llm[config.ID_COL] = ["different-uid"]
        with pytest.raises(ValueError, match="merge lost rows"):
            fuse.fuse(lex, llm)


@requires_llm_labels
class TestAgainstGold:
    @pytest.fixture(scope="class")
    @classmethod
    def built(cls):
        fused, fused_gold, aucs = fuse.build()
        return fused, fused_gold, aucs, calibrate.load_gold()

    def test_covers_every_unlabelled_study(self, built):
        fused = built[0]
        assert len(fused) == 4349
        assert fused[config.ID_COL].duplicated().sum() == 0

    def test_no_missing_targets(self, built):
        fused = built[0]
        for name in config.LABELS:
            assert fused[name].notna().all()
            assert fused[f"{name}_weight"].notna().all()

    def test_beats_both_extractors_alone(self, built):
        _, fused_gold, aucs, gold = built
        fused_macro = np.mean([
            calibrate._auc(gold[name].to_numpy().astype(int),
                           fused_gold[name].to_numpy())
            for name in config.LABELS
        ])
        assert fused_macro > aucs["llm_auc"].mean()
        assert fused_macro > aucs["lexicon_auc"].mean()
        assert fused_macro > 0.84

    def test_every_finding_clears_the_kill_threshold(self, built):
        # Under the lexicon alone five findings would have been dropped from
        # the loss entirely.
        _, fused_gold, _, gold = built
        for name in config.LABELS:
            auc = calibrate._auc(gold[name].to_numpy().astype(int),
                                 fused_gold[name].to_numpy())
            assert auc >= calibrate.KILL_THRESHOLD, f"{name} at {auc:.3f}"

    def test_no_gold_derived_constants(self, built):
        # The guard against reintroducing the overfit: fusion must depend only
        # on module constants, not on anything measured from gold.
        source = (config.PROJECT_ROOT / "src" / "labels" / "fuse.py").read_text()
        assert "measure_priors" not in source
        assert "measure_weights" not in source
