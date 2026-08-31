"""Tests for the contrast and series-composition audit (Stage 0c, 0d).

The way this check fails is by reporting a clean bill it has not earned: a
description scan that finds no contrast when a fifth of the descriptions were
anonymised away, or a composition effect read off 58 studies where the
correlation's standard error is 0.13. Both are guarded here.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from tools import series_audit as audit  # noqa: E402


def index_frame(rows):
    """Series-level rows in the shape src/data/index.py writes."""
    return pd.DataFrame(rows)


def routing_frame(rows):
    return pd.DataFrame(rows)


ORDINARY = {config.ID_COL: "s1", "SeriesInstanceUID": "a",
            "SeriesDescription": "pd_tse_fs_sag", "ProtocolName": None,
            "EchoTime": 40.0, "RepetitionTime": 3000.0}
ENHANCED = {config.ID_COL: "s1", "SeriesInstanceUID": "b",
            "SeriesDescription": "RT_t1_tse_cor_FS+C", "ProtocolName": None,
            "EchoTime": 11.0, "RepetitionTime": 470.0}
ANONYMOUS = {config.ID_COL: "s2", "SeriesInstanceUID": "c",
             "SeriesDescription": "dummyseriesdesc!", "ProtocolName": None,
             "EchoTime": 40.0, "RepetitionTime": 3000.0}


class TestContrastScan:
    def _routing(self, enhanced_is_fluid=0):
        return routing_frame([
            {config.ID_COL: "s1", "SeriesInstanceUID": "a",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1,
             "Anatomical_Plane": "Sagittal"},
            {config.ID_COL: "s1", "SeriesInstanceUID": "b",
             "Fluid_Sensitive": enhanced_is_fluid, "Fat_Suppression": 0,
             "Anatomical_Plane": "Coronal"},
            {config.ID_COL: "s2", "SeriesInstanceUID": "c",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1,
             "Anatomical_Plane": "Axial"},
        ])

    def test_it_finds_a_post_contrast_series(self):
        scan = audit.contrast_scan(index_frame([ORDINARY, ENHANCED, ANONYMOUS]),
                                   self._routing())
        assert len(scan["flagged"]) == 1
        assert scan["studies"] == 1

    def test_an_enhanced_series_routed_as_fluid_sensitive_is_called_out(self):
        clean = audit.contrast_scan(index_frame([ORDINARY, ENHANCED]),
                                    self._routing(enhanced_is_fluid=0))
        broken = audit.contrast_scan(index_frame([ORDINARY, ENHANCED]),
                                     self._routing(enhanced_is_fluid=1))
        assert len(clean["misrouted"]) == 0
        assert len(broken["misrouted"]) == 1

    def test_the_blind_spot_is_reported_not_counted_as_a_negative(self):
        # Absence of a string is not absence of contrast, so a scan that finds
        # nothing must still say how much it could not see.
        scan = audit.contrast_scan(index_frame([ORDINARY, ANONYMOUS]),
                                   self._routing())
        assert len(scan["flagged"]) == 0
        assert scan["unusable_series"] == 1
        assert scan["studies_all_unusable"] == 1

    def test_ordinary_sequence_names_are_not_false_positives(self):
        # "pd_tse_fs_tra_320", "t2_de3d_we_tra_patella_fit_t" and friends are
        # the bulk of this collection; a scan that fires on them is useless.
        rows = [dict(ORDINARY, SeriesInstanceUID=f"x{i}",
                     SeriesDescription=name)
                for i, name in enumerate(
                    ["pd_tse_fs_tra_320", "t2_de3d_we_tra_patella_fit_t",
                     "cor t1 fse", "pdw_tse_spair_sag", "t2_me2d_sag",
                     "post_op_free"])]
        routing = routing_frame([
            {config.ID_COL: "s1", "SeriesInstanceUID": f"x{i}",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1,
             "Anatomical_Plane": "Sagittal"} for i in range(len(rows))])
        scan = audit.contrast_scan(index_frame(rows), routing)
        assert len(scan["flagged"]) == 0


class TestRepeatControl:
    def test_a_repeated_t1_acquisition_in_one_plane_is_counted(self):
        rows = [dict(ENHANCED, SeriesInstanceUID="b1"),
                dict(ENHANCED, SeriesInstanceUID="b2")]
        routing = routing_frame([
            {config.ID_COL: "s1", "SeriesInstanceUID": uid,
             "Fluid_Sensitive": 0, "Fat_Suppression": 1,
             "Anatomical_Plane": "Coronal"} for uid in ("b1", "b2")])
        assert audit.repeat_control(index_frame(rows), routing)["studies"] == 1

    def test_a_long_te_sequence_is_not_t1_like(self):
        rows = [dict(ORDINARY, SeriesInstanceUID="a1"),
                dict(ORDINARY, SeriesInstanceUID="a2")]
        routing = routing_frame([
            {config.ID_COL: "s1", "SeriesInstanceUID": uid,
             "Fluid_Sensitive": 1, "Fat_Suppression": 1,
             "Anatomical_Plane": "Sagittal"} for uid in ("a1", "a2")])
        assert audit.repeat_control(index_frame(rows), routing)["studies"] == 0


class TestCompositionEffect:
    @staticmethod
    def _planted(n=2000, strength=0.0, seed=0):
        rng = np.random.default_rng(seed)
        counts = rng.integers(2, 6, size=n)
        effusion = 1 / (1 + np.exp(-rng.normal(0, 1, size=n)))
        label = (rng.random(n) < 0.2).astype(float)
        score = 1 / (1 + np.exp(-(rng.normal(0, 1, size=n)
                                  + strength * counts)))
        return counts, score, effusion, label

    def test_it_finds_a_planted_protocol_effect(self):
        counts, score, effusion, label = self._planted(strength=0.6)
        effect = audit.composition_effect(counts, score, effusion, label)
        assert effect["r_score"] > 0.3
        assert effect["r_score_partial"] > 0.3

    def test_an_unrelated_score_reads_flat(self):
        counts, score, effusion, label = self._planted(strength=0.0)
        effect = audit.composition_effect(counts, score, effusion, label)
        assert abs(effect["r_score"]) < 0.1

    def test_the_label_s_own_correlation_is_reported_for_comparison(self):
        # Sicker knees are scanned with more sequences, so the raw correlation
        # alone cannot say whether the model is reading protocol.
        counts, score, effusion, _ = self._planted(strength=0.6)
        label = (counts >= 4).astype(float)
        effect = audit.composition_effect(counts, score, effusion, label)
        assert effect["r_label"] > 0.5


class TestComposition:
    def test_it_counts_the_series_the_aggregator_sees(self):
        routing = routing_frame([
            {config.ID_COL: "s1", "SeriesInstanceUID": "a",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1},
            {config.ID_COL: "s1", "SeriesInstanceUID": "b",
             "Fluid_Sensitive": 1, "Fat_Suppression": 0},
            {config.ID_COL: "s1", "SeriesInstanceUID": "c",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        ])
        table = audit.composition(routing)
        assert table.loc["s1", "n_series"] == 3
        assert table.loc["s1", "n_fluid"] == 3
        assert table.loc["s1", "n_fluid_fatsat"] == 2
