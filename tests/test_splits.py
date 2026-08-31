"""Tests for CV splits and the gold holdout."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import splits  # noqa: E402


def fake_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = {config.ID_COL: [f"study{i}" for i in range(n)]}
    for name in config.LABELS:
        frame[name] = rng.random(n)
    return pd.DataFrame(frame)


class TestGoldHoldout:
    def test_leakage_raises(self):
        with pytest.raises(ValueError, match="leaked into training"):
            splits.assert_no_gold_leakage(["a", "b", "c"], {"b"})

    def test_clean_split_passes(self):
        splits.assert_no_gold_leakage(["a", "b"], {"x", "y"})

    def test_empty_gold_set_passes(self):
        splits.assert_no_gold_leakage(["a"], set())


class TestFolds:
    def test_every_study_gets_exactly_one_fold(self):
        folded = splits.make_folds(fake_labels())
        assert (folded["fold"] >= 0).all()
        assert len(folded) == 200

    def test_folds_are_disjoint_and_complete(self):
        folded = splits.make_folds(fake_labels())
        seen = set()
        for train, validation in splits.iter_folds(folded):
            ids = set(validation[config.ID_COL])
            assert not (ids & seen), "a study appears in two validation folds"
            seen |= ids
            assert not (set(train[config.ID_COL]) & ids)
        assert len(seen) == 200

    def test_folds_are_balanced_in_size(self):
        counts = splits.make_folds(fake_labels())["fold"].value_counts()
        assert counts.max() - counts.min() <= splits.N_FOLDS

    def test_deterministic(self):
        a = splits.make_folds(fake_labels(), seed=7)["fold"].tolist()
        b = splits.make_folds(fake_labels(), seed=7)["fold"].tolist()
        assert a == b

    def test_seed_changes_assignment(self):
        a = splits.make_folds(fake_labels(), seed=1)["fold"].tolist()
        b = splits.make_folds(fake_labels(), seed=2)["fold"].tolist()
        assert a != b


class TestStratification:
    def test_key_counts_positive_findings(self):
        frame = pd.DataFrame({name: [0.9, 0.1] for name in config.LABELS})
        key = splits.stratification_key(frame)
        assert key.iloc[0] == 4  # clipped
        assert key.iloc[1] == 0

    def test_rare_heavy_studies_spread_across_folds(self):
        # Studies with many findings are rare; clustering them in one fold
        # would make that fold's metrics incomparable to the others.
        rng = np.random.default_rng(0)
        n = 300
        frame = {config.ID_COL: [f"s{i}" for i in range(n)]}
        for name in config.LABELS:
            frame[name] = np.where(np.arange(n) < 20, 0.9, rng.random(n) * 0.4)
        folded = splits.make_folds(pd.DataFrame(frame))
        heavy = folded[folded[config.LABELS[0]] > 0.5]
        assert heavy["fold"].nunique() >= 4


class TestSeriesLookup:
    def test_selects_only_requested_studies(self):
        manifest = pd.DataFrame({
            config.ID_COL: ["a", "a", "b", "c"],
            "SeriesInstanceUID": ["s1", "s2", "s3", "s4"],
        })
        out = splits.series_for_studies(manifest, ["a", "c"])
        assert set(out["SeriesInstanceUID"]) == {"s1", "s2", "s4"}

    def test_keeps_all_series_of_a_study(self):
        # Splitting by series instead of study would put near-duplicate views
        # of one knee on both sides of the split.
        manifest = pd.DataFrame({
            config.ID_COL: ["a"] * 5,
            "SeriesInstanceUID": [f"s{i}" for i in range(5)],
        })
        assert len(splits.series_for_studies(manifest, ["a"])) == 5


class TestGroupedFolds:
    """Leave-scanner-out rather than leave-study-out.

    Random K-fold over a multi-site dataset lets a CNN memorise scanner
    signature and score it as pathology -- the plan puts that at ~0.053 of
    inflation, with one measured model showing a 0.136 grouped-vs-random gap.
    Nothing about it raises; the validation number is simply not the number.
    """

    def _labels(self, n=60):
        frame = pd.DataFrame({config.ID_COL: [f"s{i}" for i in range(n)]})
        for name in config.LABELS:
            frame[name] = (np.arange(n) % 3 == 0).astype(float)
        return frame

    def test_a_group_is_never_split_across_folds(self):
        labels = self._labels()
        groups = pd.Series({f"s{i}": f"scanner{i % 7}" for i in range(60)})
        folded = splits.make_folds(labels, groups=groups)
        per_group = folded.assign(group=folded[config.ID_COL].map(groups)) \
                          .groupby("group")["fold"].nunique()
        assert (per_group == 1).all()

    def test_every_study_still_gets_a_fold(self):
        labels = self._labels()
        groups = pd.Series({f"s{i}": f"scanner{i % 7}" for i in range(60)})
        folded = splits.make_folds(labels, groups=groups)
        assert set(folded["fold"]) <= set(range(splits.N_FOLDS))
        assert (folded["fold"] >= 0).all()

    def test_unequal_groups_are_balanced_greedily(self):
        # One scanner holding half the corpus cannot be split, so the folds
        # cannot come out even -- but nothing else should be piled on top of
        # it. The largest fold must be the one holding that group alone.
        labels = self._labels(40)
        groups = pd.Series({f"s{i}": ("big" if i < 20 else f"small{i}")
                            for i in range(40)})
        folded = splits.make_folds(labels, groups=groups)
        sizes = folded["fold"].value_counts()
        assert sizes.max() == 20

    def test_a_study_missing_from_the_group_map_is_not_dropped(self):
        labels = self._labels(10)
        groups = pd.Series({f"s{i}": "scanner0" for i in range(5)})
        folded = splits.make_folds(labels, groups=groups)
        assert (folded["fold"] >= 0).all()
        assert len(folded) == 10

    def test_the_ungrouped_path_is_unchanged(self):
        # Every reference number on record was produced by it.
        labels = self._labels()
        assert splits.make_folds(labels)["fold"].tolist() == \
            splits.make_folds(labels, groups=None)["fold"].tolist()


class TestScannerKey:
    def test_it_combines_the_three_header_fields(self):
        index = pd.DataFrame({
            config.ID_COL: ["a", "a", "b"],
            "Manufacturer": ["SIEMENS", "SIEMENS", "GE"],
            "ManufacturerModelName": ["Aera", "Aera", "Signa"],
            "MagneticFieldStrength": [1.5, 1.5, 3.0],
        })
        keys = splits.scanner_key(index)
        assert keys["a"] == "SIEMENS|Aera|1.5"
        assert keys["a"] != keys["b"]

    def test_a_study_with_no_scanner_headers_is_grouped_as_unknown(self):
        index = pd.DataFrame({
            config.ID_COL: ["a"],
            "Manufacturer": [None],
            "ManufacturerModelName": [None],
            "MagneticFieldStrength": [None],
        })
        assert splits.scanner_key(index)["a"] == "unknown"

    def test_it_works_from_whichever_fields_are_present(self):
        index = pd.DataFrame({config.ID_COL: ["a"], "Manufacturer": ["GE"]})
        assert splits.scanner_key(index)["a"] == "GE"

    def test_an_index_without_any_of_them_says_so(self):
        index = pd.DataFrame({config.ID_COL: ["a"], "Rows": [256]})
        with pytest.raises(ValueError, match="Manufacturer"):
            splits.scanner_key(index)
