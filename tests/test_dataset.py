"""Tests for the study-level dataset.

Two behaviours carry the most risk of silent breakage:

- Mirroring must follow the acquisition plane. A column flip on a sagittal
  series does not swap medial with lateral; it reflects anterior-posterior.
- Padding must be masked. Series counts vary from one to a dozen, and an
  unmasked pad row is averaged in by the pooling as though it were a real
  series, quietly diluting every short study.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import dataset as ds  # noqa: E402

DEPTH, SIZE = config.CACHE_SLICES, config.CACHE_SIZE


def make_manifest(n_series=3, study="study1"):
    return pd.DataFrame({
        config.ID_COL: [study] * n_series,
        "SeriesInstanceUID": [f"series{i}" for i in range(n_series)],
        "position": list(range(n_series)),
        "cache_path": ["shard0"] * n_series,
        "laterality_resolved": ["R"] * n_series,
        "error": [None] * n_series,
    })


def make_series_meta(n_series=3, plane="Sagittal"):
    return pd.DataFrame({
        "SeriesInstanceUID": [f"series{i}" for i in range(n_series)],
        "Fluid_Sensitive": [i % 2 for i in range(n_series)],
        "Fat_Suppression": [(i + 1) % 2 for i in range(n_series)],
        "Anatomical_Plane": [plane] * n_series,
    })


def make_labels(study="study1"):
    row = {config.ID_COL: [study]}
    for i, name in enumerate(config.LABELS):
        row[name] = [i / 12.0]
        row[f"{name}_weight"] = [1.0 - i / 24.0]
    return pd.DataFrame(row)


class TestMirroring:
    def volume(self):
        return np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)

    def test_right_knee_untouched(self):
        v = self.volume()
        assert np.array_equal(ds.mirror_volume(v, "R", "Sagittal"), v)

    def test_unknown_side_untouched(self):
        v = self.volume()
        assert np.array_equal(ds.mirror_volume(v, None, "Coronal"), v)
        assert np.array_equal(ds.mirror_volume(v, np.nan, "Coronal"), v)

    def test_sagittal_reverses_slices(self):
        v = self.volume()
        assert np.array_equal(ds.mirror_volume(v, "L", "Sagittal"), v[::-1])

    def test_coronal_and_axial_flip_columns(self):
        v = self.volume()
        assert np.array_equal(ds.mirror_volume(v, "L", "Coronal"), v[:, :, ::-1])
        assert np.array_equal(ds.mirror_volume(v, "L", "Axial"), v[:, :, ::-1])

    def test_unknown_plane_falls_back_to_columns(self):
        v = self.volume()
        assert np.array_equal(ds.mirror_volume(v, "L", None), v[:, :, ::-1])

    def test_output_is_contiguous(self):
        out = ds.mirror_volume(self.volume(), "L", "Sagittal")
        assert out.flags["C_CONTIGUOUS"]


class TestSeriesMetadata:
    def test_plane_is_one_hot(self):
        row = pd.Series({"Fluid_Sensitive": 1, "Fat_Suppression": 0,
                         "Anatomical_Plane": "Coronal"})
        meta = ds.series_metadata(row)
        assert meta.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def test_fluid_and_fat_stay_separate(self):
        # The data description states these are correlated but not equivalent,
        # so collapsing them would lose the disagreeing cases.
        row = pd.Series({"Fluid_Sensitive": 1, "Fat_Suppression": 0,
                         "Anatomical_Plane": "Axial"})
        meta = ds.series_metadata(row)
        assert meta[0] != meta[1]

    def test_missing_values_default_to_zero(self):
        meta = ds.series_metadata(pd.Series({}))
        assert meta.tolist() == [0.0] * len(ds.META_COLUMNS)


class TestStudyIndex:
    def test_drops_failed_series(self):
        manifest = make_manifest()
        manifest.loc[1, "error"] = "truncated pixel data"
        index = ds.StudyIndex(manifest, make_series_meta(), make_labels())
        assert len(index.series_for("study1")) == 2

    def test_only_labelled_studies_are_indexed(self):
        manifest = pd.concat([make_manifest(2, "study1"),
                              make_manifest(2, "study2")], ignore_index=True)
        manifest["SeriesInstanceUID"] = [f"s{i}" for i in range(4)]
        meta = pd.DataFrame({"SeriesInstanceUID": [f"s{i}" for i in range(4)],
                             "Fluid_Sensitive": 1, "Fat_Suppression": 0,
                             "Anatomical_Plane": "Axial"})
        index = ds.StudyIndex(manifest, meta, make_labels("study1"))
        assert index.studies == ["study1"]

    def test_series_cap_is_deterministic(self):
        index = ds.StudyIndex(make_manifest(10), make_series_meta(10),
                              make_labels(), max_series=4)
        first = index.series_for("study1")["SeriesInstanceUID"].tolist()
        second = index.series_for("study1")["SeriesInstanceUID"].tolist()
        assert first == second
        assert len(first) <= 4

    def test_cap_does_nothing_below_the_limit(self):
        index = ds.StudyIndex(make_manifest(3), make_series_meta(3),
                              make_labels(), max_series=8)
        assert len(index.series_for("study1")) == 3

    def test_targets_and_weights_follow_label_order(self):
        index = ds.StudyIndex(make_manifest(), make_series_meta(), make_labels())
        targets, weights = index.targets_for("study1")
        assert len(targets) == 12 and len(weights) == 12
        assert targets[0] == pytest.approx(0.0)
        assert targets[11] == pytest.approx(11 / 12)

    def test_unlabelled_study_returns_none(self):
        index = ds.StudyIndex(make_manifest(), make_series_meta())
        assert index.targets_for("study1") == (None, None)


class TestLoadAndCollate:
    def shards(self, n=3):
        rng = np.random.default_rng(0)
        return {"shard0": rng.integers(0, 255, (n, DEPTH, SIZE, SIZE),
                                       dtype=np.uint8)}

    def test_load_study_shapes(self):
        index = ds.StudyIndex(make_manifest(), make_series_meta(), make_labels())
        item = ds.load_study(index, "study1", self.shards())
        assert item["volumes"].shape == (3, DEPTH, SIZE, SIZE)
        assert item["meta"].shape == (3, len(ds.META_COLUMNS))
        assert item["targets"].shape == (12,)

    def test_collate_pads_to_longest(self):
        index3 = ds.StudyIndex(make_manifest(3), make_series_meta(3), make_labels())
        index1 = ds.StudyIndex(make_manifest(1, "study2"),
                               make_series_meta(1), make_labels("study2"))
        batch = [ds.load_study(index3, "study1", self.shards()),
                 ds.load_study(index1, "study2", self.shards(1))]
        out = ds.collate_studies(batch)
        assert out["volumes"].shape[:2] == (2, 3)
        assert out["mask"].tolist() == [[True, True, True], [True, False, False]]

    def test_padded_rows_are_masked_false(self):
        # An unmasked pad row is pooled in as a real series and dilutes every
        # short study.
        index1 = ds.StudyIndex(make_manifest(1, "s"), make_series_meta(1),
                               make_labels("s"))
        index2 = ds.StudyIndex(make_manifest(2, "t"), make_series_meta(2),
                               make_labels("t"))
        out = ds.collate_studies([ds.load_study(index1, "s", self.shards(1)),
                                  ds.load_study(index2, "t", self.shards(2))])
        assert out["mask"][0].sum() == 1
        assert out["volumes"][0, 1].sum() == 0

    def test_collate_without_targets(self):
        index = ds.StudyIndex(make_manifest(), make_series_meta())
        out = ds.collate_studies([ds.load_study(index, "study1", self.shards())])
        assert "targets" not in out


class TestSideAsCovariate:
    """Laterality is handed to the model rather than normalised away.

    Precomputed frozen embeddings cannot be mirrored -- the features are
    already computed -- so the side travels as a covariate instead. The
    distinction between "left knee" and "side unknown" has to survive, since
    they call for different behaviour: one is information, the other is its
    absence.
    """

    def meta(self, side, plane="Coronal"):
        return ds.series_metadata(pd.Series({
            "Fluid_Sensitive": 0, "Fat_Suppression": 0,
            "Anatomical_Plane": plane, "side_predicted": side}))

    def test_right_and_left_differ(self):
        assert not np.array_equal(self.meta("R"), self.meta("L"))

    def test_unknown_is_distinct_from_left(self):
        # Both have is_right = 0; only side_known separates them.
        left, unknown = self.meta("L"), self.meta(None)
        assert left[-1] == 1.0 and unknown[-1] == 0.0

    def test_right_flag_set_only_for_right(self):
        assert self.meta("R")[-2] == 1.0
        assert self.meta("L")[-2] == 0.0
        assert self.meta(None)[-2] == 0.0

    def test_nan_side_counts_as_unknown(self):
        assert self.meta(np.nan)[-1] == 0.0

    def test_width_matches_the_column_list(self):
        assert len(self.meta("R")) == len(ds.META_COLUMNS)
