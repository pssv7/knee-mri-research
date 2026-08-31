"""Tests for training augmentation.

Two properties carry the weight. **Series consistency** -- one parameter draw
per series, applied to every slice -- because per-slice parameters destroy the
through-plane structure `stack_2p5d` feeds the backbone. And **the laterality
flip swapping labels**, because a flip that moves pixels without moving
`Medial Meniscus` onto `Lateral Meniscus` silently corrupts four of the twelve
findings and raises nothing.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import augment  # noqa: E402
from src.model.dataset import META_COLUMNS  # noqa: E402

DEPTH, SIZE = 6, 8


def volume(fill=100):
    return np.full((DEPTH, SIZE, SIZE), fill, dtype=np.uint8)


def study(n_series=2, sagittal=True):
    meta = np.zeros((n_series, len(META_COLUMNS)), dtype=np.float32)
    meta[:, augment.SAGITTAL_COLUMN] = 1.0 if sagittal else 0.0
    return {
        "volumes": np.stack([np.arange(DEPTH * SIZE * SIZE, dtype=np.uint8)
                             .reshape(DEPTH, SIZE, SIZE)
                             for _ in range(n_series)]),
        "meta": meta,
        "targets": np.arange(len(config.LABELS), dtype=np.float32),
        "weights": np.arange(len(config.LABELS), dtype=np.float32) + 100,
    }


class TestDefaultsAreTheIncumbent:
    def test_reference_is_intensity_scale_only(self):
        cfg = augment.REFERENCE
        assert cfg.intensity_scale == (0.9, 1.1)
        assert cfg.intensity_shift == 0.0
        assert cfg.gamma is None
        assert cfg.noise_std == 0.0
        assert cfg.cutout_boxes == 0
        assert cfg.slice_reverse == 0.0
        assert cfg.laterality_flip == 0.0

    def test_the_reference_scale_matches_the_legacy_range(self):
        # The legacy path is np.random.uniform(0.9, 1.1) on the whole study.
        rng = np.random.default_rng(0)
        out = [augment.augment_series(volume(100), rng).astype(float).mean()
               for _ in range(200)]
        assert 89.0 <= min(out) and max(out) <= 111.0

    def test_an_impossible_probability_is_refused(self):
        with pytest.raises(ValueError, match="slice_reverse"):
            augment.AugmentConfig(slice_reverse=1.5)
        with pytest.raises(ValueError, match="laterality_flip"):
            augment.AugmentConfig(laterality_flip=-0.1)
        with pytest.raises(ValueError, match="cutout_fraction"):
            augment.AugmentConfig(cutout_fraction=0.0)


class TestSeriesConsistency:
    def test_one_intensity_scale_for_every_slice_of_a_series(self):
        cfg = augment.AugmentConfig(intensity_scale=(0.5, 2.0))
        out = augment.augment_series(volume(100), np.random.default_rng(1), cfg)
        # A flat volume stays flat: every slice took the same scale.
        assert len(np.unique(out)) == 1

    def test_the_cutout_box_is_in_the_same_place_on_every_slice(self):
        cfg = augment.AugmentConfig(intensity_scale=None, cutout_boxes=1,
                                    cutout_fraction=0.5)
        out = augment.augment_series(volume(200), np.random.default_rng(2), cfg)
        zeroed = out == 0
        # Identical mask on each slice, and it actually covers something.
        assert zeroed.any()
        for d in range(1, DEPTH):
            assert np.array_equal(zeroed[0], zeroed[d])

    def test_the_noise_field_is_shared_across_slices(self):
        cfg = augment.AugmentConfig(intensity_scale=None, noise_std=20.0)
        out = augment.augment_series(volume(128), np.random.default_rng(3), cfg)
        for d in range(1, DEPTH):
            assert np.array_equal(out[0], out[d])

    def test_two_series_get_independent_draws(self):
        cfg = augment.AugmentConfig(intensity_scale=(0.5, 2.0))
        rng = np.random.default_rng(4)
        item = study(n_series=2)
        item["volumes"] = np.stack([volume(100), volume(100)])
        out = augment.augment_study(item, rng, cfg)["volumes"]
        assert out[0][0, 0, 0] != out[1][0, 0, 0]


class TestTransformsDoWhatTheySay:
    def test_gamma_moves_midtones_and_keeps_the_ends(self):
        cfg = augment.AugmentConfig(intensity_scale=None, gamma=(2.0, 2.0))
        out = augment.augment_series(volume(128), np.random.default_rng(5), cfg)
        assert out[0, 0, 0] < 128
        for value in (0, 255):
            edge = augment.augment_series(volume(value),
                                          np.random.default_rng(5), cfg)
            assert abs(int(edge[0, 0, 0]) - value) <= 1

    def test_slice_reverse_reverses_the_stack(self):
        cfg = augment.AugmentConfig(intensity_scale=None, slice_reverse=1.0)
        source = np.stack([np.full((SIZE, SIZE), i, dtype=np.uint8)
                           for i in range(DEPTH)])
        out = augment.augment_series(source, np.random.default_rng(6), cfg)
        assert [int(s[0, 0]) for s in out] == list(range(DEPTH))[::-1]

    def test_nothing_leaves_the_uint8_range(self):
        cfg = augment.AugmentConfig(intensity_scale=(3.0, 3.0),
                                    intensity_shift=200.0, noise_std=90.0)
        out = augment.augment_series(volume(200), np.random.default_rng(7), cfg)
        assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


class TestLateralityFlip:
    def test_medial_and_lateral_labels_are_swapped(self):
        item = augment.flip_laterality(study(), np.random.default_rng(8))
        index = {n: i for i, n in enumerate(config.LABELS)}
        for medial, lateral in config.LATERAL_LABEL_PAIRS:
            assert item["targets"][index[medial]] == float(index[lateral])
            assert item["targets"][index[lateral]] == float(index[medial])

    def test_weights_travel_with_their_findings(self):
        item = augment.flip_laterality(study(), np.random.default_rng(9))
        index = {n: i for i, n in enumerate(config.LABELS)}
        for medial, lateral in config.LATERAL_LABEL_PAIRS:
            assert item["weights"][index[medial]] == 100 + index[lateral]

    def test_unpaired_findings_are_left_alone(self):
        item = augment.flip_laterality(study(), np.random.default_rng(10))
        index = {n: i for i, n in enumerate(config.LABELS)}
        paired = {n for pair in config.LATERAL_LABEL_PAIRS for n in pair}
        for name, i in index.items():
            if name not in paired:
                assert item["targets"][i] == float(i)

    def test_a_sagittal_series_flips_through_plane(self):
        before = study(n_series=1, sagittal=True)
        source = before["volumes"][0].copy()
        after = augment.flip_laterality(before, np.random.default_rng(11))
        assert np.array_equal(after["volumes"][0], source[::-1])

    def test_a_coronal_series_flips_in_plane(self):
        before = study(n_series=1, sagittal=False)
        source = before["volumes"][0].copy()
        after = augment.flip_laterality(before, np.random.default_rng(12))
        assert np.array_equal(after["volumes"][0], source[:, :, ::-1])

    def test_flipping_twice_is_the_identity(self):
        # The strongest statement that pixels and labels move together.
        original = study()
        volumes = original["volumes"].copy()
        targets = original["targets"].copy()
        once = augment.flip_laterality(original, np.random.default_rng(13))
        twice = augment.flip_laterality(once, np.random.default_rng(14))
        assert np.array_equal(twice["volumes"], volumes)
        assert np.array_equal(twice["targets"], targets)


class TestAugmentStudy:
    def test_the_flip_probability_is_honoured(self):
        cfg = augment.AugmentConfig(intensity_scale=None, laterality_flip=1.0)
        item = augment.augment_study(study(n_series=1, sagittal=True),
                                     np.random.default_rng(15), cfg)
        source = study(n_series=1, sagittal=True)["volumes"][0]
        assert np.array_equal(item["volumes"][0], source[::-1])

    def test_zero_probability_never_flips(self):
        cfg = augment.AugmentConfig(intensity_scale=None, laterality_flip=0.0)
        item = augment.augment_study(study(n_series=1),
                                     np.random.default_rng(16), cfg)
        assert np.array_equal(item["volumes"][0],
                              study(n_series=1)["volumes"][0])

    def test_a_study_with_no_series_does_not_raise(self):
        empty = {"volumes": np.zeros((0, DEPTH, SIZE, SIZE), dtype=np.uint8),
                 "meta": np.zeros((0, len(META_COLUMNS)), dtype=np.float32),
                 "targets": None, "weights": None}
        out = augment.augment_study(empty, np.random.default_rng(17),
                                    augment.AugmentConfig(laterality_flip=1.0))
        assert out["volumes"].shape[0] == 0
