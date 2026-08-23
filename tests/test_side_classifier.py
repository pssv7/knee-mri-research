"""Tests for left/right side prediction.

The load-bearing property is that slice ORDER survives into the model input.
For sagittal series the medial-lateral axis is the slice axis, so the side is
encoded in the direction of travel through the volume and nowhere else -- a
pipeline that sorts, shuffles, or collapses slices destroys the only signal
there is.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import side_classifier as sc  # noqa: E402


class TestPrepareBatch:
    def volume(self, depth=None):
        depth = depth or config.CACHE_SLICES
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (depth, 224, 224), dtype=np.uint8)

    def test_shape_and_range(self):
        out = sc.prepare_batch(self.volume()[None])
        assert out.shape == (1, config.CACHE_SLICES, sc.INPUT_SIZE, sc.INPUT_SIZE)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_single_volume_is_promoted_to_a_batch(self):
        assert sc.prepare_batch(self.volume()).shape[0] == 1

    def test_slice_order_is_preserved(self):
        # The whole reason sagittal sides are learnable. A volume and its
        # slice-reversed twin must not produce the same input.
        volume = self.volume()
        forward = sc.prepare_batch(volume[None])
        reversed_ = sc.prepare_batch(volume[::-1][None])
        assert not np.allclose(forward, reversed_)
        assert np.allclose(forward[0], reversed_[0][::-1], atol=1e-6)

    def test_channels_map_to_slices(self):
        # Channel j must come from slice j, not from a pooled or sorted view.
        volume = np.zeros((config.CACHE_SLICES, 224, 224), dtype=np.uint8)
        volume[3] = 255
        out = sc.prepare_batch(volume[None])
        assert out[0, 3].mean() > 0.9
        assert out[0, 4].mean() < 0.1


class TestFlipInput:
    def test_sagittal_flips_the_slice_dimension(self):
        batch = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
        out = sc.flip_input(batch, plane_axis=[0, 0])
        assert np.array_equal(out[0], batch[0][::-1])

    def test_coronal_flips_the_column_dimension(self):
        batch = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
        out = sc.flip_input(batch, plane_axis=[2, 2])
        assert np.array_equal(out[0], batch[0][:, :, ::-1])

    def test_items_flip_independently(self):
        batch = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
        out = sc.flip_input(batch, plane_axis=[0, 2])
        assert np.array_equal(out[0], batch[0][::-1])
        assert np.array_equal(out[1], batch[1][:, :, ::-1])

    def test_flipping_twice_restores(self):
        batch = np.random.default_rng(1).random((2, 4, 3, 3)).astype(np.float32)
        once = sc.flip_input(batch, [0, 2])
        twice = sc.flip_input(once, [0, 2])
        assert np.allclose(twice, batch)


class TestStudyVote:
    def test_majority_decides(self):
        result = sc.study_level_vote([0.9, 0.8, 0.2], ["s", "s", "s"])
        assert result["s"] == "R"

    def test_studies_are_independent(self):
        result = sc.study_level_vote([0.9, 0.1], ["a", "b"])
        assert result == {"a": "R", "b": "L"}

    def test_confidence_weights_the_vote(self):
        # Two unconfident R against one confident L.
        result = sc.study_level_vote([0.6, 0.6, 0.0], ["s"] * 3,
                                     confidences=[0.1, 0.1, 5.0])
        assert result["s"] == "L"

    def test_length_mismatch_does_not_silently_misalign(self):
        # zip truncates; a wrong-length confidence array must not shift
        # weights onto the wrong series.
        with pytest.raises(Exception):
            sc.study_level_vote(np.array([0.9, 0.1]), ["a", "b"],
                                confidences=np.array([1.0]))


class TestAccuracy:
    def test_perfect_and_inverted(self):
        truth = np.array([1, 1, 0, 0])
        assert sc.accuracy(np.array([0.9, 0.7, 0.1, 0.3]), truth) == 1.0
        assert sc.accuracy(np.array([0.1, 0.3, 0.9, 0.7]), truth) == 0.0
