"""Tests for volume construction, especially plane-aware mirroring.

Mirroring a knee reflects through the patient's left-right axis. Which array
axis that is depends on the acquisition plane, and getting it wrong is silent:
a column flip on a sagittal series leaves medial and lateral untouched and
reflects anterior-posterior instead.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.data import cache  # noqa: E402

AXIAL = [1, 0, 0, 0, 1, 0]        # rows along +x, cols along +y -> normal +z
CORONAL = [1, 0, 0, 0, 0, -1]     # left-right in-plane, normal +y
SAGITTAL = [0, 1, 0, 0, 0, -1]    # left-right is the slice axis


class TestMirrorAxis:
    def test_axial_and_coronal_flip_columns(self):
        assert cache.mirror_axis(AXIAL) == 2
        assert cache.mirror_axis(CORONAL) == 2

    def test_sagittal_reverses_slices(self):
        # Medial and lateral are separated across slices here, not across the
        # image, so a column flip would not swap them at all.
        assert cache.mirror_axis(SAGITTAL) == 0

    def test_oblique_sagittal_from_real_data(self):
        # The dev series: obliqued Siemens sagittal. Slice-axis alignment with
        # the left-right axis is 0.987 against 0.158 for the columns.
        oblique = [0.158049, 0.985205, 0.0662728,
                   0.016105, 0.0645355, -0.997785]
        assert cache.mirror_axis(oblique) == 0

    def test_missing_orientation_falls_back_to_columns(self):
        assert cache.mirror_axis(None) == 2
        assert cache.mirror_axis([1, 0, 0]) == 2


class TestCanonicalizeSide:
    def volume(self):
        # Distinct values per axis so a flip is detectable on each.
        return np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)

    def test_right_knee_is_untouched(self):
        v = self.volume()
        assert np.array_equal(cache.canonicalize_side(v, "R", CORONAL), v)

    def test_unknown_side_is_untouched(self):
        # Guessing would silently swap four of the twelve labels.
        v = self.volume()
        assert np.array_equal(cache.canonicalize_side(v, None, CORONAL), v)

    def test_left_coronal_flips_columns(self):
        v = self.volume()
        out = cache.canonicalize_side(v, "L", CORONAL)
        assert np.array_equal(out, v[:, :, ::-1])

    def test_left_sagittal_reverses_slice_order(self):
        v = self.volume()
        out = cache.canonicalize_side(v, "L", SAGITTAL)
        assert np.array_equal(out, v[::-1, :, :])
        # And crucially does NOT flip the in-plane axes, which would move the
        # patella from the front of the knee to the back.
        assert not np.array_equal(out, v[:, :, ::-1])

    def test_mirroring_is_its_own_inverse(self):
        v = self.volume()
        once = cache.canonicalize_side(v, "L", SAGITTAL)
        twice = cache.canonicalize_side(once, "L", SAGITTAL)
        assert np.array_equal(twice, v)

    def test_result_is_contiguous(self):
        # Reversed views are not contiguous, and a non-contiguous array can
        # fail or silently copy downstream in torch.
        out = cache.canonicalize_side(self.volume(), "L", SAGITTAL)
        assert out.flags["C_CONTIGUOUS"]
