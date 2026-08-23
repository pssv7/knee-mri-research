"""Tests for slice ordering and laterality resolution.

The synthetic tests run anywhere. The tests marked with `requires_dev_series`
need a single real series downloaded to data/dev/ -- see docs/dev_data.md.
"""

import glob
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import dicom_io as dio  # noqa: E402

DEV_SERIES = sorted(glob.glob(os.path.join("data", "dev", "*.dcm")))
requires_dev_series = pytest.mark.skipif(
    len(DEV_SERIES) == 0, reason="no dev series downloaded"
)

# Axial acquisition: rows along +x, columns along +y, so the normal is +z.
AXIAL_ORIENTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def make_header(position, orientation=None, instance=None, **extra):
    header = {
        "ImagePositionPatient": position,
        "ImageOrientationPatient": orientation or AXIAL_ORIENTATION,
        "InstanceNumber": instance,
    }
    header.update(extra)
    return header


class TestSliceNormal:
    def test_axial_normal_is_z(self):
        normal = dio.slice_normal(AXIAL_ORIENTATION)
        assert np.allclose(normal, [0.0, 0.0, 1.0])

    def test_normal_is_unit_length(self):
        # Realistic obliqued sagittal cosines from a Siemens knee series.
        orientation = [0.158049, 0.985205, 0.0662728,
                       0.016105, 0.0645355, -0.997785]
        assert math.isclose(np.linalg.norm(dio.slice_normal(orientation)), 1.0,
                            rel_tol=1e-6)

    def test_missing_orientation_falls_back(self):
        assert np.allclose(dio.slice_normal(None), [0.0, 0.0, 1.0])
        assert np.allclose(dio.slice_normal([1.0, 0.0]), [0.0, 0.0, 1.0])

    def test_degenerate_orientation_falls_back(self):
        # Parallel cosines cross to a zero vector.
        assert np.allclose(dio.slice_normal([1, 0, 0, 1, 0, 0]), [0.0, 0.0, 1.0])


class TestSortSlices:
    def test_orders_by_position_not_instance_number(self):
        # InstanceNumber runs opposite to geometry, as seen in real Siemens
        # series. Geometry must win.
        headers = [
            make_header([0.0, 0.0, 10.0], instance=1),
            make_header([0.0, 0.0, 0.0], instance=2),
            make_header([0.0, 0.0, 20.0], instance=3),
        ]
        ordered = dio.sort_slices(headers)
        assert [h["ImagePositionPatient"][2] for h in ordered] == [0.0, 10.0, 20.0]

    def test_position_projects_onto_normal(self):
        # Sagittal stack whose cosines cross to +x, so only the x coordinate
        # orders the slices and the large y/z values must be ignored.
        sagittal = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        assert np.allclose(dio.slice_normal(sagittal), [1.0, 0.0, 0.0])
        headers = [
            make_header([5.0, 99.0, 99.0], orientation=sagittal),
            make_header([1.0, 0.0, 0.0], orientation=sagittal),
        ]
        ordered = dio.sort_slices(headers)
        assert ordered[0]["ImagePositionPatient"][0] == 1.0

    def test_ordering_follows_normal_direction(self):
        # The same positions under a normal pointing the other way must
        # reverse. This is geometry, not a convention we get to choose.
        flipped = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        assert np.allclose(dio.slice_normal(flipped), [-1.0, 0.0, 0.0])
        headers = [
            make_header([5.0, 0.0, 0.0], orientation=flipped),
            make_header([1.0, 0.0, 0.0], orientation=flipped),
        ]
        ordered = dio.sort_slices(headers)
        assert ordered[0]["ImagePositionPatient"][0] == 5.0

    def test_falls_back_to_instance_number_without_geometry(self):
        headers = [
            {"ImagePositionPatient": None, "InstanceNumber": 2},
            {"ImagePositionPatient": None, "InstanceNumber": 1},
        ]
        ordered = dio.sort_slices(headers)
        assert [h["InstanceNumber"] for h in ordered] == [1, 2]


class TestHasGeometry:
    def test_true_when_complete(self):
        assert dio.has_geometry([make_header([0.0, 0.0, 0.0])])

    def test_false_when_position_missing(self):
        assert not dio.has_geometry([make_header(None)])

    def test_false_when_any_slice_incomplete(self):
        headers = [make_header([0.0, 0.0, 0.0]), make_header(None)]
        assert not dio.has_geometry(headers)


class TestResolveLaterality:
    def test_prefers_image_laterality_tag(self):
        assert dio.resolve_laterality([{"ImageLaterality": "R"}]) == "R"

    def test_falls_back_to_laterality_tag(self):
        # The real series tested had ImageLaterality empty and Laterality set.
        headers = [{"ImageLaterality": None, "Laterality": "L"}]
        assert dio.resolve_laterality(headers) == "L"

    def test_parses_description_in_several_languages(self):
        cases = [
            ("MR Knie Rechts 15ch AA", "R"),
            ("Rodilla izquierda", "L"),
            ("KNEE LEFT SAG T2", "L"),
            ("genou droite", "R"),
        ]
        for text, expected in cases:
            headers = [{"SeriesDescription": text}]
            assert dio.resolve_laterality(headers) == expected, text

    def test_returns_none_when_ambiguous(self):
        # Mentioning both sides tells us nothing; guessing would swap the
        # medial and lateral labels.
        headers = [{"SeriesDescription": "compare left and right knee"}]
        assert dio.resolve_laterality(headers) is None

    def test_returns_none_when_absent(self):
        assert dio.resolve_laterality([{"SeriesDescription": "t2_tse_sag_d"}]) is None
        assert dio.resolve_laterality([{}]) is None

    def test_does_not_match_substrings(self):
        # "Sagittal" contains no standalone side token; a naive substring
        # search for "l" or "r" would fire here.
        headers = [{"SeriesDescription": "Sagittal PD FSE"}]
        assert dio.resolve_laterality(headers) is None


@requires_dev_series
class TestRealSeries:
    @pytest.fixture(scope="class")
    @classmethod
    def headers(cls):
        return [dio.read_header(p) for p in DEV_SERIES]

    def test_headers_parse(self, headers):
        assert len(headers) == len(DEV_SERIES)
        assert all(h["SeriesInstanceUID"] for h in headers)

    def test_numeric_fields_are_numeric(self, headers):
        for field in ("Rows", "Columns", "EchoTime"):
            value = headers[0][field]
            assert isinstance(value, float), f"{field} is {type(value)}"

    def test_geometry_is_complete(self, headers):
        assert dio.has_geometry(headers)

    def test_sorted_positions_are_monotonic(self, headers):
        positions = [dio.slice_position(h) for h in dio.sort_slices(headers)]
        assert all(b >= a for a, b in zip(positions, positions[1:]))

    def test_slice_spacing_is_regular(self, headers):
        positions = [dio.slice_position(h) for h in dio.sort_slices(headers)]
        gaps = np.diff(positions)
        # A scrambled order would produce wildly irregular gaps.
        assert gaps.std() < 0.1 * abs(gaps.mean())

    def test_laterality_resolves(self, headers):
        assert dio.resolve_laterality(headers) in ("L", "R")

    def test_volume_loads_with_expected_shape(self, headers):
        ordered = dio.sort_slices(headers)
        volume = dio.load_series_volume([h["path"] for h in ordered])
        assert volume.shape[0] == len(ordered)
        assert volume.ndim == 3
        assert volume.std() > 0


class TestStudyLateralityPropagation:
    def test_fills_unknown_series_from_a_tagged_sibling(self):
        # One study is one knee, so a side found anywhere settles all of it.
        sides = {"study": {"a": "R", "b": None, "c": None}}
        filled = dio.resolve_study_laterality(sides)
        assert filled["study"] == {"a": "R", "b": "R", "c": "R"}

    def test_leaves_a_study_with_no_evidence_alone(self):
        sides = {"study": {"a": None, "b": None}}
        assert dio.resolve_study_laterality(sides)["study"] == {"a": None, "b": None}

    def test_conflicting_sides_are_not_resolved(self):
        # Contradiction means the one-knee-per-study assumption fails here;
        # guessing would swap medial and lateral on four of twelve labels.
        sides = {"study": {"a": "L", "b": "R", "c": None}}
        filled = dio.resolve_study_laterality(sides)
        assert filled["study"]["c"] is None
        assert filled["study"]["a"] == "L"

    def test_studies_are_independent(self):
        sides = {"one": {"a": "L", "b": None}, "two": {"c": "R", "d": None}}
        filled = dio.resolve_study_laterality(sides)
        assert filled["one"]["b"] == "L"
        assert filled["two"]["d"] == "R"

    def test_coverage_counts(self):
        sides = {
            "a": {"a1": "L", "a2": None, "a3": None},
            "b": {"b1": None, "b2": None},
            "c": {"c1": "L", "c2": "R"},
        }
        stats = dio.laterality_coverage(sides)
        assert stats["total_series"] == 7
        assert stats["known_before"] == 3
        assert stats["known_after"] == 5
        assert stats["recovered"] == 2
        assert stats["conflicting_studies"] == 1
