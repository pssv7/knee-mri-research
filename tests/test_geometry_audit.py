"""The slice-spacing audit, on series whose geometry is known by construction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audit_geometry import audit, series_geometry, summarise

SAGITTAL = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]      # normal is +z


def series(uid, positions, spacing=None, thickness=3.0, pixel=(0.5, 0.5),
           orientation=SAGITTAL):
    return pd.DataFrame({
        "SeriesInstanceUID": uid, "StudyInstanceUID": f"study-{uid}",
        "ImagePositionPatient": [[0.0, 0.0, float(z)] for z in positions],
        "ImageOrientationPatient": [list(orientation)] * len(positions),
        "SliceThickness": thickness, "SpacingBetweenSlices": spacing,
        "PixelSpacing": [list(pixel)] * len(positions),
        "InstanceNumber": range(1, len(positions) + 1)})


def test_even_spacing_measures_the_gap_and_the_span_the_kernel_covers():
    frame = series("even", np.arange(24) * 4.0, spacing=4.0)
    row = series_geometry(frame, cached_slices=24, kernel=3)
    assert row["n_slices"] == 24 and row["has_geometry"]
    assert row["acquired_gap_mm"] == pytest.approx(4.0)
    assert row["gap_cv"] == pytest.approx(0.0, abs=1e-9)
    assert row["declared_error_mm"] == pytest.approx(0.0, abs=1e-9)
    # 24 acquired into 24 cached: one cached step is one acquired slice.
    assert row["sampled_gap_mm"] == pytest.approx(4.0)
    assert row["kernel_span_mm"] == pytest.approx(8.0)
    assert row["in_plane_mm"] == pytest.approx(0.5)


def test_a_long_series_is_subsampled_and_a_short_one_repeats():
    long_row = series_geometry(series("long", np.arange(47) * 2.0),
                               cached_slices=24, kernel=3)
    assert long_row["sampled_gap_mm"] == pytest.approx(4.0)     # every other slice
    short_row = series_geometry(series("short", np.arange(12) * 4.0),
                                cached_slices=24, kernel=3)
    # Fewer slices than the cache keeps: cached neighbours are partly repeats,
    # so one cached step spans less than one acquired gap.
    assert short_row["sampled_gap_mm"] < short_row["acquired_gap_mm"]
    assert short_row["sampled_gap_mm"] == pytest.approx(4.0 * 11 / 23)


def test_uneven_spacing_is_reported_rather_than_averaged_away():
    frame = series("uneven", [0.0, 3.0, 6.0, 12.0, 15.0], spacing=3.0)
    row = series_geometry(frame, cached_slices=24, kernel=3)
    assert row["gap_min_mm"] == pytest.approx(3.0)
    assert row["gap_max_mm"] == pytest.approx(6.0)
    assert row["gap_cv"] > 0.01
    assert row["extent_mm"] == pytest.approx(15.0)


def test_a_declared_spacing_that_contradicts_the_images_is_flagged():
    row = series_geometry(series("wrong", np.arange(10) * 5.0, spacing=1.0),
                          cached_slices=24, kernel=3)
    assert row["declared_gap_mm"] == pytest.approx(1.0)
    assert row["declared_error_mm"] == pytest.approx(4.0)


def test_thickness_stands_in_when_spacing_is_absent():
    row = series_geometry(series("thick", np.arange(10) * 3.0, spacing=None,
                                 thickness=3.0), cached_slices=24, kernel=3)
    assert row["declared_gap_mm"] == pytest.approx(3.0)


def test_series_without_geometry_report_no_spacing_instead_of_a_guess():
    frame = series("blind", np.arange(10) * 3.0)
    frame["ImagePositionPatient"] = [None] * len(frame)
    row = series_geometry(frame, cached_slices=24, kernel=3)
    assert np.isnan(row["acquired_gap_mm"]) and not row["has_geometry"]
    assert "kernel_span_mm" not in row


def test_the_audit_summarises_a_mixed_corpus(tmp_path):
    frames = [series("even", np.arange(24) * 4.0, spacing=4.0),
              series("uneven", [0.0, 3.0, 6.0, 12.0, 15.0], spacing=3.0),
              series("short", np.arange(12) * 4.0, spacing=4.0)]
    index = tmp_path / "index.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(index, index=False)

    table = audit(index, cached_slices=24, kernel=3, tolerance=0.01)
    summary = summarise(table, cached_slices=24, kernel=3, tolerance=0.01)
    assert summary["series"] == 3
    assert summary["irregular_series"] == 1
    assert summary["series_shorter_than_cache"] == 2       # uneven and short
    assert summary["kernel_span_mm"]["median"] > 0
    assert "sampled slice index" in summary["limitation"]


def test_the_real_corpus_is_evenly_spaced_within_series_and_not_between_them():
    """Guards the claim the spatial experiment's limitation note rests on."""
    path = Path(__file__).resolve().parents[1] / "outputs/geometry_audit/per_series.parquet"
    if not path.exists():
        pytest.skip("run tools/audit_geometry.py first")
    table = pd.read_parquet(path)
    assert len(table) == 24371
    assert table["gap_cv"].max() < 0.01
    spans = table["kernel_span_mm"].dropna()
    assert np.quantile(spans, .95) / np.quantile(spans, .05) > 1.5
