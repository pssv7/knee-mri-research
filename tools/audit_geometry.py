"""Measure the physical slice spacing the 3D blocks are about to convolve over.

`SpatialVolumeEncoder` runs a kernel of three along the depth axis of a cached
volume. That axis is *sampled slice index*, not millimetres: the cache keeps a
fixed number of positions per series, so the same kernel spans a different
physical distance in every series, and within a series it is only uniform if
the acquisition was. The plan requires that limitation be documented rather
than assumed away, and this measures its size from the header index that
already exists locally -- no DICOM pixels are read.

Reported per series, from the slice positions projected onto the plane normal
exactly as `dicom_io.sort_slices` orders them:

    acquired_gap_mm       median distance between neighbouring acquired slices
    gap_cv                spread of those distances within the series
    irregular             gap_cv above `--tolerance`, i.e. not evenly sampled
    declared_gap_mm       SpacingBetweenSlices, or SliceThickness where absent
    declared_error_mm     how far the declared value is from the measured one
    sampled_gap_mm        what one step of the cached volume actually spans
    kernel_span_mm        what a kernel of `--kernel` spans in this series

The last two are the numbers that matter for the experiment: a depth-3 kernel
is a local neighbourhood in one series and a quarter of the knee in another.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.data import dicom_io


def series_geometry(frame, cached_slices, kernel):
    """One row per series, from its slice headers."""
    positions, orientations = frame["ImagePositionPatient"], frame["ImageOrientationPatient"]
    usable = [(p, o) for p, o in zip(positions, orientations)
              if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3
              and isinstance(o, (list, tuple, np.ndarray)) and len(o) == 6]
    row = {"n_slices": len(frame), "n_with_geometry": len(usable)}
    if len(usable) < 2:
        # Ordered by InstanceNumber instead; no physical spacing is recoverable.
        return {**row, "acquired_gap_mm": np.nan, "gap_cv": np.nan,
                "declared_gap_mm": declared(frame), "has_geometry": False}

    normal = dicom_io.slice_normal(usable[0][1])
    projected = np.sort([float(np.dot(np.asarray(p, dtype=float), normal))
                         for p, _ in usable])
    gaps = np.diff(projected)
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return {**row, "acquired_gap_mm": np.nan, "gap_cv": np.nan,
                "declared_gap_mm": declared(frame), "has_geometry": False}

    median = float(np.median(gaps))
    row.update(
        has_geometry=len(usable) == len(frame),
        acquired_gap_mm=median,
        gap_cv=float(gaps.std() / median) if median else np.nan,
        gap_min_mm=float(gaps.min()), gap_max_mm=float(gaps.max()),
        extent_mm=float(projected[-1] - projected[0]),
        declared_gap_mm=declared(frame))
    row["declared_error_mm"] = (abs(row["declared_gap_mm"] - median)
                                if row["declared_gap_mm"] is not None else np.nan)
    # The cache keeps `cached_slices` positions spread across the series, so one
    # step of the cached volume is this many acquired slices, hence millimetres.
    step = (len(frame) - 1) / max(cached_slices - 1, 1)
    row["sampled_gap_mm"] = median * step
    row["kernel_span_mm"] = row["sampled_gap_mm"] * (kernel - 1)
    row["in_plane_mm"] = in_plane(frame)
    return row


def declared(frame):
    for column in ("SpacingBetweenSlices", "SliceThickness"):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        values = values[values > 0]
        if len(values):
            return float(values.median())
    return None


def in_plane(frame):
    for value in frame["PixelSpacing"]:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
            return float(np.mean([float(v) for v in value]))
    return np.nan


def audit(index_path, cached_slices, kernel, tolerance, limit=None):
    columns = ["SeriesInstanceUID", "StudyInstanceUID", "ImagePositionPatient",
               "ImageOrientationPatient", "SliceThickness", "SpacingBetweenSlices",
               "PixelSpacing", "InstanceNumber"]
    index = pd.read_parquet(index_path, columns=columns)
    grouped = index.groupby("SeriesInstanceUID", sort=True)
    rows = []
    for i, (uid, frame) in enumerate(grouped):
        if limit is not None and i >= limit:
            break
        rows.append({"SeriesInstanceUID": uid,
                     "StudyInstanceUID": frame["StudyInstanceUID"].iloc[0],
                     **series_geometry(frame, cached_slices, kernel)})
    table = pd.DataFrame(rows)
    table["irregular"] = table["gap_cv"] > tolerance
    return table


def summarise(table, cached_slices, kernel, tolerance):
    measured = table.dropna(subset=["acquired_gap_mm"])
    spans = measured["kernel_span_mm"].dropna()

    def quantiles(values):
        if not len(values):
            return {}
        q = np.quantile(values, [.05, .25, .5, .75, .95])
        return {"p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
                "p75": float(q[3]), "p95": float(q[4])}

    declared_error = measured["declared_error_mm"].dropna()
    return {
        "series": int(len(table)),
        "series_with_measurable_spacing": int(len(measured)),
        "series_without_full_geometry": int((~table["has_geometry"].fillna(False)).sum()),
        "cached_slices": cached_slices, "depth_kernel": kernel,
        "irregular_tolerance_cv": tolerance,
        "irregular_series": int(measured["irregular"].sum()),
        "max_gap_cv": float(measured["gap_cv"].max()) if len(measured) else None,
        # A series shorter than the cache is sampled with repeats, so some
        # neighbours along the cached depth axis are the same image twice and
        # the kernel sees a duplicate rather than new anatomy.
        "series_shorter_than_cache": int((table["n_slices"] < cached_slices).sum()),
        "series_shorter_than_cache_fraction": (
            float((table["n_slices"] < cached_slices).mean()) if len(table) else None),
        "irregular_fraction": float(measured["irregular"].mean()) if len(measured) else None,
        "acquired_gap_mm": quantiles(measured["acquired_gap_mm"]),
        "sampled_gap_mm": quantiles(measured["sampled_gap_mm"].dropna()),
        "kernel_span_mm": quantiles(spans),
        "kernel_span_ratio_p95_p05": (float(np.quantile(spans, .95) / np.quantile(spans, .05))
                                      if len(spans) else None),
        "in_plane_mm": quantiles(measured["in_plane_mm"].dropna()),
        "declared_vs_measured_mm": quantiles(declared_error),
        "declared_disagrees_over_0_5mm": int((declared_error > 0.5).sum()),
        "limitation": (
            "The depth axis of a cached volume is sampled slice index, not "
            "millimetres. Within a series the acquired spacing is even, but it "
            "differs between series, and a series shorter than the cache is "
            "sampled with repeats -- so a depth kernel assumes ordering only. "
            "Resampling to physical spacing is a separate experiment, not a "
            "correction this audit applies."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/meta/resolution_index/index_train.parquet")
    parser.add_argument("--cached-slices", type=int, default=config.CACHE_SLICES)
    parser.add_argument("--kernel", type=int, default=3,
                        help="depth extent of the 3D convolution kernel")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="gap coefficient of variation above which a series is irregular")
    parser.add_argument("--limit", type=int, help="audit only the first N series")
    parser.add_argument("--out", default="outputs/geometry_audit")
    args = parser.parse_args()

    table = audit(ROOT / args.index, args.cached_slices, args.kernel,
                  args.tolerance, args.limit)
    summary = summarise(table, args.cached_slices, args.kernel, args.tolerance)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "per_series.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
