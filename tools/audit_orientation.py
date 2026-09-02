"""Corpus-wide audit of the two mirroring rulers, header-only.

Four of the twelve findings are compartment-specific, so mirroring a knee along
the wrong axis does not add noise -- it *transposes* Medial/Lateral Meniscus and
Medial/Lateral OA for that study. Two independent implementations decide that
axis:

    cache.mirror_axis          from ImageOrientationPatient (scanner geometry)
    dataset.PLANE_MIRROR_AXIS  from the Anatomical_Plane string in the CSV

Only the second one runs in production (`MIRROR_IN_CACHE` is False, so
`canonicalize_side` is never reached). The first is the better definition --
continuous, and it handles obliquely-prescribed series -- and it is dead code.
Nobody has ever checked that they agree across the corpus.

They can only diverge once a series is tilted more than 45 degrees off its
nominal plane, i.e. only when the CSV's plane label is flatly wrong. This
script measures whether that happens, and how close to the boundary the rest of
the corpus sits.

Decodes **no pixel data**. Runs on a CPU kernel in minutes off the existing
index; only falls back to reading DICOM headers if no index is attached.

    !python tools/audit_orientation.py --out /kaggle/working/orientation_audit.parquet

Exit status is non-zero if any series would be mirrored along different axes by
the two rulers.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import cache, dicom_io
from src.model import dataset as knee_dataset

# Left-right in DICOM patient coordinates. The axis most aligned with this is
# the one a mirror has to reflect through.
LEFT_RIGHT = np.array([1.0, 0.0, 0.0])

AXIS_NAMES = {0: "reverse slices", 1: "flip rows", 2: "flip columns"}


def find_one(pattern):
    """Locate an input on Kaggle, where dataset directory names are unpredictable.

    Deliberately shaped like notebook 08's `find_inputs`: shallow candidates
    first, and **never** a recursive walk of `/kaggle/input/competitions`. The
    competition mount holds ~820,000 DICOM files under 24,371 directories, and
    a `**` glob across it costs minutes before returning nothing useful.
    """
    for candidate in (f"data/meta/{pattern}", f"data/{pattern}", pattern,
                      f"/kaggle/working/{pattern}",
                      f"/kaggle/input/competitions/*/{pattern}",
                      f"/kaggle/input/*/{pattern}"):
        hits = sorted(glob.glob(candidate))
        if hits:
            return hits[0]

    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits[0]
    return None


def obliquity_degrees(orientation):
    """Angle between the slice normal and the patient left-right axis.

    0 degrees is a textbook sagittal acquisition (normal points left-right);
    90 degrees is a textbook coronal or axial one. `mirror_axis` switches its
    answer at 45, so this is the margin to a flip -- a series at 44 degrees
    agrees with a "Sagittal" label only barely.
    """
    normal = dicom_io.slice_normal(orientation)
    cosine = abs(float(np.dot(normal, LEFT_RIGHT)))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def within_series_spread(orientations, distinct):
    """Does a series' orientation vary enough to change the mirror decision?

    Scanners write per-slice orientation, and the six floats are not always
    bit-identical down a stack -- 13 of the 24,371 series carry two distinct
    tuples. That is not by itself a defect: what matters is whether the
    *derived* axis differs, because that is the only thing the pipeline reads.
    Reporting raw tuple inequality as a failure would be a false alarm, and an
    audit that cries wolf stops being read.

    Only computed when the tuples actually differ, so the common case costs
    nothing.
    """
    if distinct <= 1 or not orientations:
        return {"distinct_axes": 1, "obliquity_spread_deg": 0.0}

    axes = {cache.mirror_axis(o) for o in orientations}
    degrees = [obliquity_degrees(o) for o in orientations]
    return {"distinct_axes": len(axes),
            "obliquity_spread_deg": float(max(degrees) - min(degrees))}


def series_from_index(index_path):
    """One row per series from the header index, plus within-series consistency."""
    wanted = ["SeriesInstanceUID", "StudyInstanceUID", "ImageOrientationPatient",
              "ImagePositionPatient", "ImageLaterality", "Laterality",
              "SeriesDescription", "ProtocolName", "BodyPartExamined"]
    index = pd.read_parquet(index_path)
    present = [c for c in wanted if c in index.columns]
    index = index[present]
    print(f"index: {len(index):,} headers, {index['SeriesInstanceUID'].nunique():,} series")

    rows = []
    for series_uid, group in index.groupby("SeriesInstanceUID", sort=False):
        headers = group.to_dict("records")
        raw = [h["ImageOrientationPatient"] for h in headers
               if isinstance(h.get("ImageOrientationPatient"), (list, np.ndarray))
               and len(h["ImageOrientationPatient"]) == 6]
        orientations = [tuple(np.round(o, 4)) for o in raw]

        row = {
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": headers[0].get("StudyInstanceUID"),
            "orientation": list(raw[0]) if raw else None,
            "n_slices": len(headers),
            "n_with_orientation": len(raw),
            "distinct_orientations": len(set(orientations)),
            "laterality_tag": dicom_io.resolve_laterality(headers),
        }
        row.update(within_series_spread(raw, row["distinct_orientations"]))
        rows.append(row)
    return pd.DataFrame(rows)


def series_from_dicom(split):
    """Fallback: read one header per series straight off the DICOM tree."""
    root = Path(config.DATA_ROOT) / split
    if not root.exists():
        raise SystemExit(f"no index and no DICOM tree at {root}")

    rows = []
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            paths = sorted(series_dir.glob("*.dcm"))
            if not paths:
                continue
            # Orientation is fixed within an acquisition; two headers is enough
            # to notice when it is not.
            probe = [dicom_io.read_header(p) for p in paths[:2]]
            raw = [h["ImageOrientationPatient"] for h in probe
                   if isinstance(h.get("ImageOrientationPatient"), (list, np.ndarray))
                   and len(h["ImageOrientationPatient"]) == 6]
            orientations = [tuple(np.round(o, 4)) for o in raw]
            row = {
                "SeriesInstanceUID": series_dir.name,
                "StudyInstanceUID": study_dir.name,
                "orientation": list(raw[0]) if raw else None,
                "n_slices": len(paths),
                "n_with_orientation": len(raw),
                "distinct_orientations": len(set(orientations)),
                "laterality_tag": dicom_io.resolve_laterality(probe),
            }
            row.update(within_series_spread(raw, row["distinct_orientations"]))
            rows.append(row)
        if len(rows) % 500 == 0 and rows:
            print(f"  scanned {len(rows)} series", flush=True)
    return pd.DataFrame(rows)


def audit(series, series_meta):
    """Attach both rulers' answers to every series and compare them."""
    frame = series.merge(
        series_meta[["SeriesInstanceUID", "Anatomical_Plane"]],
        on="SeriesInstanceUID", how="left")

    frame["geometry_axis"] = [
        cache.mirror_axis(o) if o is not None else None for o in frame["orientation"]
    ]
    frame["plane_axis"] = [
        knee_dataset.PLANE_MIRROR_AXIS.get(str(p), knee_dataset.DEFAULT_MIRROR_AXIS)
        for p in frame["Anatomical_Plane"]
    ]
    frame["obliquity_deg"] = [
        obliquity_degrees(o) if o is not None else None for o in frame["orientation"]
    ]
    frame["has_geometry"] = frame["orientation"].notna()
    frame["plane_known"] = frame["Anatomical_Plane"].isin(knee_dataset.PLANE_MIRROR_AXIS)
    frame["axes_agree"] = (frame["geometry_axis"] == frame["plane_axis"]) | ~frame["has_geometry"]
    return frame


def report(frame):
    failures = []
    total = len(frame)

    # --- the question the script exists to answer -------------------------
    comparable = frame[frame["has_geometry"]]
    disagree = comparable[comparable["geometry_axis"] != comparable["plane_axis"]]
    ok = len(disagree) == 0
    if not ok:
        failures.append("axis disagreement")
    print(f"[{'PASS' if ok else 'FAIL'}] the two mirroring rulers agree on every series "
          f"with geometry")
    print(f"         {len(comparable) - len(disagree):,}/{len(comparable):,} agree; "
          f"{len(disagree):,} disagree")
    if len(disagree):
        affected = disagree["StudyInstanceUID"].nunique()
        print(f"         {affected:,} studies affected -- these have Medial/Lateral "
              f"Meniscus and Medial/Lateral OA transposed today")
        print(disagree.groupby(["Anatomical_Plane", "geometry_axis"])
              .size().to_string())

    # --- how robust that agreement is -------------------------------------
    print()
    print("[INFO] obliquity: angle between the slice normal and patient left-right")
    print("       (mirror_axis switches its answer at 45 degrees)")
    for plane, group in comparable.groupby("Anatomical_Plane"):
        deg = group["obliquity_deg"]
        near = int(((deg > 35) & (deg < 55)).sum())
        print(f"       {plane:<9} n={len(group):>6,}  median {deg.median():>5.1f}deg  "
              f"range {deg.min():>5.1f}-{deg.max():>5.1f}  "
              f"{near:>5,} within 10deg of the 45deg boundary")

    # --- the branches that silently take a default ------------------------
    print()
    no_geom = frame[~frame["has_geometry"]]
    ok = len(no_geom) == 0
    print(f"[{'PASS' if ok else 'WARN'}] every series carries ImageOrientationPatient")
    print(f"         {len(no_geom):,}/{total:,} lack it. These sort by InstanceNumber "
          f"(unreliable) and, if the geometry ruler were ever made live, would "
          f"default to '{AXIS_NAMES[2]}' -- wrong for a sagittal series.")
    if len(no_geom):
        print(no_geom["Anatomical_Plane"].value_counts(dropna=False)
              .to_string(header=False))

    unknown_plane = frame[~frame["plane_known"]]
    ok = len(unknown_plane) == 0
    if not ok:
        failures.append("unknown plane string")
    print()
    print(f"[{'PASS' if ok else 'FAIL'}] every series has a recognised Anatomical_Plane")
    print(f"         {len(unknown_plane):,}/{total:,} unrecognised or missing -- these "
          f"silently take DEFAULT_MIRROR_AXIS = {knee_dataset.DEFAULT_MIRROR_AXIS} "
          f"('{AXIS_NAMES[knee_dataset.DEFAULT_MIRROR_AXIS]}') in the LIVE path")
    if len(unknown_plane):
        print(unknown_plane["Anatomical_Plane"].value_counts(dropna=False)
              .to_string(header=False))

    split_axis = frame[frame["distinct_axes"] > 1]
    ok = len(split_axis) == 0
    if not ok:
        failures.append("mirror axis varies within a series")
    print()
    print(f"[{'PASS' if ok else 'FAIL'}] the mirror axis is constant down every series")
    print(f"         {len(split_axis):,}/{total:,} series would mirror different "
          f"slices along different axes -- one axis per series is then the wrong model")

    mixed = frame[frame["distinct_orientations"] > 1]
    if len(mixed):
        spread = mixed["obliquity_spread_deg"]
        print(f"         ({len(mixed):,} series carry more than one orientation tuple, "
              f"but only as float variation: obliquity spread max "
              f"{spread.max():.3f}deg, far below the 45deg that would change the "
              f"answer. Not a defect.)")

    # --- the other half of the mirror decision ----------------------------
    print()
    sides = {}
    for study, group in frame.groupby("StudyInstanceUID"):
        sides[study] = dict(zip(group["SeriesInstanceUID"], group["laterality_tag"]))
    coverage = dicom_io.laterality_coverage(sides)
    print("[INFO] laterality from DICOM tags alone (before the classifier):")
    print(f"       known before study propagation: {coverage['known_before']:,}"
          f"/{coverage['total_series']:,} "
          f"({100 * coverage['known_before'] / coverage['total_series']:.1f}%)")
    print(f"       known after:                    {coverage['known_after']:,}"
          f"/{coverage['total_series']:,} "
          f"({100 * coverage['known_after'] / coverage['total_series']:.1f}%), "
          f"recovered {coverage['recovered']:,}")
    print(f"       studies whose series disagree on the side: "
          f"{coverage['conflicting_studies']:,} "
          f"(left unresolved by design -- a conflict breaks the one-knee-per-study "
          f"assumption)")
    unknown = coverage["total_series"] - coverage["known_after"]
    print(f"       {unknown:,} series ({100 * unknown / coverage['total_series']:.1f}%) "
          f"have no side from metadata and rest entirely on the classifier")

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_series")
    parser.add_argument("--index", default=None,
                        help="header index parquet; auto-discovered if omitted")
    parser.add_argument("--series-csv", default=None,
                        help="train_series.csv; auto-discovered if omitted")
    parser.add_argument("--out", default=None,
                        help="write the per-series audit rows here as parquet")
    parser.add_argument("--scan", action="store_true",
                        help="read DICOM headers directly instead of using an index")
    args = parser.parse_args()

    csv_name = "test_series.csv" if args.split == "test_series" else "train_series.csv"
    series_csv = args.series_csv or find_one(csv_name)
    if not series_csv:
        raise SystemExit(f"{csv_name} not found")
    series_meta = pd.read_csv(series_csv)
    print(f"series metadata: {series_csv}  ({len(series_meta):,} rows)")

    if args.scan:
        series = series_from_dicom(args.split)
    else:
        index_name = ("index_test.parquet" if args.split == "test_series"
                      else "index_train.parquet")
        index_path = args.index or find_one(index_name)
        if index_path:
            print(f"header index: {index_path}")
            series = series_from_index(index_path)
        else:
            print(f"no {index_name} found; falling back to a DICOM header scan")
            series = series_from_dicom(args.split)

    print(f"series audited: {len(series):,}\n")
    frame = audit(series, series_meta)
    failures = report(frame)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        keep = [c for c in frame.columns if c != "orientation"]
        frame[keep].to_parquet(args.out, index=False)
        print(f"\nper-series audit written to {args.out}")

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        raise SystemExit(1)
    print("the two mirroring rulers agree across the corpus")


if __name__ == "__main__":
    main()
