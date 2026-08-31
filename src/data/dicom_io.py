"""DICOM reading, geometric slice ordering, and laterality resolution.

Two operations here are load-bearing and fail silently if done naively:

1. **Slice ordering.** InstanceNumber is not reliable across the 19 contributing
   institutions. Slices are ordered by projecting ImagePositionPatient onto the
   slice normal derived from ImageOrientationPatient, which is defined by the
   scanner geometry rather than by convention.

2. **Laterality.** Four of the twelve labels are compartment-specific
   (medial vs. lateral). In an image of a left knee the medial compartment sits
   on the opposite side of the frame from a right knee, so left knees must be
   mirrored into a canonical orientation before a model can learn the
   distinction.
"""

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pydicom

# Header fields pulled during the index scan. Kept explicit so the parquet
# schema is stable and the scan stays header-only.
HEADER_FIELDS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SliceThickness",
    "SpacingBetweenSlices",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "ScanningSequence",
    "SequenceVariant",
    # (0018,0010). The one header that states contrast administration
    # outright, and the index built before 2026-08-31 does not carry it --
    # so the contrast audit has to fall back on typed descriptions, 17.7% of
    # which are anonymised away. Absence of a string is not absence of
    # contrast; absence of this tag is at least a direct question asked.
    "ContrastBolusAgent",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "ImageLaterality",
    "Laterality",
    "InstanceNumber",
]

# DICOM DS/IS values carry their original string representation, which would
# land in the index as object dtype. Cast these explicitly.
NUMERIC_FIELDS = {
    "Rows",
    "Columns",
    "SliceThickness",
    "SpacingBetweenSlices",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "MagneticFieldStrength",
    "InstanceNumber",
}

_LEFT_PATTERN = re.compile(
    r"\b(l|lt|left|links|izquierda|izq|gauche|sinistra|sin|links?e)\b",
    re.IGNORECASE,
)
_RIGHT_PATTERN = re.compile(
    r"\b(r|rt|right|rechts|derecha|der|droite|destra|dx)\b",
    re.IGNORECASE,
)


def read_header(path) -> dict:
    """Read one DICOM's headers without decoding pixel data."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    row = {"path": str(path)}
    for field in HEADER_FIELDS:
        value = getattr(ds, field, None)
        if value is None:
            row[field] = None
        elif isinstance(value, pydicom.multival.MultiValue):
            # Geometry fields are numeric; ScanningSequence and friends are
            # multi-valued strings, so fall back to a joined token.
            try:
                row[field] = [float(v) for v in value]
            except (TypeError, ValueError):
                row[field] = "\\".join(str(v) for v in value)
        elif field in NUMERIC_FIELDS:
            try:
                row[field] = float(value)
            except (TypeError, ValueError):
                row[field] = None
        elif isinstance(value, (int, float, str)):
            row[field] = str(value) if isinstance(value, str) else value
        else:
            row[field] = str(value)
    return row


def slice_normal(orientation: Optional[Sequence[float]]) -> np.ndarray:
    """Unit normal of the image plane, from the two direction cosines.

    Falls back to the z axis when orientation is missing, which keeps a
    malformed series ordered by patient-superior position rather than crashing.
    """
    if orientation is None or len(orientation) != 6:
        return np.array([0.0, 0.0, 1.0])

    row_cosine = np.asarray(orientation[:3], dtype=float)
    col_cosine = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row_cosine, col_cosine)

    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        return np.array([0.0, 0.0, 1.0])
    return normal / magnitude


def slice_position(header: dict) -> float:
    """Scalar position of a slice along its own stack axis."""
    position = header.get("ImagePositionPatient")
    if position is None or len(position) != 3:
        # No geometry available; fall back to InstanceNumber so ordering is at
        # least deterministic. The index scan flags these series separately.
        number = header.get("InstanceNumber")
        return float(number) if number is not None else 0.0

    normal = slice_normal(header.get("ImageOrientationPatient"))
    return float(np.dot(np.asarray(position, dtype=float), normal))


def sort_slices(headers: Iterable[dict]) -> List[dict]:
    """Order a series' slices along the acquisition axis."""
    return sorted(headers, key=slice_position)


def has_geometry(headers: Sequence[dict]) -> bool:
    """True when every slice carries the position and orientation tags."""
    for header in headers:
        position = header.get("ImagePositionPatient")
        orientation = header.get("ImageOrientationPatient")
        if position is None or len(position) != 3:
            return False
        if orientation is None or len(orientation) != 6:
            return False
    return True


def resolve_laterality(headers: Sequence[dict]) -> Optional[str]:
    """Determine whether a series images a left or right knee.

    Prefers the explicit DICOM tags, then falls back to parsing the free-text
    description fields, which carry side markers in several languages. Returns
    "L", "R", or None when the side cannot be established -- callers must treat
    None as unusable rather than guessing, since a wrong side silently swaps the
    medial and lateral labels.
    """
    for header in headers:
        for field in ("ImageLaterality", "Laterality"):
            value = header.get(field)
            if isinstance(value, str):
                # The tag holds "L"/"R" but also "LEFT"/"RIGHT", and is far more
                # often present-but-empty than absent. An empty string is not a
                # side and must not be read as one.
                token = value.strip().upper()
                if token in ("L", "LEFT"):
                    return "L"
                if token in ("R", "RIGHT"):
                    return "R"

    for field in ("SeriesDescription", "ProtocolName", "BodyPartExamined"):
        for header in headers:
            text = header.get(field)
            if not isinstance(text, str):
                continue
            # Split on non-letters so "MR_Knie_Rechts" yields usable tokens.
            spaced = re.sub(r"[^A-Za-zÀ-ÿ]+", " ", text)
            left = bool(_LEFT_PATTERN.search(spaced))
            right = bool(_RIGHT_PATTERN.search(spaced))
            # Ambiguous strings mentioning both sides tell us nothing.
            if left and not right:
                return "L"
            if right and not left:
                return "R"

    return None


def load_series_volume(paths: Sequence[str]) -> np.ndarray:
    """Decode an ordered list of DICOM paths into a (slices, rows, cols) array.

    Applies the rescale slope and intercept when present. Slices whose in-plane
    shape differs from the first are skipped rather than silently broadcast.
    """
    slices = []
    reference_shape = None
    skipped = 0

    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            pixels = ds.pixel_array.astype(np.float32)
        except Exception:  # noqa: BLE001
            # A truncated or malformed file loses one slice, not the series.
            # The dataset contains files whose declared pixel length exceeds
            # what is actually stored, and dropping 23 good slices over one
            # bad one costs a whole study its imaging.
            skipped += 1
            continue

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            pixels = pixels * slope + intercept

        if reference_shape is None:
            reference_shape = pixels.shape
        elif pixels.shape != reference_shape:
            skipped += 1
            continue

        slices.append(pixels)

    if not slices:
        raise ValueError(f"no decodable slices in series ({skipped} unreadable)")

    return np.stack(slices, axis=0)


def series_paths(series_dir) -> List[str]:
    """All DICOM files under one series directory."""
    return sorted(str(p) for p in Path(series_dir).glob("*.dcm"))


def resolve_study_laterality(series_sides: dict) -> dict:
    """Fill in unknown series sides from other series in the same study.

    A knee MRI study images one knee, so a side established on any series
    applies to all of them. This matters because the DICOM side tags are
    populated inconsistently across the contributing institutions -- a study
    can carry the tag on its coronal series and omit it on the sagittals.

    Report text is deliberately not used as a source here. Reports exist only
    for training studies, so recovering a side from them would mirror the
    training volumes and leave the test volumes untouched -- a train/test skew
    that is worse than the missing side it fixes.

    Args:
        series_sides: {study_uid: {series_uid: "L" | "R" | None}}

    Returns:
        The same structure with unknown entries filled where the study allows.
        Studies whose series disagree are left untouched, since a conflict
        means the assumption of one knee per study does not hold there.
    """
    resolved = {}
    for study_uid, series in series_sides.items():
        known = {side for side in series.values() if side in ("L", "R")}

        if len(known) == 1:
            side = known.pop()
            resolved[study_uid] = {uid: side for uid in series}
        else:
            # No evidence, or contradictory evidence. Guessing would silently
            # swap the medial and lateral labels on four of twelve findings.
            resolved[study_uid] = dict(series)

    return resolved


def laterality_coverage(series_sides: dict) -> dict:
    """Counts before and after study-level propagation, for auditing."""
    before = sum(1 for s in series_sides.values() for v in s.values()
                 if v in ("L", "R"))
    total = sum(len(s) for s in series_sides.values())

    filled = resolve_study_laterality(series_sides)
    after = sum(1 for s in filled.values() for v in s.values()
                if v in ("L", "R"))

    conflicts = sum(
        1 for s in series_sides.values()
        if len({v for v in s.values() if v in ("L", "R")}) > 1
    )

    return {
        "total_series": total,
        "known_before": before,
        "known_after": after,
        "recovered": after - before,
        "conflicting_studies": conflicts,
    }
