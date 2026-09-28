"""Turn a raw DICOM series into a compact, model-ready uint8 volume.

The same code path runs during cache construction and at submission time. Any
divergence between the two produces a train/test preprocessing skew, which is
invisible in local validation and only shows up as a collapsed leaderboard
score -- so callers must import from here rather than reimplementing.

Output per series: (CACHE_SLICES, CACHE_SIZE, CACHE_SIZE) uint8, in the
orientation it was acquired in. Mirroring to a canonical side happens at load
time rather than here -- see config.MIRROR_IN_CACHE for why.
"""

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src import config
from src.data import dicom_io


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Scale a volume to uint8 using its own robust percentile range.

    MRI carries no standardised units -- intensity depends on scanner, field
    strength, coil, and whether fat suppression was applied -- so each volume is
    scaled against its own distribution rather than a global constant.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [config.CLIP_PCT_LOW, config.CLIP_PCT_HIGH])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        # Flat or degenerate volume; nothing to stretch.
        return np.zeros(volume.shape, dtype=np.uint8)

    scaled = (np.clip(volume, low, high) - low) / (high - low)
    return (scaled * 255.0).round().astype(np.uint8)


def resize_volume(volume: np.ndarray, size: int = None) -> np.ndarray:
    """Resample every slice to size x size with area/linear interpolation."""
    size = size or config.CACHE_SIZE
    output = np.empty((volume.shape[0], size, size), dtype=volume.dtype)

    for i, plane in enumerate(volume):
        # Area interpolation avoids the aliasing that bilinear introduces when
        # downsampling 800px acquisitions; linear is correct when upsampling.
        interp = cv2.INTER_AREA if plane.shape[0] > size else cv2.INTER_LINEAR
        output[i] = cv2.resize(plane, (size, size), interpolation=interp)

    return output


def select_slices(count: int, target: int = None) -> List[int]:
    """Indices of `target` slices spread evenly across a stack of `count`.

    Short series are handled by repeating edge slices rather than zero-padding,
    so the volume stays anatomically continuous.
    """
    target = target or config.CACHE_SLICES
    if count <= 0:
        raise ValueError("cannot select slices from an empty series")
    if count == target:
        return list(range(count))

    positions = np.linspace(0, count - 1, target)
    return [int(round(p)) for p in positions]


def mirror_axis(orientation: Optional[Sequence[float]]) -> int:
    """Which array axis runs medial-to-lateral, from the scanner geometry.

    Mirroring a knee means reflecting through the patient's left-right axis,
    and which array axis that corresponds to depends on the acquisition plane:

        coronal / axial   left-right lies in-plane  -> flip columns (axis 2)
        sagittal          left-right is the slice axis -> reverse slices (axis 0)

    A blanket column flip is wrong for sagittal series. It leaves medial and
    lateral exactly where they were -- they are separated across slices, not
    across the image -- and instead reflects anterior-posterior, moving the
    patella to the back of the knee.

    Returns 0, 1, or 2: the axis of a (slices, rows, cols) volume to reverse.
    """
    left_right = np.array([1.0, 0.0, 0.0])

    if orientation is None or len(orientation) != 6:
        # No geometry to reason from. Columns is the common case (coronal and
        # axial together outnumber sagittal), but the caller should prefer not
        # to mirror at all when geometry is missing.
        return 2

    width = np.asarray(orientation[:3], dtype=float)    # increasing column index
    height = np.asarray(orientation[3:], dtype=float)   # increasing row index
    normal = dicom_io.slice_normal(orientation)

    alignments = [
        (abs(float(normal @ left_right)), 0),
        (abs(float(height @ left_right)), 1),
        (abs(float(width @ left_right)), 2),
    ]
    return max(alignments)[1]


def canonicalize_side(volume: np.ndarray, side: Optional[str],
                      orientation: Optional[Sequence[float]] = None) -> np.ndarray:
    """Mirror left knees so the medial compartment is always on one side.

    Four labels are compartment-specific (Medial/Lateral Meniscus, Medial/
    Lateral OA). Without this the model would have to infer the side before it
    could place a finding, from a cue that is not reliably in the image.

    The axis reflected is chosen from the acquisition geometry -- see
    mirror_axis -- because it differs between sagittal and coronal series.
    """
    if side is None:
        # Unknown side: leave untouched and let the caller decide. Guessing
        # would silently swap four of the twelve labels.
        return volume
    if side.upper() == config.CANONICAL_SIDE:
        return volume

    axis = mirror_axis(orientation)
    return np.flip(volume, axis=axis).copy()


def submission_indices(count: int, slices: int) -> List[int]:
    """Raw slice indices a `slices`-deep submission actually reads.

    **Not** `select_slices(count, slices)`, and the difference is the whole
    reason this exists. The submission decodes `CACHE_SLICES` and then
    subsamples with `np.linspace(0, CACHE_SLICES - 1, TUNED_SLICES)`; picking
    `slices` evenly across the raw stack in one step lands on *different
    files* for 41 of the 44 series lengths between 16 and 60. The model would
    then be scored on anatomy it was not trained to read, and nothing would
    raise -- the shapes agree either way. That is the same failure the
    `volumes[:MAX_SERIES]` head truncation already cost this project.

    So the two selections are composed rather than collapsed: the coarse pick
    the cache would have made, then the fine pick the model applies to it.
    """
    coarse = select_slices(count)
    if slices == config.CACHE_SLICES:
        return coarse
    return [coarse[i] for i in select_slices(config.CACHE_SLICES, slices)]


def build_series_volume(headers: Sequence[dict],
                        slices: Optional[int] = None,
                        size: Optional[int] = None,
                        require_complete_decode: bool = False) -> Dict:
    """Full pipeline for one series: sort, decode, normalise, resize.

    Mirroring is applied only when config.MIRROR_IN_CACHE is set, which it is
    not by default. The recorded `laterality` is what the loader mirrors on.
    Returns the volume plus the provenance needed to audit it later.

    `slices` exists for the submission, which reads only `TUNED_SLICES` of the
    `CACHE_SLICES` this decodes: at 16 of 24 that is a third of the pixel
    decoding done for images the model never sees, and inference time is a
    *scored quantity* in the Efficiency Prize. It defaults to None, so cache
    construction is unchanged and every existing shard stays valid.

    **It is not bit-identical to decoding the full depth.**
    `normalize_intensity` scales each volume by its own 1st/99th percentiles,
    computed over whatever was decoded, so a shallower stack shifts the
    intensity mapping slightly. The slices themselves are the same files --
    see `submission_indices` -- but the uint8 values differ a little, and the
    submission notebook measures that difference rather than assuming it away.
    """
    if not headers:
        raise ValueError("no headers supplied")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise ValueError("size must be a positive integer")

    depth = config.CACHE_SLICES if slices is None else slices
    ordered = dicom_io.sort_slices(headers)
    first_header = ordered[0]
    side = dicom_io.resolve_laterality(ordered)
    geometry_ok = dicom_io.has_geometry(ordered)

    indices = submission_indices(len(ordered), depth)
    selected = [ordered[i] for i in indices]

    raw = dicom_io.load_series_volume([h["path"] for h in selected])

    # Slices skipped as unreadable leave the volume short of the requested
    # depth, and the cache is a fixed-shape array. Re-select across whatever
    # decoded so the depth is exact regardless of how many files were dropped.
    if raw.shape[0] != depth:
        if require_complete_decode:
            raise ValueError('Incomplete decode; cannot establish sampling provenance')
        raw = raw[select_slices(raw.shape[0], depth)]

    resized = resize_volume(raw.astype(np.float32), size=size)
    normalized = normalize_intensity(resized)

    # Left as acquired by default; the loader mirrors using the recorded side.
    # See config.MIRROR_IN_CACHE for why this is not baked in here.
    volume = (canonicalize_side(normalized, side,
                                first_header.get("ImageOrientationPatient"))
              if config.MIRROR_IN_CACHE else normalized)

    first = ordered[0]
    return {
        "volume": volume,
        "StudyInstanceUID": first.get("StudyInstanceUID"),
        "SeriesInstanceUID": first.get("SeriesInstanceUID"),
        "n_slices_raw": len(ordered),
        "laterality": side,
        "mirrored": bool(config.MIRROR_IN_CACHE and side is not None
                         and side.upper() != config.CANONICAL_SIDE),
        "geometry_ok": geometry_ok,
        "rows": int(first.get("Rows") or 0),
        "cols": int(first.get("Columns") or 0),
    }


def build_model_series_volume(headers, side, plane, *, duplicate_free=False, size=None):
    """Opt-in parity path: full cache normalization, mirroring, then 16 slices.

    Both arms use this path when testing the sampler. Unlike the historical
    shallow-decode shortcut, it normalizes the same full 24 slices as training.
    Existing submission callers and defaults remain unchanged.
    """
    from src.model.dataset import mirror_volume
    from src.data.sampling import sample_cached_volume
    if config.MIRROR_IN_CACHE:
        raise ValueError('Parity path requires unmirrored source caches')
    result = build_series_volume(headers, size=size, require_complete_decode=True)
    volume = mirror_volume(result['volume'], side, plane)
    reverse = str(side).strip().upper() == 'L' and plane == 'Sagittal'
    result['volume'] = sample_cached_volume(volume, result['n_slices_raw'],
        duplicate_free=duplicate_free, depth_reversed=reverse)
    return result


def build_series_record(item):
    """Picklable worker for a process pool: (series_uid, headers) -> result.

    Defined at module level deliberately. A worker defined inside a notebook
    cell cannot be pickled by name, so ProcessPoolExecutor fails under the
    spawn start method and survives under fork only by accident. Keeping it
    importable makes the cache build work the same way everywhere.

    Failures are returned rather than raised: one unreadable series must not
    end a ninety-minute run, and the manifest records what to investigate.
    """
    series_uid, headers = item
    try:
        result = build_series_volume(headers)
        return series_uid, result["volume"], {
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": result["StudyInstanceUID"],
            "n_slices_raw": result["n_slices_raw"],
            "laterality": result["laterality"],
            "mirrored": result["mirrored"],
            "geometry_ok": result["geometry_ok"],
            "rows": result["rows"],
            "cols": result["cols"],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        study_uid = headers[0].get("StudyInstanceUID") if headers else None
        return series_uid, None, {
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": study_uid,
            "n_slices_raw": len(headers) if headers else 0,
            "laterality": None,
            "mirrored": False,
            "geometry_ok": False,
            "rows": 0,
            "cols": 0,
            "error": str(exc)[:200],
        }
