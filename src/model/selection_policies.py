"""Alternative series-selection policies for an inference-only check.

`dataset.select_series(rows, 5, 'protocol')` is the frozen rule: rows sorted by
SeriesInstanceUID, then greedily cover acquisition planes, then distinct
(plane, fluid-sensitive, fat-suppressed) tuples, then spread by position. On
outer0, 205 of 691 studies carry more than five series, and in 13% of all
studies the rule drops a distinct protocol; which duplicate survives is decided
by UID order, which carries no meaning. `dataset.py` is hash-pinned by the
frozen design, so the alternatives live here and are checked against it.

Every policy uses metadata available at inference (plane and sequence flags
from the series table; slice thickness, slice count and pixel spacing from the
DICOM headers), never labels or reports.
"""
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.model.dataset import PLANE_MIRROR_AXIS, select_series, sequence_flag

REFERENCE = "protocol5"


def _protocols(rows: pd.DataFrame):
    return [(str(r.get("Anatomical_Plane", "")), sequence_flag(r.get("Fluid_Sensitive")),
             sequence_flag(r.get("Fat_Suppression"))) for _, r in rows.iterrows()]


def greedy(rows: pd.DataFrame, cap: int, tiebreak: Callable[[int, Sequence[int]], float],
           seed: Sequence[int] = ()) -> np.ndarray:
    """The frozen greedy with a replaceable third key and optional forced picks.

    With `tiebreak = spread` and no seed this is `select_series(rows, cap,
    'protocol')` exactly; the tests hold it to that.
    """
    count = len(rows)
    if count <= cap:
        return np.arange(count)
    protocols = _protocols(rows)
    chosen = list(seed)
    planes = {protocols[i][0] for i in chosen}
    covered = {protocols[i] for i in chosen}
    while len(chosen) < cap:
        remaining = [i for i in range(count) if i not in chosen]
        best = max(remaining, key=lambda i: (
            protocols[i][0] in PLANE_MIRROR_AXIS and protocols[i][0] not in planes,
            protocols[i] not in covered,
            tiebreak(i, chosen)))
        chosen.append(best)
        planes.add(protocols[best][0])
        covered.add(protocols[best])
    return np.array(sorted(chosen))


def spread(i: int, chosen: Sequence[int]) -> float:
    """The frozen third key: distance in UID order from what is already kept."""
    return min((abs(i - j) for j in chosen), default=0)


def quality_scores(rows: pd.DataFrame) -> np.ndarray:
    """Higher is better: thinner slices, then more slices, then finer pixels.

    Missing header values rank last. Encoded as one sortable float per row so
    the greedy can use it as its third key.
    """
    def column(name):
        if name not in rows:
            return np.full(len(rows), np.nan)
        return pd.to_numeric(rows[name], errors="coerce").to_numpy(float)

    thickness, slices, spacing = column("slice_thickness"), column("n_instances"), column("pixel_spacing")
    order = pd.DataFrame({"t": np.nan_to_num(thickness, nan=np.inf),
                          "n": np.nan_to_num(slices, nan=-np.inf),
                          "p": np.nan_to_num(spacing, nan=np.inf)})
    ranked = order.sort_values(["t", "n", "p"], ascending=[True, False, True], kind="stable")
    score = np.empty(len(rows))
    score[ranked.index.to_numpy()] = np.arange(len(rows), 0, -1)
    return score


def sagittal_fluid(rows: pd.DataFrame) -> Optional[int]:
    """First sagittal fluid-sensitive series in the frozen (UID) order, if any."""
    for i, p in enumerate(_protocols(rows)):
        if p[0] == "Sagittal" and p[1] == 1.0:
            return i
    return None


def choose(policy: str, rows: pd.DataFrame) -> np.ndarray:
    """Positions (into UID-sorted `rows`) that the policy keeps."""
    rows = rows.reset_index(drop=True)
    if policy == "protocol5":
        return select_series(rows, 5, "protocol")
    if policy == "protocol7":
        return select_series(rows, 7, "protocol")
    if policy == "all":
        return np.arange(len(rows))
    if policy == "sagfluid5":
        # One change only: the frozen choice stands unless it drops every
        # sagittal fluid-sensitive series, and only then is one forced in.
        frozen = select_series(rows, 5, "protocol")
        first = sagittal_fluid(rows)
        protocols = _protocols(rows)
        if first is None or any(protocols[i][:2] == ("Sagittal", 1.0) for i in frozen):
            return frozen
        return greedy(rows, 5, spread, seed=(first,))
    if policy == "quality5":
        score = quality_scores(rows)
        return greedy(rows, 5, lambda i, chosen: score[i])
    raise KeyError(policy)


POLICIES = ("protocol5", "sagfluid5", "quality5", "protocol7", "all")


def series_quality(index: pd.DataFrame) -> pd.DataFrame:
    """Per-series header summary from the instance-level DICOM index."""
    spacing = index.PixelSpacing.map(lambda v: float(np.asarray(v, float).ravel()[0])
                                     if v is not None and len(np.asarray(v).ravel()) else np.nan)
    frame = index.assign(pixel_spacing=spacing)
    return frame.groupby("SeriesInstanceUID").agg(
        slice_thickness=("SliceThickness", "median"),
        n_instances=("SOPInstanceUID", "size"),
        pixel_spacing=("pixel_spacing", "median")).reset_index()
