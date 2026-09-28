"""Versioned 24-to-16 sampling; duplicate repair is opt-in and label-free."""
import numpy as np

POLICY = 'distinct_raw_17_23_v1'


def cached_indices(raw_count, *, duplicate_free=False, depth_reversed=False):
    legacy = np.linspace(0, 23, 16).round().astype(int)
    if not duplicate_free:
        return legacy
    if isinstance(raw_count, (bool, np.bool_)) or not isinstance(raw_count, (int, np.integer)) or raw_count <= 0:
        raise ValueError('Duplicate-free sampling requires a positive integer raw count')
    if not 17 <= raw_count <= 23:
        return legacy
    coarse = np.linspace(0, raw_count - 1, 24).round().astype(int)
    _, first = np.unique(coarse, return_index=True)
    first[-1] = 23  # preserve the acquired endpoint, including its cache position
    keep = first[np.linspace(0, len(first) - 1, 16).round().astype(int)]
    return 23 - keep[::-1] if depth_reversed else keep


def sample_cached_volume(volume, raw_count, *, duplicate_free=False, depth_reversed=False):
    """Sample an already mirrored full cache, preserving its normalization.

    For affected short acquisitions, validate the implied repeated-cache slots
    before trusting the raw-count provenance. This cannot certify raw decoding
    history; that still requires a matching manifest and source index.
    """
    if volume.ndim != 3 or volume.shape[0] != 24:
        raise ValueError('Sampling requires a full 24-slice cache')
    keep = cached_indices(raw_count, duplicate_free=duplicate_free, depth_reversed=depth_reversed)
    if duplicate_free and 17 <= raw_count <= 23:
        source = volume[::-1] if depth_reversed else volume
        coarse = np.linspace(0, raw_count - 1, 24).round().astype(int)
        for i in np.flatnonzero(np.diff(coarse) == 0):
            if not np.array_equal(source[i], source[i+1]):
                raise ValueError('Cache repeats disagree with raw-count provenance')
    return volume[keep]
