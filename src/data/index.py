"""Header-only scan of the DICOM tree into a single parquet index.

Runs on Kaggle, where the competition data is mounted. Reads no pixel data, so
the cost is dominated by file-system traversal rather than decoding. The
resulting index drives cache construction and lets series be filtered and
audited without touching the 570 GB of imaging again.

Usage (in a Kaggle notebook):

    from src.data.index import build_index
    build_index("train_series", out_path="/kaggle/working/index_train.parquet")
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from src import config
from src.data import dicom_io


def iter_dicom_paths(root) -> Iterator[Path]:
    """Yield every .dcm under root/study/series/ without materialising a list."""
    root = Path(root)
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            for path in sorted(series_dir.glob("*.dcm")):
                yield path


def scan_paths(paths: List[Path], workers: int = 8) -> pd.DataFrame:
    """Read headers for a batch of files in parallel.

    Header reads are I/O bound, so threads are the right tool -- the GIL is
    released during file reads and a process pool would only add overhead.
    """
    def safe_read(path):
        try:
            return dicom_io.read_header(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a scan
            return {"path": str(path), "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(safe_read, paths))

    return pd.DataFrame(rows)


def build_index(
    split: str = "train_series",
    out_path: Optional[str] = None,
    data_root=None,
    workers: int = 8,
    batch_size: int = 5000,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Scan one split's DICOM tree and write a parquet index.

    Processes in batches so a long scan reports progress and so peak memory
    stays flat regardless of how many files the split holds.
    """
    root = Path(data_root or config.DATA_ROOT) / split
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    frames = []
    batch: List[Path] = []
    scanned = 0

    for path in iter_dicom_paths(root):
        batch.append(path)
        if limit is not None and len(batch) + scanned >= limit:
            break
        if len(batch) >= batch_size:
            frames.append(scan_paths(batch, workers=workers))
            scanned += len(batch)
            batch = []
            print(f"  scanned {scanned} files", flush=True)

    if batch:
        frames.append(scan_paths(batch, workers=workers))
        scanned += len(batch)

    if not frames:
        raise RuntimeError(f"no DICOM files found under {root}")

    index = pd.concat(frames, ignore_index=True)
    print(f"scanned {scanned} files across {index['SeriesInstanceUID'].nunique()} series")

    if "error" in index.columns:
        failures = int(index["error"].notna().sum())
        if failures:
            print(f"WARNING: {failures} files failed to parse")

    if out_path:
        os.makedirs(Path(out_path).parent, exist_ok=True)
        index.to_parquet(out_path, index=False)
        print(f"wrote {out_path}")

    return index


def summarize(index: pd.DataFrame) -> pd.DataFrame:
    """Per-series roll-up: slice count, geometry completeness, laterality.

    Surfaces the series that need attention -- missing geometry, unresolvable
    side -- before an hour of cache construction is spent on them.
    """
    records = []
    for series_uid, group in index.groupby("SeriesInstanceUID", sort=False):
        headers = group.to_dict("records")
        records.append({
            "StudyInstanceUID": headers[0].get("StudyInstanceUID"),
            "SeriesInstanceUID": series_uid,
            "n_slices": len(headers),
            "geometry_ok": dicom_io.has_geometry(headers),
            "laterality": dicom_io.resolve_laterality(headers),
            "rows": headers[0].get("Rows"),
            "cols": headers[0].get("Columns"),
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_series")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_index(split=args.split, out_path=args.out,
                workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
