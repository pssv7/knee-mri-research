"""Run the Kaggle notebooks end to end against a miniature fake dataset.

Kaggle time is expensive and a failure three hours into a cache build is
expensive twice. This builds a small stand-in for the competition data out of
the dev series, rewrites the /kaggle paths to a temp directory, and executes
every code cell of notebooks 01 and 02 for real -- header scan, volume build,
save, and the visual check included.

    python3 tools/check_kaggle_notebooks.py

Requires data/dev/*.dcm (see docs/dev_data.md).
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")  # no display in a terminal; the figures are still drawn

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV = ROOT / "data" / "dev"

# Enough studies that the hash-based shard assignment has something to spread,
# while keeping the run to a few seconds.
N_STUDIES = 3
SERIES_PER_STUDY = 2


def build_fake_dataset(base: pathlib.Path):
    """A miniature competition tree: base/train_series/<study>/<series>/*.dcm.

    The UIDs inside each copy are rewritten. Copying the dev files verbatim
    leaves every copy carrying the same SeriesInstanceUID, so the notebook's
    groupby collapses the whole fake dataset to a single series and the run
    proves far less than it appears to.
    """
    import pydicom

    source = sorted(DEV.glob("*.dcm"))
    if not source:
        raise SystemExit("no dev DICOM found -- see docs/dev_data.md")

    for split in ("train_series", "test_series"):
        n_studies = N_STUDIES if split == "train_series" else 1
        for study in range(n_studies):
            study_uid = f"1.2.826.0.1.3680043.8.498.{split}.study{study}"
            for series in range(SERIES_PER_STUDY):
                series_uid = f"{study_uid}.series{series}"
                target = base / split / study_uid / series_uid
                target.mkdir(parents=True, exist_ok=True)
                for path in source:
                    dataset = pydicom.dcmread(str(path))
                    dataset.StudyInstanceUID = study_uid
                    dataset.SeriesInstanceUID = series_uid
                    dataset.save_as(target / path.name)

    total = sum(1 for _ in (base / "train_series").rglob("*.dcm"))
    print(f"  fake dataset: {total} files under {base.name}/train_series")
    return base


def code_cells(path):
    with open(path) as handle:
        notebook = json.load(handle)
    return {
        index: "".join(cell["source"])
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    }


def run_cell(source, namespace, label, rewrites):
    for old, new in rewrites.items():
        source = source.replace(old, new)
    try:
        exec(source, namespace)  # noqa: S102 - validating generated output
        print(f"  {label}: OK")
        return True
    except Exception:
        print(f"  {label}: FAILED")
        traceback.print_exc()
        return False


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        data_root = base / "data"
        working = base / "working"
        working.mkdir(parents=True, exist_ok=True)
        data_root.mkdir(parents=True, exist_ok=True)

        print("setup")
        build_fake_dataset(data_root)

        # config resolves the data root from this at import time.
        os.environ["KNEE_DATA_ROOT"] = str(data_root)

        scratch = base / "scratch"
        scratch.mkdir()
        os.chdir(scratch)

        rewrites = {
            "/kaggle/working": str(working),
            "/kaggle/input/knee-index": str(working),
            "LIMIT = None": "LIMIT = None",
            "WORKERS = 8": "WORKERS = 4",
        }

        # ---- notebook 01 -------------------------------------------------
        print("\n01_scan_index.ipynb")
        cells = code_cells(ROOT / "notebooks" / "01_scan_index.ipynb")
        namespace = {"__name__": "__main__"}
        for index in sorted(cells):
            label = f"cell {index}"
            if not run_cell(cells[index], namespace, label, rewrites):
                failures.append(f"01 {label}")

        # ---- notebook 02 -------------------------------------------------
        print("\n02_build_cache.ipynb")
        cells = code_cells(ROOT / "notebooks" / "02_build_cache.ipynb")
        namespace = {"__name__": "__main__"}
        # One shard locally: sharding logic is exercised, the split is not the
        # thing under test and three shards over six series is mostly empty.
        shard_rewrites = dict(rewrites)
        shard_rewrites["N_SHARDS = 3"] = "N_SHARDS = 1"
        shard_rewrites["WORKERS = 4"] = "WORKERS = 2"

        for index in sorted(cells):
            label = f"cell {index}"
            if not run_cell(cells[index], namespace, label, shard_rewrites):
                failures.append(f"02 {label}")

        # ---- outputs -----------------------------------------------------
        print("\noutputs written")
        for path in sorted(working.iterdir()):
            print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")

    os.chdir(ROOT)
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("both notebooks ran clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
