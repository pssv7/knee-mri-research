"""Verify the shipped cache against the code that claims to have built it.

The local test suite proves the step-1 *functions* are correct. It cannot prove
the 29 GB of shards on Kaggle are what those functions produce today -- the
shards were built once, weeks before several changes to `src/data/`, and every
failure mode here is silent: a stale shard, a study split across shards, a
volume that normalised to all zeros, a series that failed to build and is now a
block of black images with a real label attached.

Run on a **CPU** Kaggle kernel (no GPU hours) with the competition data and the
three cache shards attached:

    !python tools/verify_cache.py --rebuild 40

The decisive check is `--rebuild`: it re-derives N random series from the raw
DICOM with today's code and compares them byte for byte against the cached
rows. Anything other than an exact match means the cache and the code have
diverged, and every number this project has measured was measured on the cache.
"""

import argparse
import glob
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import cache, dicom_io

FAILURES = []


def check(name, ok, detail=""):
    """Record and print one invariant. PASS means a failure mode is absent."""
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def find_inputs(pattern):
    """Cache shards land under /kaggle/input/<dataset>/ with unpredictable names.

    Skips `/kaggle/input/competitions` exactly as notebook 08 does: a recursive
    glob over the competition mount walks ~820,000 DICOM files for nothing.
    """
    hits = sorted(glob.glob(pattern))
    hits += sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        hits += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return hits


def load_manifests(split):
    """Every shard's manifest, tagged with the shard file it belongs to."""
    frames = []
    for cache_path in find_inputs(f"cache_{split}_*.npy"):
        manifest_path = cache_path.replace("cache_", "manifest_").replace(".npy", ".parquet")
        if not os.path.exists(manifest_path):
            stem = os.path.basename(cache_path)[len("cache_"):-len(".npy")]
            hits = [m for m in find_inputs(f"manifest_{split}_*.parquet")
                    if os.path.basename(m).startswith(f"manifest_{stem}")]
            if not hits:
                print(f"  WARNING: no manifest beside {cache_path}")
                continue
            manifest_path = hits[0]
        frame = pd.read_parquet(manifest_path)
        frame["cache_path"] = cache_path
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no cache shards found for split {split}")
    return pd.concat(frames, ignore_index=True)


def check_inventory(manifest, expected_series, expected_studies):
    n_series = manifest["SeriesInstanceUID"].nunique()
    n_studies = manifest["StudyInstanceUID"].nunique()
    n_shards = manifest["cache_path"].nunique()

    check("A1 all three shards are attached", n_shards == 3, f"{n_shards} shard(s) found")
    check("A2 series count matches the documented cache",
          n_series == expected_series,
          f"{n_series} series (expected {expected_series})")
    check("A3 study count matches the documented corpus",
          n_studies == expected_studies,
          f"{n_studies} studies (expected {expected_studies})")
    check("A4 no duplicate series across shards",
          len(manifest) == n_series,
          f"{len(manifest)} rows for {n_series} distinct series")

    if "error" in manifest.columns:
        failed = manifest[manifest["error"].notna()]
        check("A5 no series failed to build",
              len(failed) == 0,
              f"{len(failed)} failed; these rows are ZEROS in the cache and are "
              f"dropped by StudyIndex -- confirm the drop actually happens\n"
              + failed[["SeriesInstanceUID", "error"]].head(10).to_string()
              if len(failed) else "0 build errors recorded")

    if "geometry_ok" in manifest.columns:
        bad_geom = int((~manifest["geometry_ok"].astype(bool)).sum())
        print(f"[INFO] {bad_geom} series lack complete geometry "
              f"({100 * bad_geom / len(manifest):.2f}%) -- these sorted by "
              f"InstanceNumber, which is unreliable")

    if "laterality_resolved" in manifest.columns:
        known = manifest["laterality_resolved"].isin(["L", "R"]).sum()
        print(f"[INFO] laterality after study propagation: {known}/{len(manifest)} "
              f"known ({100 * known / len(manifest):.1f}%), "
              f"{100 * (1 - known / len(manifest)):.1f}% left to the classifier")

    if "n_slices_raw" in manifest.columns:
        raw = manifest["n_slices_raw"]
        short = int((raw < config.CACHE_SLICES).sum())
        print(f"[INFO] acquired depth: median {int(raw.median())}, "
              f"range {int(raw.min())}-{int(raw.max())}; "
              f"{short} series ({100 * short / len(manifest):.1f}%) are shorter "
              f"than {config.CACHE_SLICES} and had slices repeated")


def check_sharding(manifest):
    """Every series of a study must live in one shard, or study assembly is partial."""
    per_study = manifest.groupby("StudyInstanceUID")["cache_path"].nunique()
    split = per_study[per_study > 1]
    check("B1 no study is split across shards",
          len(split) == 0,
          f"{len(split)} studies span multiple shards -- those studies train on "
          f"a subset of their own series\n{split.head().to_string()}"
          if len(split) else f"all {len(per_study)} studies are shard-local")

    # The sharding rule itself, re-derived rather than trusted.
    def shard_of(study_uid):
        return int(hashlib.md5(str(study_uid).encode()).hexdigest(), 16) % 3

    shard_index = {path: i for i, path in enumerate(sorted(manifest["cache_path"].unique()))}
    if len(shard_index) == 3:
        actual = manifest["cache_path"].map(shard_index)
        expected = manifest["StudyInstanceUID"].map(shard_of)
        agree = float((actual == expected).mean())
        check("B2 shard assignment reproduces md5(study_uid) % 3",
              agree > 0.999,
              f"{100 * agree:.2f}% of series sit in the shard the hash predicts "
              f"(shard files sorted by name; a low number may mean the file "
              f"order differs from the SHARD constant, not that the rule broke)")


def check_volumes(manifest, sample):
    """Find volumes that are blank or degenerate -- normalize_intensity's silent path."""
    rng = random.Random(0)
    rows = manifest.sample(n=min(sample, len(manifest)), random_state=0)

    blank, low_var, stats = 0, 0, []
    by_shard = {}
    for path in rows["cache_path"].unique():
        by_shard[path] = np.load(path, mmap_mode="r")

    for _, row in rows.iterrows():
        shard = by_shard[row["cache_path"]]
        # One mid slice per series keeps this to ~50 KB of reads per volume.
        plane = np.asarray(shard[int(row["position"])][config.CACHE_SLICES // 2])
        if plane.max() == 0:
            blank += 1
        if plane.std() < 1.0:
            low_var += 1
        stats.append((float(plane.mean()), float(plane.std())))

    means = np.array([s[0] for s in stats])
    stds = np.array([s[1] for s in stats])
    check(f"C1 no blank volumes in a {len(rows)}-series sample",
          blank == 0, f"{blank} all-zero mid slices")
    check(f"C2 no degenerate (near-constant) volumes in the sample",
          low_var == 0, f"{low_var} mid slices with std < 1.0/255")
    print(f"[INFO] sampled intensity: mean {means.mean():.1f}+/-{means.std():.1f}, "
          f"per-slice std {stds.mean():.1f}+/-{stds.std():.1f} (uint8 scale)")
    return by_shard


def check_rebuild(manifest, by_shard, count, split):
    """The decisive check: does today's code reproduce the shipped bytes?"""
    root = Path(config.DATA_ROOT) / split
    if not root.exists():
        print(f"[SKIP] D  competition data not attached ({root} missing); "
              f"the rebuild check cannot run")
        return

    rows = manifest[manifest.get("error").isna()] if "error" in manifest.columns else manifest
    rows = rows.sample(n=min(count, len(rows)), random_state=1)

    exact, differing, missing = 0, [], 0
    for _, row in rows.iterrows():
        series_dir = root / str(row["StudyInstanceUID"]) / str(row["SeriesInstanceUID"])
        paths = sorted(series_dir.glob("*.dcm"))
        if not paths:
            missing += 1
            continue

        headers = [dicom_io.read_header(p) for p in paths]
        rebuilt = cache.build_series_volume(headers)["volume"]
        cached = np.asarray(by_shard[row["cache_path"]][int(row["position"])])

        if np.array_equal(rebuilt, cached):
            exact += 1
        else:
            diff = np.abs(rebuilt.astype(int) - cached.astype(int))
            differing.append((row["SeriesInstanceUID"], float(diff.mean()), int(diff.max())))

    compared = exact + len(differing)
    check(f"D1 all {compared} rebuilt series are byte-identical to the cache",
          not differing,
          f"{exact}/{compared} exact, {missing} series not found on disk\n"
          + "\n".join(f"{uid[-12:]}  mean |diff| {m:.3f}  max {x}"
                      for uid, m, x in differing[:10])
          if differing else f"{exact}/{compared} exact, {missing} not found on disk")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_series")
    parser.add_argument("--sample", type=int, default=2000,
                        help="series to scan for blank/degenerate volumes")
    parser.add_argument("--rebuild", type=int, default=40,
                        help="series to rebuild from raw DICOM and compare byte for byte")
    parser.add_argument("--expect-series", type=int, default=24371)
    parser.add_argument("--expect-studies", type=int, default=4407)
    args = parser.parse_args()

    print(f"cache geometry from config: {config.CACHE_SLICES} x {config.CACHE_SIZE} "
          f"x {config.CACHE_SIZE} {config.CACHE_DTYPE}, "
          f"percentiles {config.CLIP_PCT_LOW}-{config.CLIP_PCT_HIGH}, "
          f"MIRROR_IN_CACHE={config.MIRROR_IN_CACHE}")
    print(f"data root: {config.DATA_ROOT}\n")

    manifest = load_manifests(args.split)
    check_inventory(manifest, args.expect_series, args.expect_studies)
    print()
    check_sharding(manifest)
    print()
    by_shard = check_volumes(manifest, args.sample)
    print()
    check_rebuild(manifest, by_shard, args.rebuild, args.split)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed -- the shipped cache matches the code that reads it")


if __name__ == "__main__":
    main()
