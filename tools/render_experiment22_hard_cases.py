#!/usr/bin/env python3
"""Render blinded MRI montages for Experiment 22 from the 384px cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ID = "StudyInstanceUID"


def unique(root: Path, name: str):
    found = [path for path in root.rglob(name) if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {found}")
    return found[0]


def truthy(value):
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def load_index(cache_root: Path):
    frames = []
    arrays = {}
    for shard in range(6):
        array_path = unique(cache_root, f"cache_train_series_{shard:02d}.npy")
        manifest_path = unique(cache_root, f"manifest_train_series_{shard:02d}.parquet")
        array = np.load(array_path, mmap_mode="r")
        if array.dtype != np.uint8 or tuple(array.shape[1:]) != (24, 384, 384):
            raise RuntimeError(f"Invalid cache geometry: {array_path} {array.shape}")
        frame = pd.read_parquet(manifest_path)
        if len(frame) != len(array) or frame.position.duplicated().any():
            raise RuntimeError("Cache manifest mismatch: " + str(manifest_path))
        frame["array_path"] = str(array_path)
        frames.append(frame)
        arrays[str(array_path)] = array
    manifest = pd.concat(frames, ignore_index=True)
    if len(manifest) != 24_371 or manifest.SeriesInstanceUID.duplicated().any():
        raise RuntimeError("Expected 24,371 unique cached RSNA series")
    series = pd.read_csv(unique(cache_root, "train_series.csv"), dtype={ID: str, "SeriesInstanceUID": str})
    manifest[ID] = manifest[ID].astype(str)
    manifest["SeriesInstanceUID"] = manifest.SeriesInstanceUID.astype(str)
    return manifest.merge(series, on=[ID, "SeriesInstanceUID"], how="left", validate="one_to_one"), arrays


def choose_series(rows: pd.DataFrame, arm: str):
    rows = rows[rows.error.isna()].copy() if "error" in rows else rows.copy()
    rows["fluid"] = rows.Fluid_Sensitive.map(truthy)
    rows["fat"] = rows.Fat_Suppression.map(truthy)
    if arm == "pfoa":
        desired = ["Axial", "Sagittal"]
        rows["rank"] = (~rows.fat).astype(int)
    else:
        desired = ["Sagittal", "Axial"]
        rows["rank"] = (~rows.fluid).astype(int) * 2 + (~rows.fat).astype(int)
    chosen = []
    for plane in desired:
        candidates = rows[rows.Anatomical_Plane.astype(str) == plane]
        if len(candidates):
            chosen.append(candidates.sort_values(["rank", "SeriesInstanceUID"], kind="stable").iloc[0])
    if len(chosen) != 2:
        raise RuntimeError(f"Study lacks required axial/sagittal series for {arm}")
    return chosen


def render_case(case, rows, arrays, destination: Path):
    chosen = choose_series(rows, case.arm)
    count = 8
    fig, axes = plt.subplots(2, count, figsize=(16, 4.5), facecolor="black")
    for row_index, series in enumerate(chosen):
        volume = arrays[series.array_path][int(series.position)]
        indices = np.linspace(2, len(volume) - 3, count).round().astype(int)
        for axis, slice_index in zip(axes[row_index], indices):
            axis.imshow(volume[slice_index], cmap="gray", vmin=0, vmax=255)
            axis.set_title(f"{series.Anatomical_Plane} {slice_index + 1}/{len(volume)}",
                           color="white", fontsize=8)
            axis.axis("off")
    fig.suptitle(f"{case.case_id}  |  {str(case.arm).upper()}", color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.15)
    fig.savefig(destination, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir()
    selection = pd.read_parquet(args.selection)
    review = pd.read_csv(args.template)
    if len(selection) != 80 or len(review) != 80 or set(selection.case_id) != set(review.case_id):
        raise RuntimeError("Experiment 22 selection/template contract failed")
    index, arrays = load_index(args.cache_root)
    grouped = {uid: rows for uid, rows in index.groupby(ID, sort=False)}
    missing = sorted(set(selection[ID].astype(str)) - set(grouped))
    if missing:
        raise RuntimeError(f"{len(missing)} selected studies are absent from the cache")
    rendered = []
    for case in tqdm(selection.itertuples(index=False), total=len(selection),
                     desc="Render blinded cases", dynamic_ncols=True):
        name = f"{case.case_id}_{case.arm}.png"
        destination = images / name
        render_case(case, grouped[str(getattr(case, ID))], arrays, destination)
        if destination.stat().st_size < 50_000:
            raise RuntimeError("Suspiciously small montage: " + name)
        rendered.append(name)
    review = review.set_index("case_id").loc[selection.case_id].reset_index()
    review["montage_file"] = rendered
    review = review[["case_id", "arm", "montage_file", "expert_label",
                     "image_quality", "reviewer_confidence", "notes"]]
    review.to_csv(args.output / "expert_label_template.csv", index=False)
    instructions = """Experiment 22 blinded knee MRI review

Review only the target named in the arm column.
- pfoa: patellofemoral osteoarthritis
- synovitis: synovitis

Enter expert_label as 0 (absent) or 1 (present).
Enter image_quality as diagnostic, limited, or unreadable.
Enter reviewer_confidence as low, medium, or high.
Do not infer the label from the case identifier; cases are blinded.
"""
    (args.output / "REVIEW_INSTRUCTIONS.txt").write_text(instructions)
    packet = args.output / "experiment22_blinded_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.output / "expert_label_template.csv", "expert_label_template.csv")
        archive.write(args.output / "REVIEW_INSTRUCTIONS.txt", "REVIEW_INSTRUCTIONS.txt")
        for name in rendered:
            archive.write(images / name, "images/" + name)
    with zipfile.ZipFile(packet) as archive:
        names = archive.namelist()
        if len([name for name in names if name.startswith("images/")]) != 80:
            raise RuntimeError("Review packet does not contain 80 images")
        if any("sealed" in name.lower() for name in names):
            raise RuntimeError("Sealed metadata leaked into the blinded packet")
    receipt = {"status": "complete", "experiment": 22, "cases": 80,
               "montages": len(rendered), "packet": packet.name,
               "sealed_metadata_in_packet": False, "gpu_required": False}
    (args.output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
