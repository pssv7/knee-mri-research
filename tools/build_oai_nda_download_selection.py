#!/usr/bin/env python3
"""Cross-reference the restricted NDA image manifest with Experiment 18 MRI roles.

The derived selection remains under ignored outputs because it contains restricted
NDA record identifiers. Console output intentionally reports counts and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nda-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nda = pd.read_csv(args.nda_manifest, sep="\t", dtype=str)
    request = pd.read_csv(args.request, dtype=str)
    expected = {"collection_id", "dataset_id", "image03_id", "subjectkey", "image_file"}
    if set(nda) != expected or len(nda) != 11994:
        raise ValueError("Unexpected NDA Study 3407 exact-input manifest")

    nda["file_id"] = nda["image_file"].map(
        lambda value: Path(value).name.removesuffix(".tar.gz")
    )
    request["file_id"] = request["mri_barcode"].str[-8:]
    selected = request.merge(nda, on="file_id", how="left", validate="one_to_one")
    # The 29 absent request rows are repeat acquisitions; the public attachment
    # identifies one primary acquisition for each required knee/role.
    selected = selected[selected["image03_id"].notna()].copy()
    if len(selected) != 6458 or selected["image03_id"].nunique() != 6458:
        raise ValueError(f"Expected 6,458 primary image records, found {len(selected):,}")
    role_counts = selected["series_role"].value_counts().sort_index().to_dict()
    expected_roles = {"pf_axial": 2399, "pf_dess": 2399, "syn_sagittal": 1660}
    if role_counts != expected_roles:
        raise ValueError(f"Unexpected role counts: {role_counts}")

    args.output.mkdir(parents=True, exist_ok=True)
    selected_path = args.output / "private_selected_image03.tsv"
    guid_path = args.output / "private_subjectkeys.csv"
    image_paths = args.output / "private_image_files.txt"
    selected[[
        "collection_id", "dataset_id", "image03_id", "subjectkey", "image_file",
        "series_role", "oai_subject_id", "side", "mri_barcode",
    ]].to_csv(selected_path, sep="\t", index=False)
    pd.DataFrame({"subjectkey": sorted(selected["subjectkey"].unique())}).to_csv(
        guid_path, index=False
    )
    image_paths.write_text("\n".join(sorted(selected["image_file"].unique())) + "\n")
    receipt = {
        "status": "ready_for_nda_package",
        "study": 3407,
        "selected_image_records": len(selected),
        "subjects": int(selected["subjectkey"].nunique()),
        "knees": int(selected[["oai_subject_id", "side"]].drop_duplicates().shape[0]),
        "roles": role_counts,
        "excluded_repeat_acquisitions": len(request) - len(selected),
        "source_sha256": {
            args.nda_manifest.name: sha256(args.nda_manifest),
            args.request.name: sha256(args.request),
        },
        "output_sha256": {
            selected_path.name: sha256(selected_path),
            guid_path.name: sha256(guid_path),
            image_paths.name: sha256(image_paths),
        },
        "privacy": "Derived files contain restricted NDA identifiers and remain in ignored outputs.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in [
        "status", "selected_image_records", "subjects", "knees", "roles",
        "excluded_repeat_acquisitions",
    ]}, indent=2))


if __name__ == "__main__":
    main()
