#!/usr/bin/env python3
"""Build the frozen OAI baseline targets and MRI request for Experiment 18.

The source files are the baseline ASCII tables distributed by the OAI portal.
No competition validation labels are read here. Repeated MOAKS readings are
combined per knee and disagreements are retained with lower loss weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["ID", "SIDE"]
PF_FIELDS = ["V00MCMPM", "V00MCMPL", "V00MCMFMA", "V00MCMFLA"]
SERIES = {
    "pf_axial": "AXIAL MPR",
    "pf_dess": "SAG 3D DESS WE",
    "syn_sagittal": "SAG IW TSE FS",
}
SOFT_ORDINAL = {0: 0.0, 1: 0.4, 2: 0.7, 3: 1.0}
PF_SOFT = {
    "0": 0.0,
    "1": 0.3,
    "1.1": 0.4,
    "2": 0.65,
    "2.1": 0.75,
    "2.2": 0.85,
    "3": 0.8,
    "3.1": 0.9,
    "3.2": 0.95,
    "3.3": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).split(":", 1)[0].strip()
    return None if not value or value == "." else value


def ordinal(value: object) -> float:
    value = code(value)
    if value is None:
        return np.nan
    try:
        return SOFT_ORDINAL[int(float(value))]
    except (KeyError, ValueError):
        return np.nan


def pf_read(row: pd.Series) -> float:
    grades = [PF_SOFT.get(code(row[field]), np.nan) for field in PF_FIELDS]
    grades = [value for value in grades if np.isfinite(value)]
    return max(grades) if grades else np.nan


def aggregate(values: list[float]) -> tuple[float, float, int, bool]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return np.nan, 0.0, 0, False
    disagreement = len({round(value, 6) for value in values}) > 1
    # Multiple agreeing reads receive full weight. A single read is reliable but
    # cannot establish inter-reader agreement; conflicts are explicitly softened.
    weight = 0.70 if disagreement else (1.0 if len(values) > 1 else 0.85)
    return float(np.median(values)), weight, len(values), disagreement


def build_targets(moaks: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (subject, side), group in moaks.groupby(KEY, sort=True):
        pf_values = [pf_read(row) for _, row in group.iterrows()]
        syn_values = [ordinal(value) for value in group["V00MSYIC"]]
        eff_values = [ordinal(value) for value in group["V00MEFFWK"]]
        baker_values = [ordinal(value) for value in group["V00MPOPCYS"]]
        record: dict[str, object] = {
            "oai_subject_id": str(subject),
            "side": "R" if str(side).startswith("1") else "L",
            "source_projects": "+".join(sorted(group["READPRJ"].dropna().unique())),
            "source_rows": len(group),
        }
        for name, values in [
            ("PF OA", pf_values),
            ("Synovitis", syn_values),
            ("Effusion", eff_values),
            ("Baker's", baker_values),
        ]:
            target, weight, reads, conflict = aggregate(values)
            record[name] = target
            record[f"{name}_weight"] = weight
            record[f"{name}_reads"] = reads
            record[f"{name}_reader_conflict"] = conflict
        records.append(record)
    targets = pd.DataFrame.from_records(records)
    # The public OAI cohort is defined by an available baseline PF assessment.
    return targets[targets["PF OA_weight"] > 0].reset_index(drop=True)


def build_image_request(targets: pd.DataFrame, mri: pd.DataFrame) -> pd.DataFrame:
    desired = mri[mri["V00MEXAMTP"].str.endswith(tuple(SERIES.values()), na=False)].copy()
    desired["side"] = desired["V00MEXAMTP"].str[0]
    desired["series_role"] = desired["V00MEXAMTP"].str[2:].map(
        {value: key for key, value in SERIES.items()}
    )
    desired = desired.rename(columns={
        "ID": "oai_subject_id",
        "V00MEXAMTP": "exam_type",
        "V00MRBARCD": "mri_barcode",
        "V00MRCOMP": "mri_complete",
        "V00QCRESLT": "qc_result",
        "V00MRTECID": "scanner_id",
    })
    columns = [
        "oai_subject_id", "side", "series_role", "exam_type", "mri_barcode",
        "mri_complete", "qc_result", "scanner_id",
    ]
    desired = desired[columns]
    target_columns = [
        "oai_subject_id", "side", "Synovitis_weight", "Effusion_weight", "Baker's_weight"
    ]
    request = targets[target_columns].merge(
        desired, on=["oai_subject_id", "side"], how="left", validate="one_to_many"
    )
    has_synovial_target = request[
        ["Synovitis_weight", "Effusion_weight", "Baker's_weight"]
    ].max(axis=1) > 0
    request = request[(request["series_role"] != "syn_sagittal") | has_synovial_target]
    return request[columns].sort_values(
        ["oai_subject_id", "side", "series_role"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    moaks_path = args.source / "kMRI_SQ_MOAKS_BICL00.txt"
    mri_path = args.source / "MRI00.txt"
    moaks = pd.read_csv(moaks_path, sep="|", dtype=str, low_memory=False)
    mri = pd.read_csv(mri_path, sep="|", dtype=str, low_memory=False)
    missing = sorted((set(KEY + PF_FIELDS + ["READPRJ", "V00MSYIC", "V00MEFFWK", "V00MPOPCYS"]) - set(moaks)))
    if missing:
        raise ValueError(f"MOAKS source is missing columns: {missing}")

    targets = build_targets(moaks)
    request = build_image_request(targets, mri)
    if len(targets) != 2399:
        raise ValueError(f"Expected frozen 2,399-knee cohort; found {len(targets):,}")
    syn_knees = int((targets[["Synovitis_weight", "Effusion_weight", "Baker's_weight"]]
                     .max(axis=1) > 0).sum())
    expected_request = len(targets) * 2 + syn_knees
    coverage = request.groupby(["oai_subject_id", "side"])["series_role"].agg(set)
    if not coverage.map(lambda roles: {"pf_axial", "pf_dess"} <= roles).all():
        raise ValueError("PF imaging is unavailable for at least one cohort knee")
    syn_keys = targets.loc[
        targets[["Synovitis_weight", "Effusion_weight", "Baker's_weight"]].max(axis=1) > 0,
        ["oai_subject_id", "side"],
    ]
    syn_request = syn_keys.merge(
        request[request["series_role"] == "syn_sagittal"],
        on=["oai_subject_id", "side"], how="left",
    )
    if syn_request["mri_barcode"].isna().any() or request["mri_barcode"].isna().any():
        raise ValueError("A required MRI series has no downloadable barcode")

    args.output.mkdir(parents=True, exist_ok=True)
    targets_path = args.output / "oai_experiment18_targets.csv"
    request_path = args.output / "oai_experiment18_mri_request.csv"
    barcodes_path = args.output / "oai_experiment18_mri_barcodes.txt"
    targets.to_csv(targets_path, index=False)
    request.to_csv(request_path, index=False)
    barcodes = request["mri_barcode"].dropna().astype(str).drop_duplicates().sort_values()
    barcodes_path.write_text("\n".join(barcodes) + "\n")
    summary = {
        "cohort_knees": len(targets),
        "unique_subjects": int(targets["oai_subject_id"].nunique()),
        "requested_series": len(request),
        "minimum_requested_series": expected_request,
        "repeat_acquisitions_retained": len(request) - expected_request,
        "pf_series_per_knee": 2,
        "synovial_knees": syn_knees,
        "source_sha256": {moaks_path.name: sha256(moaks_path), mri_path.name: sha256(mri_path)},
        "output_sha256": {
            targets_path.name: sha256(targets_path),
            request_path.name: sha256(request_path),
            barcodes_path.name: sha256(barcodes_path),
        },
        "labels": {},
    }
    for label in ["PF OA", "Synovitis", "Effusion", "Baker's"]:
        valid = targets[f"{label}_weight"] > 0
        summary["labels"][label] = {
            "usable": int(valid.sum()),
            "mean_soft_target": float(targets.loc[valid, label].mean()),
            "reader_conflicts": int(targets.loc[valid, f"{label}_reader_conflict"].sum()),
        }
    (args.output / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
