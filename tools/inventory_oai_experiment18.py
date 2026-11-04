#!/usr/bin/env python3
"""Inventory an NDA OAI package without copying images or exposing identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile


ASSESSMENT_HINTS = ("moaks", "kmrisemiquant", "kmri_fnih_sq", "bicL00".lower())
REQUIRED_MOAKS = {
    "m_syic": {"m_syic", "v00msyic"},
    "m_eff_wk": {"m_eff_wk", "v00meffwk"},
}
IDENTIFIER_ALIASES = {
    "subject": {"src_subject_id", "subject_id", "id", "oai_id"},
    "visit": {"visit", "visit_id", "eventname", "visit_code", "version"},
    "side": {"side", "knee_side", "laterality"},
}
IMAGE_HINTS = ("image03", "manifest", "dicom", "mri")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")


def delimited_header(path: Path):
    raw = path.read_bytes()[:1 << 20]
    text = raw.decode("utf-8-sig", errors="replace")
    sample = "\n".join(text.splitlines()[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ","
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    header = next((row for row in rows if len(row) > 2), [])
    return [normalise(value) for value in header], delimiter


def spreadsheet_header(path: Path):
    import pandas as pd
    frame = pd.read_excel(path, nrows=0)
    return [normalise(value) for value in frame.columns]


def parquet_header(path: Path):
    import pyarrow.parquet as pq
    return [normalise(value) for value in pq.read_schema(path).names]


def table_header(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        header, delimiter = delimited_header(path)
        return header, {"delimiter": delimiter}
    if suffix in {".xlsx", ".xls"}:
        return spreadsheet_header(path), {}
    if suffix == ".parquet":
        return parquet_header(path), {}
    return [], {}


def classify_table(path: Path, columns):
    lowered_name = path.name.lower()
    column_set = set(columns)
    patellar = sorted(name for name in column_set if (
        name == "m_cm_p" or name.startswith("m_cm_p") or
        name in {"v00mcmpm", "v00mcmpl", "v00mcmfma", "v00mcmfla"}
    ))
    identifiers = {
        role: sorted(column_set & aliases)
        for role, aliases in IDENTIFIER_ALIASES.items()
    }
    return {
        "assessment_name_hint": any(hint in lowered_name for hint in ASSESSMENT_HINTS),
        "image_name_hint": any(hint in lowered_name for hint in IMAGE_HINTS),
        "required_moaks": sorted(
            canonical for canonical, aliases in REQUIRED_MOAKS.items()
            if column_set & aliases
        ),
        "patellar_cartilage_fields": patellar,
        "identifier_fields": identifiers,
        "technical_fields": sorted(name for name in column_set
                                   if "technical" in name or name.endswith("_qc")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    extensions, tables, archives = {}, [], []
    total_bytes = 0
    dicom_files = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower() or "<none>"
        extensions[suffix] = extensions.get(suffix, 0) + 1
        total_bytes += path.stat().st_size
        if suffix in {".dcm", ".dicom"}:
            dicom_files += 1
        if suffix in {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}:
            try:
                columns, parsing = table_header(path)
                if columns:
                    tables.append({
                        "path": str(path.relative_to(root)),
                        "size": path.stat().st_size,
                        "sha256": sha256(path),
                        "columns": columns,
                        **parsing,
                        **classify_table(path, columns),
                    })
            except Exception as error:
                tables.append({"path": str(path.relative_to(root)),
                               "error": f"{type(error).__name__}: {error}"})
        if suffix == ".zip":
            record = {"path": str(path.relative_to(root)), "size": path.stat().st_size}
            try:
                with zipfile.ZipFile(path) as bundle:
                    names = bundle.namelist()
                    dcm = sum(Path(name).suffix.lower() in {".dcm", ".dicom"}
                              for name in names)
                    record.update(members=len(names), dicom_members=dcm,
                                  name_hints=sorted({hint for hint in IMAGE_HINTS
                                                     if any(hint in n.lower() for n in names)}))
                    dicom_files += dcm
            except (zipfile.BadZipFile, OSError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            archives.append(record)

    candidates = [table for table in tables
                  if not table.get("error") and
                  (table.get("assessment_name_hint") or table.get("required_moaks")
                   or table.get("patellar_cartilage_fields"))]
    moaks_union = set().union(*(set(row.get("required_moaks", [])) for row in candidates))
    patellar_union = set().union(*(set(row.get("patellar_cartilage_fields", []))
                                   for row in candidates))
    identifier_coverage = {
        role: any(row.get("identifier_fields", {}).get(role) for row in tables
                  if not row.get("error"))
        for role in IDENTIFIER_ALIASES
    }
    checks = {
        "has_m_syic": "m_syic" in moaks_union,
        "has_m_eff_wk": "m_eff_wk" in moaks_union,
        "has_patellar_cartilage": bool(patellar_union),
        "has_subject_identifier": identifier_coverage["subject"],
        "has_visit_identifier": identifier_coverage["visit"],
        "has_side_identifier": identifier_coverage["side"],
        "has_imaging": dicom_files > 0,
    }
    result = {
        "status": "ready_for_target_build" if all(checks.values()) else "incomplete",
        "package_root_name": root.name,
        "files": sum(extensions.values()),
        "total_gib": total_bytes / 2**30,
        "extensions": dict(sorted(extensions.items())),
        "dicom_files_or_members": dicom_files,
        "checks": checks,
        "assessment_candidates": candidates,
        "archives": archives,
        "privacy": "No subject identifiers or table values are written to this inventory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("status", "files", "total_gib", "dicom_files_or_members", "checks")},
                     indent=2))


if __name__ == "__main__":
    main()
