"""Offline integrity and annotation checks; never treats AI answers as ground truth."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labels.evaluation_sampling import digest
from src.labels.openai_extract import FIELDS, validate
from src.validation.proxy import LABELS, sha


def validate_packet(packet):
    bare = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if digest(json.dumps(bare, sort_keys=True, ensure_ascii=False)) != packet["packet_sha256"]:
        raise ValueError("Packet hash changed")
    if packet["review_mode"] != "AI-assisted only" or packet["human_reference_available"]:
        raise ValueError("Incorrect reference provenance")
    if packet["findings"] != LABELS or digest(packet["rubric"]) != packet["rubric_sha256"]:
        raise ValueError("Rubric or findings changed")
    ids = set()
    for case in packet["cases"]:
        if set(case) != {"case_id", "report", "report_sha256"}:
            raise ValueError("Blind case contains unexpected metadata")
        if case["case_id"] in ids or digest(case["report"]) != case["report_sha256"]:
            raise ValueError("Duplicate case or changed report")
        ids.add(case["case_id"])


def annotation_status(packet, annotations):
    validate_packet(packet)
    if (annotations["packet_id"] != packet["packet_id"] or
            annotations["packet_sha256"] != packet["packet_sha256"] or
            annotations["review_mode"] != "AI-assisted only"):
        raise ValueError("Annotations belong to a different packet or provenance")
    expected = {(c["case_id"], finding): c for c in packet["cases"] for finding in LABELS}
    received = {}
    for row in annotations["rows"]:
        key = (row["case_id"], row["finding"])
        if key not in expected or key in received or row["report_sha256"] != expected[key]["report_sha256"]:
            raise ValueError("Unknown, duplicated or changed annotation row")
        if not all(field in row and isinstance(row[field], str) for field in FIELDS):
            raise ValueError("Invalid annotation fields")
        received[key] = row
    if set(received) != set(expected):
        raise ValueError("Missing annotation rows; retain blanks instead of dropping abstentions")
    complete, errors = 0, []
    for case in packet["cases"]:
        values = {finding: {field: received[(case["case_id"], finding)][field] for field in FIELDS}
                  for finding in LABELS}
        try:
            validate(values, case["report"])
        except ValueError as exc:
            errors.append({"case_id": case["case_id"], "error": str(exc)})
        else:
            complete += 1
    named = isinstance(annotations.get("reviewer"), str) and bool(annotations["reviewer"].strip())
    return {"status": "complete_unverified_ai_annotations" if not errors and named else "draft",
            "reports": len(packet["cases"]), "structurally_complete_reports": complete,
            "incomplete_or_invalid_reports": len(errors), "reviewer_recorded": named,
            "errors": errors, "independent_accuracy_established": False}


def verify_bundle(bundle, root=ROOT):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for entry in manifest["inputs"].values():
        if sha(root / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen input changed: " + entry["path"])
    for path, expected in manifest["source_sha256"].items():
        if sha(root / path) != expected:
            raise ValueError("Frozen source changed: " + path)
    for path, expected in manifest["output_sha256"].items():
        if sha(bundle / path) != expected:
            raise ValueError("Frozen output changed: " + path)
    counts = {}
    for name in ("pilot", "fresh"):
        folder = bundle / "reviewer" / name
        packet = json.loads((folder / "packet.json").read_text())
        blank = json.loads((folder / "annotations.blank.json").read_text())
        state = annotation_status(packet, blank)
        if state["structurally_complete_reports"] or state["status"] != "draft":
            raise ValueError("Blank review packet contains annotations")
        counts[name] = state["reports"]
    return {"status": "verified", "reports": counts, "frozen_output_files": len(manifest["output_sha256"]),
            "manifest_sha256": sha(bundle / "manifest.json"), "api_calls_made": 0,
            "review_mode": "AI-assisted only", "independent_reference_available": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--bundle", type=Path, default=ROOT / "outputs/report_label_evaluation_v1")
    annotations = commands.add_parser("annotations")
    annotations.add_argument("--packet", type=Path, required=True)
    annotations.add_argument("--annotations", type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "verify":
        result = verify_bundle(**args)
    else:
        result = annotation_status(json.loads(args["packet"].read_text()), json.loads(args["annotations"].read_text()))
    print(json.dumps(result, indent=2))
