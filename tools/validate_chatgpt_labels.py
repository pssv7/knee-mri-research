"""Validate a manually returned ChatGPT batch without changing training labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.proxy import sha, write_json_new
from tools.validate_report_evaluation import annotation_status


def check(batch, annotations, out):
    batch, annotations, out = Path(batch), Path(annotations), Path(out)
    if out.exists():
        raise FileExistsError("Keep the previous validation receipt")
    manifest = json.loads((batch.parent / "manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        if sha(batch.parent / name) != expected:
            raise ValueError("Frozen packet or private mapping changed: " + name)
    packet = json.loads((batch / "packet.json").read_text())
    result = json.loads(annotations.read_text())
    status = annotation_status(packet, result)
    conflicts = [{"case_id": r["case_id"], "finding": r["finding"],
                  "error": "present/negated requires definite certainty"}
                 for r in result["rows"]
                 if r["report_states"] in {"present", "negated"} and r["certainty"] != "definite"]
    complete = not status["errors"] and status["reviewer_recorded"] and not conflicts
    receipt = {**status,
               "status": "validated_ai_annotations_not_ground_truth" if complete else "needs_correction",
               "state_certainty_conflicts": conflicts,
               "finding_rows": len(result["rows"]),
               "packet_file_sha256": sha(batch / "packet.json"),
               "response_file_sha256": sha(annotations),
               "validator_sha256": sha(__file__),
               "production_labels_changed": False, "api_calls_made": 0,
               "semantic_correctness_established": False}
    write_json_new(out, receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(check(**vars(p.parse_args())), indent=2))
