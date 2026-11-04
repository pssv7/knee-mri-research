"""Prepare and execute blind Opus review; compare only after all requests finish.

Frozen Sonnet code and outputs remain untouched. Every API request is built only
from one original report and the existing rubric. Other annotations are used
locally for cost accounting and, after review completion, comparison.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels.opus_review import MODEL, request_body, parse_response
from src.labels.claude_extract import normalization_audit
from src.labels.openai_extract import request_hash
from src.validation.proxy import ID, sha, write_json_new
from tools import claude_label_pilot as base
from tools.openai_label_pilot import write

SOURCES = ["src/labels/opus_review.py", "tools/opus_blind_review.py"]


def validate_record(record, row, manifest_hash, report, settings, parser):
    if (record["request_sha256"] != row["request_sha256"] or record[ID] != row[ID] or
            record["report_sha256"] != row["report_sha256"] or record["pilot_sha256"] != manifest_hash):
        raise ValueError("Response provenance differs from the frozen request")
    if record["status"] not in {"accepted", "rejected"}:
        raise RuntimeError("An earlier request needs review; no automatic paid retry")
    if record["charged_usd_upper"] != base.usage_cost(record["response"], settings):
        raise ValueError("Cached usage accounting changed")
    try:
        parsed = parser(record["response"], report)
    except ValueError as exc:
        if record["status"] != "rejected" or record.get("validation_error") != str(exc) or record.get("assertions"):
            raise ValueError("Cached rejection changed") from None
    else:
        if record["status"] != "accepted" or record["assertions"] != parsed:
            raise ValueError("Cached assertions changed")
        if record.get("evidence_normalizations", []) != normalization_audit(record["response"], parsed):
            raise ValueError("Cached normalization audit changed")


def prepare(out, sonnet_pilot, config):
    out, sonnet_pilot = Path(out), Path(sonnet_pilot)
    parent_path = sonnet_pilot / "pilot.json"
    parent = json.loads(parent_path.read_text())
    reports, rubric = base.load_inputs(parent)
    parent_files = list((sonnet_pilot / "attempts").glob("*.json"))
    by_hash = {row["request_sha256"]: row for row in parent["studies"]}
    records = {}
    for path in parent_files:
        record = json.loads(path.read_text())
        row = by_hash[record["request_sha256"]]
        validate_record(record, row, sha(parent_path), reports.loc[row[ID], "Report"], parent["config"], base.parse_response)
        if record["request_sha256"] in records:
            raise ValueError("Duplicate Sonnet response")
        records[record["request_sha256"]] = record
    if set(records) != set(by_hash):
        raise ValueError("Sonnet pilot must be complete before freezing blind review")
    settings = json.loads(Path(config).read_text())
    if (settings["model"] != MODEL or settings["reasoning_effort"] != "medium" or
            settings["max_output_tokens"] != 6000 or settings["max_cost_usd"] != 10):
        raise ValueError("Settings differ from the approved Opus review")
    manifest = copy.deepcopy(parent)
    manifest.update(config=settings, review_type="LLM blind annotation; not an independent human reference",
                    sonnet_pilot={"path": str(parent_path.resolve().relative_to(ROOT)), "sha256": sha(parent_path)},
                    blinding={"sent": ["one original report", "fixed research rubric and output schema"],
                              "withheld": ["Sonnet responses", "previous extractor calls", "soft targets", "gold labels",
                                           "study IDs", "disagreement categories", "previous validation outcomes"],
                              "comparison_unsealed_only_after_all_reviews": True})
    for key in ["continuation_from", "validation_failure_policy", "postprocessing_revision"]:
        manifest.pop(key, None)
    manifest["source_sha256"].update({name: sha(ROOT / name) for name in SOURCES})
    manifest["prior_cost_records"] = copy.deepcopy(parent.get("prior_cost_records", [])) + [
        {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha(path),
         "charged_usd_upper": json.loads(path.read_text())["charged_usd_upper"]} for path in parent_files]
    initial_cost = base.carried_cost(manifest)
    reservations = []
    for row in manifest["studies"]:
        body = request_body(reports.loc[row[ID], "Report"], rubric)
        row["request_sha256"] = request_hash(body)
        reservations.append(base.reserve_cost(body, settings))
    if initial_cost + min(reservations) > settings["max_cost_usd"]:
        raise ValueError("Not enough of the approved shared budget for one review")
    out.mkdir(parents=True, exist_ok=False)
    write_json_new(out / "pilot.json", manifest)
    plan = {"reports": manifest["study_count"], "prior_accounted_cost_usd": initial_cost,
            "shared_cap_usd": 10.0, "all_request_maximum_reservations_usd": sum(reservations),
            "note": "The sum of maximum reservations is not expected spend. Each request is gated against actual accounted cost; the run stops if the next reservation will not fit.",
            "api_calls_made": 0, "annotations_sent_to_reviewer": False}
    write_json_new(out / "cost_plan.json", plan)
    return plan


def load_review(pilot):
    pilot = Path(pilot)
    manifest = json.loads((pilot / "pilot.json").read_text())
    reports, rubric = base.load_inputs(manifest)
    parent = ROOT / manifest["sonnet_pilot"]["path"]
    if sha(parent) != manifest["sonnet_pilot"]["sha256"]:
        raise ValueError("Sonnet manifest changed")
    records = {}
    by_hash = {row["request_sha256"]: row for row in manifest["studies"]}
    for path in (pilot / "attempts").glob("*.json"):
        record = json.loads(path.read_text())
        row = by_hash.get(record["request_sha256"])
        if row is None or record["request_sha256"] in records:
            raise ValueError("Unknown or duplicate review record")
        validate_record(record, row, sha(pilot / "pilot.json"), reports.loc[row[ID], "Report"], manifest["config"], parse_response)
        records[record["request_sha256"]] = record
    return manifest, reports, rubric, records


def run(pilot, max_cost_usd=10.0, send=base.post):
    pilot = Path(pilot)
    if not 0 < max_cost_usd <= 10.0:
        raise ValueError("Cannot exceed the approved shared $10 cap")
    key = base.local_setting("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured locally")
    with (pilot / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, reports, rubric, records = load_review(pilot)
        settings = manifest["config"]
        spent = base.carried_cost(manifest) + sum(r["charged_usd_upper"] for r in records.values())
        counts = Counter(r["status"] for r in records.values())
        attempts = pilot / "attempts"
        attempts.mkdir(exist_ok=True)
        for row in manifest["studies"]:
            report = reports.loc[row[ID], "Report"]
            body = request_body(report, rubric)
            digest = request_hash(body)
            if digest != row["request_sha256"]:
                raise ValueError("Review request differs from its pre-review freeze")
            if digest in records:
                continue
            reserved = base.reserve_cost(body, settings)
            if spent + reserved > max_cost_usd:
                return {"status": "shared_budget_limit", "reports_reviewed": sum(counts.values()),
                        "total_accounted_cost_usd_upper": spent}
            path = attempts / (digest + ".json")
            record = {ID: row[ID], "request_sha256": digest, "report_sha256": row["report_sha256"],
                      "status": "pending", "charged_usd_upper": reserved, "reserved_usd": reserved,
                      "pilot_sha256": sha(pilot / "pilot.json")}
            write_json_new(path, record)
            try:
                record["response"] = send(body, key)
                record["charged_usd_upper"] = base.usage_cost(record["response"], settings)
                if record["charged_usd_upper"] > reserved:
                    raise RuntimeError("Review usage exceeds reservation; cost reconciliation required")
                try:
                    parsed = parse_response(record["response"], report)
                except ValueError as exc:
                    record.update(status="rejected", validation_error=str(exc))
                else:
                    record.update(status="accepted", assertions=parsed,
                                  evidence_normalizations=normalization_audit(record["response"], parsed))
            except Exception as exc:
                record.update(status="needs_review", error_type=type(exc).__name__)
                write(path, record)
                raise
            write(path, record)
            counts[record["status"]] += 1
            spent += record["charged_usd_upper"]
            print(json.dumps({"accepted": counts["accepted"], "rejected": counts["rejected"],
                              "total_accounted_cost_usd_upper": spent}), flush=True)
        result = {"status": "blind_review_complete", "accepted": counts["accepted"], "rejected": counts["rejected"],
                  "reports_reviewed": sum(counts.values()), "total_accounted_cost_usd_upper": spent,
                  "human_accuracy_established": False, "production_labels_changed": False}
        write(pilot / "run_result.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, default=ROOT / "outputs/opus_blind_review_v1")
    prep.add_argument("--sonnet-pilot", type=Path, default=ROOT / "outputs/claude_label_pilot_v4")
    prep.add_argument("--config", type=Path, default=ROOT / "configs/opus_blind_review.json")
    execute = commands.add_parser("run")
    execute.add_argument("--pilot", type=Path, default=ROOT / "outputs/opus_blind_review_v1")
    execute.add_argument("--max-cost-usd", type=float, default=10.0)
    args = vars(parser.parse_args())
    action = args.pop("action")
    print(json.dumps(prepare(**args) if action == "prepare" else run(**args), indent=2))
