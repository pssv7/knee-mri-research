"""Frozen fresh Sonnet extraction with one shared budget and no paid retries."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labels.report_candidate import request_body, parse_response
from src.labels.claude_extract import normalization_audit
from src.labels.openai_extract import request_hash
from src.validation.proxy import sha, write_json_new
from tools import claude_label_pilot as base
from tools.openai_label_pilot import write
from tools.validate_report_evaluation import verify_bundle, validate_packet

CLARIFICATIONS = "docs/report_extraction_clarifications_v1.md"
SOURCES = ["tools/fresh_report_extraction.py", "src/labels/report_candidate.py", CLARIFICATIONS,
           "src/labels/claude_extract.py", "src/labels/openai_extract.py", "tools/claude_label_pilot.py",
           "tools/validate_report_evaluation.py", "src/labels/ledger.py"]


def prepare(out, bundle):
    out, bundle = Path(out), Path(bundle)
    if out.exists():
        raise FileExistsError("Use a new version; preserve the frozen request cohort")
    verify_bundle(bundle)
    packet_path = bundle / "reviewer/fresh/packet.json"
    packet = json.loads(packet_path.read_text()); validate_packet(packet)
    settings = json.loads((ROOT / "configs/claude_label_pilot.json").read_text())
    opus = ROOT / "outputs/opus_blind_review_v1"
    parent = json.loads((opus / "pilot.json").read_text())
    costs = parent["prior_cost_records"] + [
        {"path": str(p.relative_to(ROOT)), "sha256": sha(p),
         "charged_usd_upper": json.loads(p.read_text())["charged_usd_upper"]}
        for p in sorted((opus / "attempts").glob("*.json"))]
    if len(list((opus / "attempts").glob("*.json"))) != parent["study_count"]:
        raise ValueError("Prior review has not completed")
    manifest = {"config": settings, "prior_cost_records": costs,
                "packet": {"path": str(packet_path.relative_to(ROOT)), "sha256": sha(packet_path)},
                "bundle": {"path": str(bundle.relative_to(ROOT)), "manifest_sha256": sha(bundle / "manifest.json")},
                "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
                "adjudication": {"path": "outputs/report_label_adjudication_v1/ledger.jsonl",
                                 "sha256": sha(ROOT / "outputs/report_label_adjudication_v1/ledger.jsonl")},
                "scope": "200 frozen fresh reports; training labels unchanged; AI-only evaluation",
                "ordering": "frozen blinded packet order, independent of outcomes", "cases": []}
    previous = base.carried_cost(manifest)
    checks = (ROOT / CLARIFICATIONS).read_text()
    for case in packet["cases"]:
        body = request_body(case["report"], packet["rubric"], checks)
        manifest["cases"].append({"case_id": case["case_id"], "report_sha256": case["report_sha256"],
                                  "request_sha256": request_hash(body), "reserved_usd": base.reserve_cost(body, settings)})
    if len({r["request_sha256"] for r in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("Duplicate fresh request")
    out.mkdir(parents=True)
    write_json_new(out / "manifest.json", manifest)
    plan = {"reports": len(manifest["cases"]), "prior_accounted_cost_usd_upper": previous,
            "shared_cap_usd": 10, "remaining_usd": 10 - previous,
            "estimated_new_cost_at_pilot_mean_usd": 1.091174 / 40 * len(manifest["cases"]),
            "estimate_note": "Pilot mean only; revised instructions and report mix can change cost. Each call requires a conservative reservation. Stop before the shared cap if needed.",
            "api_calls_made": 0}
    write_json_new(out / "cost_plan.json", plan)
    return plan


def load(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for key in ["packet", "adjudication"]:
        entry = manifest[key]
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen " + key + " changed")
    bundle = ROOT / manifest["bundle"]["path"]
    if sha(bundle / "manifest.json") != manifest["bundle"]["manifest_sha256"]:
        raise ValueError("Frozen cohort manifest changed")
    verify_bundle(bundle)
    packet = json.loads((ROOT / manifest["packet"]["path"]).read_text()); validate_packet(packet)
    by_id = {c["case_id"]: c for c in packet["cases"]}
    expected = {c["request_sha256"]: c for c in manifest["cases"]}
    records = {}
    for path in (out / "attempts").glob("*.json"):
        record = json.loads(path.read_text())
        digest = record["request_sha256"]
        row = expected.get(digest)
        if row is None or digest in records or any(record.get(k) != row[k] for k in ["case_id", "report_sha256"]):
            raise ValueError("Unknown, duplicate or mismatched cached request")
        if record["manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("Cached manifest differs")
        if record["status"] not in {"accepted", "rejected"}:
            raise RuntimeError("Prior request needs reconciliation; no automatic retry")
        if record["charged_usd_upper"] != base.usage_cost(record["response"], manifest["config"]):
            raise ValueError("Cached cost differs")
        try:
            parsed = parse_response(record["response"], by_id[row["case_id"]]["report"])
        except ValueError as exc:
            if record["status"] != "rejected" or record["validation_error"] != str(exc) or record.get("assertions"):
                raise ValueError("Cached rejection differs") from None
        else:
            if record["status"] != "accepted" or record["assertions"] != parsed:
                raise ValueError("Cached assertions differ")
            if record["evidence_normalizations"] != normalization_audit(record["response"], parsed):
                raise ValueError("Cached quote audit differs")
        records[digest] = record
    return manifest, packet, records


def run(out, max_cost_usd=10.0, send=base.post):
    out = Path(out)
    if not 0 < max_cost_usd <= 10:
        raise ValueError("Cannot exceed the existing shared $10 cap")
    key = base.local_setting("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Configure ANTHROPIC_API_KEY locally")
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, packet, records = load(out)
        spent = base.carried_cost(manifest) + sum(r["charged_usd_upper"] for r in records.values())
        counts = Counter(r["status"] for r in records.values())
        (out / "attempts").mkdir(exist_ok=True)
        by_id = {c["case_id"]: c for c in packet["cases"]}
        checks = (ROOT / CLARIFICATIONS).read_text()
        status = "complete"
        for row in manifest["cases"]:
            case = by_id[row["case_id"]]
            body = request_body(case["report"], packet["rubric"], checks)
            digest = request_hash(body)
            reserved = base.reserve_cost(body, manifest["config"])
            if digest != row["request_sha256"] or reserved != row["reserved_usd"]:
                raise ValueError("Frozen request or reservation differs")
            if digest in records:
                continue
            if spent + reserved > max_cost_usd:
                status = "shared_budget_limit"
                break
            path = out / "attempts" / (digest + ".json")
            record = {**row, "manifest_sha256": sha(out / "manifest.json"),
                      "status": "pending", "charged_usd_upper": reserved}
            write_json_new(path, record)
            try:
                record["response"] = send(body, key)
                record["charged_usd_upper"] = base.usage_cost(record["response"], manifest["config"])
                if record["charged_usd_upper"] > reserved:
                    raise RuntimeError("Usage exceeds reservation; reconciliation required")
                try:
                    record["assertions"] = parse_response(record["response"], case["report"])
                except ValueError as exc:
                    record.update(status="rejected", validation_error=str(exc))
                else:
                    record.update(status="accepted", evidence_normalizations=normalization_audit(record["response"], record["assertions"]))
            except Exception as exc:
                record.update(status="needs_review", error_type=type(exc).__name__)
                write(path, record)
                raise
            write(path, record)
            spent += record["charged_usd_upper"]
            counts[record["status"]] += 1
            print(json.dumps({"completed": sum(counts.values()), "accepted": counts["accepted"],
                              "rejected": counts["rejected"], "shared_accounted_usd": round(spent, 8)}), flush=True)
        result = {"status": status, "reports_planned": len(manifest["cases"]), "reports_completed": sum(counts.values()),
                  "accepted": counts["accepted"], "rejected": counts["rejected"],
                  "shared_accounted_usd_upper": round(spent, 8), "production_labels_changed": False,
                  "independent_accuracy_established": False}
        write(out / "run_result.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, default=ROOT / "outputs/fresh_report_extraction_v1")
    prep.add_argument("--bundle", type=Path, default=ROOT / "outputs/report_label_evaluation_v1")
    execute = commands.add_parser("run")
    execute.add_argument("--out", type=Path, default=ROOT / "outputs/fresh_report_extraction_v1")
    execute.add_argument("--max-cost-usd", type=float, default=10.0)
    args = vars(parser.parse_args()); action = args.pop("action")
    print(json.dumps((prepare if action == "prepare" else run)(**args), indent=2))
