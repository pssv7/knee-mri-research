"""Offline audit and review exports for a completed Claude pilot; no API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.labels.claude_extract import parse_response, normalization_audit
from src.validation.proxy import ID, LABELS, sha
from tools.claude_label_pilot import load_inputs, carried_cost, usage_cost
from tools.openai_label_pilot import write


def summarize(pilot):
    pilot = Path(pilot)
    manifest = json.loads((pilot / "pilot.json").read_text())
    reports, _ = load_inputs(manifest)
    baseline = pd.read_parquet(pilot / "baseline_labels.parquet")
    assert sha(pilot / "baseline_labels.parquet") == manifest["baseline_labels_sha256"]
    files = list((pilot / "attempts").glob("*.json"))
    assert len(files) == manifest["study_count"], "Pilot is not complete"
    records = {json.loads(path.read_text())["request_sha256"]: json.loads(path.read_text()) for path in files}
    assert len(records) == manifest["study_count"], "Duplicate request records"
    assertions, comparisons, review_reports, normalizations, rejected = [], [], [], [], []
    input_tokens = output_tokens = cache_write_tokens = cache_read_tokens = 0
    successful_cost = 0.0
    for study in manifest["studies"]:
        uid = study[ID]
        record = records[study["request_sha256"]]
        assert record["status"] in {"accepted", "rejected"} and record[ID] == uid
        assert record["pilot_sha256"] == sha(pilot / "pilot.json")
        assert record["report_sha256"] == study["report_sha256"]
        cost = usage_cost(record["response"], manifest["config"])
        assert record["charged_usd_upper"] == cost
        successful_cost += cost
        usage = record["response"]["usage"]
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        cache_write_tokens += usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        review_reports.append({ID: uid, "Report": reports.loc[uid, "Report"],
                               "report_sha256": study["report_sha256"]})
        try:
            extracted = parse_response(record["response"], reports.loc[uid, "Report"])
        except ValueError as exc:
            assert record["status"] == "rejected" and record["validation_error"] == str(exc)
            assert not record.get("assertions")
            rejected.append({ID: uid, "validation_error": str(exc),
                             "attempt_file": str(pilot / "attempts" / (study["request_sha256"] + ".json"))})
            continue
        assert record["status"] == "accepted" and record["assertions"] == extracted
        changes = normalization_audit(record["response"], extracted)
        assert record.get("evidence_normalizations", []) == changes
        normalizations.extend({ID: uid, **change} for change in changes)
        for finding in LABELS:
            assertion = {ID: uid, "finding": finding, **extracted[finding]}
            assertions.append(assertion)
            target = float(baseline.loc[uid, finding])
            state = assertion["report_states"]
            reason = ""
            if state == "present" and target < .5:
                reason = "Claude present; previous fused target below 0.5"
            elif target >= .5 and state in {"negated", "unmentioned", "present_below_threshold"}:
                reason = "Previous fused target at least 0.5; Claude negated, unmentioned, or below threshold"
            elif state in {"uncertain", "unattributable"}:
                reason = "Claude ambiguity requires review"
            if reason:
                comparisons.append({**assertion, "baseline_fused_target": target,
                                    "baseline_weight": float(baseline.loc[uid, finding + "_weight"]),
                                    "review_reason": reason})
    assert len(assertions) == (manifest["study_count"] - len(rejected)) * len(LABELS)
    reserved = carried_cost(manifest)
    summary = {
        "model": manifest["config"]["model"], "studies_completed": manifest["study_count"],
        "finding_assertions": len(assertions), "responses_received": len(records),
        "reports_accepted": manifest["study_count"] - len(rejected), "reports_rejected": len(rejected),
        "findings_without_accepted_extraction": len(rejected) * len(LABELS),
        "quotes_with_whitespace_restored": len(normalizations),
        "earlier_rejected_requests": len(manifest.get("prior_cost_records", [])),
        "response_usage_cost_usd_upper": round(successful_cost, 8),
        "earlier_request_cost_reserved_usd": round(reserved, 8),
        "total_cost_accounted_usd_upper": round(successful_cost + reserved, 8),
        "budget_cap_usd": manifest["config"]["max_cost_usd"],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write_tokens, "cache_read_input_tokens": cache_read_tokens,
        "state_counts": dict(Counter(row["report_states"] for row in assertions)),
        "review_candidates": len(comparisons),
        "review_candidate_studies": len({row[ID] for row in comparisons}),
        "review_reason_counts": dict(Counter(row["review_reason"] for row in comparisons)),
        "per_finding_states": {name: dict(Counter(row["report_states"] for row in assertions if row["finding"] == name)) for name in LABELS},
        "accuracy_gain_established": False, "production_labels_changed": False,
        "cost_note": "API usage at frozen rates plus conservative reservations for rejected requests; not the provider invoice",
        "review_note": "Review candidates are disagreements or ambiguity, not verified errors. Baseline 0.5 is a screening threshold only. Complete blind review before opening model outputs."
    }
    assert summary["total_cost_accounted_usd_upper"] <= summary["budget_cap_usd"]
    for name, rows in [("assertions.jsonl", assertions), ("model_comparison.jsonl", comparisons),
                       ("blind_review_reports.jsonl", review_reports),
                       ("evidence_normalizations.jsonl", normalizations),
                       ("rejected_reports.jsonl", rejected)]:
        (pilot / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write(pilot / "result_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "outputs/claude_label_pilot_v4")
    print(json.dumps(summarize(**vars(parser.parse_args())), indent=2))
