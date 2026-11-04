"""Unseal a completed Opus review and compare annotations offline; no API calls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.proxy import ID, LABELS, sha
from tools import opus_blind_review as review
from tools.openai_label_pilot import write


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def agreement(rows):
    comparable = [r for r in rows if r["sonnet"] is not None and r["opus"] is not None]
    equal = sum(r["sonnet"]["report_states"] == r["opus"]["report_states"] for r in comparable)
    present_union = [r for r in comparable if "present" in
                     {r["sonnet"]["report_states"], r["opus"]["report_states"]}]
    both_present = sum(r["sonnet"]["report_states"] == r["opus"]["report_states"] == "present"
                       for r in present_union)
    confusion = Counter((r["sonnet"]["report_states"], r["opus"]["report_states"]) for r in comparable)
    return {"finding_slots": len(rows), "comparable_slots": len(comparable),
            "unavailable_slots": len(rows) - len(comparable),
            "exact_state_agreements": equal, "state_disagreements": len(comparable) - equal,
            "exact_state_agreement_fraction": equal / len(comparable) if comparable else None,
            "either_model_present": len(present_union), "both_models_present": both_present,
            "both_present_over_either_present": both_present / len(present_union) if present_union else None,
            "state_confusion": [{"sonnet": s, "opus": o, "count": n}
                                for (s, o), n in sorted(confusion.items())],
            "certainty_disagreements": sum(r["sonnet"]["certainty"] != r["opus"]["certainty"] for r in comparable),
            "severity_disagreements_when_both_present": sum(
                r["sonnet"]["severity_norm"] != r["opus"]["severity_norm"] for r in present_union
                if r["sonnet"]["report_states"] == r["opus"]["report_states"] == "present")}


def compare(pilot):
    pilot = Path(pilot)
    manifest, reports, _, records = review.load_review(pilot)
    expected = {r["request_sha256"] for r in manifest["studies"]}
    if set(records) != expected:
        raise ValueError("Blind comparison is sealed until all review responses finish")
    # Cost hashes also bind the original Sonnet responses before unsealing.
    previous_cost = review.base.carried_cost(manifest)
    parent_path = review.ROOT / manifest["sonnet_pilot"]["path"]
    parent = json.loads(parent_path.read_text())
    parent_rows = {r[ID]: r for r in parent["studies"]}
    if set(parent_rows) != {r[ID] for r in manifest["studies"]}:
        raise ValueError("Review study cohort differs from Sonnet")
    sonnet = {}
    for path in (parent_path.parent / "attempts").glob("*.json"):
        record = json.loads(path.read_text())
        uid = record[ID]
        if uid not in parent_rows or uid in sonnet:
            raise ValueError("Unknown or duplicate Sonnet response")
        review.validate_record(record, parent_rows[uid], sha(parent_path), reports.loc[uid, "Report"],
                               parent["config"], review.base.parse_response)
        sonnet[uid] = record
    if set(sonnet) != set(parent_rows):
        raise ValueError("Incomplete Sonnet responses")
    rows, assertions, rejected, normalizations = [], [], [], []
    usage = Counter()
    statuses = Counter()
    unique_response_ids = set()
    for study in manifest["studies"]:
        uid = study[ID]
        o, s = records[study["request_sha256"]], sonnet[uid]
        response_id = o["response"]["id"]
        if response_id in unique_response_ids:
            raise ValueError("Duplicate provider response ID")
        unique_response_ids.add(response_id)
        statuses[o["status"]] += 1
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[key] += o["response"]["usage"].get(key, 0)
        if o["status"] == "rejected":
            rejected.append({ID: uid, "validation_error": o["validation_error"]})
        normalizations.extend({ID: uid, **n} for n in o.get("evidence_normalizations", []))
        for finding in LABELS:
            sa, oa = s.get("assertions", {}).get(finding), o.get("assertions", {}).get(finding)
            row = {ID: uid, "finding": finding, "sonnet_status": s["status"], "opus_status": o["status"],
                   "sonnet": sa, "opus": oa}
            rows.append(row)
            if oa is not None:
                assertions.append({ID: uid, "finding": finding, **oa})
    lookup = {(r[ID], r["finding"]): r for r in rows}
    candidates, diagnostic_counts = [], defaultdict(Counter)
    diagnostic_path = parent_path.parent / "baseline_diagnostic_cases.jsonl"
    if diagnostic_path.exists():
        for candidate in read_jsonl(diagnostic_path):
            row = lookup[(candidate[ID], candidate["finding"])]
            if row["sonnet"] is None or row["sonnet"]["report_states"] != "present":
                raise ValueError("Baseline diagnostic is inconsistent with Sonnet annotations")
            outcome = row["opus"]["report_states"] if row["opus"] is not None else "unavailable"
            diagnostic_counts[candidate["diagnostic_group"]][outcome] += 1
            candidates.append({**candidate, "opus": row["opus"], "opus_review_state": outcome})
    queue = []
    for row in rows:
        s, o = row["sonnet"], row["opus"]
        reasons = []
        if s is None or o is None:
            reasons.append("one_or_both_reports_rejected")
        else:
            if s["report_states"] != o["report_states"]:
                reasons.append("state_disagreement")
            if s["certainty"] != o["certainty"]:
                reasons.append("certainty_disagreement")
            if s["report_states"] == o["report_states"] == "present" and s["severity_norm"] != o["severity_norm"]:
                reasons.append("severity_disagreement_when_both_present")
        if reasons:
            queue.append({**row, "review_reasons": reasons})
    cost = sum(r["charged_usd_upper"] for r in records.values())
    summary = {
        "status": "complete_unsealed_offline", "reviewer_model": manifest["config"]["model"],
        "reviewer_effort": manifest["config"]["reasoning_effort"], "reports_reviewed": len(records),
        "reports_accepted": statuses["accepted"], "reports_rejected": statuses["rejected"],
        "distinct_api_responses": len(unique_response_ids), "accepted_finding_assertions": len(assertions),
        "quotes_with_whitespace_restored": len(normalizations), "usage": dict(usage),
        "opus_response_token_cost_usd_upper": round(cost, 8),
        "prior_accounted_cost_usd_upper": round(previous_cost, 8),
        "shared_total_accounted_cost_usd_upper": round(cost + previous_cost, 8),
        "shared_budget_cap_usd": manifest["config"]["max_cost_usd"],
        "agreement": agreement(rows),
        "per_finding": {finding: agreement([r for r in rows if r["finding"] == finding]) for finding in LABELS},
        "opus_state_counts": dict(Counter(a["report_states"] for a in assertions)),
        "old_extractor_diagnostic_groups": {group: dict(counts) for group, counts in diagnostic_counts.items()},
        "review_queue_rows": len(queue), "review_queue_studies": len({r[ID] for r in queue}),
        "human_reference_accuracy_established": False, "image_model_accuracy_gain_established": False,
        "production_labels_changed": False,
        "limitations": ["Model agreement is not correctness; same provider and shared rubric can cause correlated errors.",
                        "The 40-report random training sample has no independent reference labels and is not powered for rare findings.",
                        "Rejected responses are unavailable, not agreements or negative labels.",
                        "Low soft targets may encode severity rather than a missed positive call."],
        "provenance": {"pilot_sha256": sha(pilot / "pilot.json"), "sonnet_pilot_sha256": sha(parent_path),
                       "comparison_source_sha256": sha(Path(__file__)),
                       "response_record_sha256": {p.name: sha(p) for p in sorted((pilot / "attempts").glob("*.json"))},
                       "baseline_diagnostic_sha256": sha(diagnostic_path) if diagnostic_path.exists() else None}}
    if cost + previous_cost > manifest["config"]["max_cost_usd"]:
        raise ValueError("Shared budget exceeded")
    for name, values in [("assertions.jsonl", assertions), ("all_comparisons.jsonl", rows),
                         ("adjudication_queue.jsonl", queue), ("baseline_candidate_review.jsonl", candidates),
                         ("rejected_reports.jsonl", rejected), ("evidence_normalizations.jsonl", normalizations)]:
        (pilot / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
    write(pilot / "comparison_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "outputs/opus_blind_review_v1")
    result = compare(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key not in {"provenance", "per_finding"}}, indent=2))
