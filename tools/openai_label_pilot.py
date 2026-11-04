"""Prepare or run a bounded report-extraction pilot; never edits training labels.

Preparation is offline. Execution requires OPENAI_API_KEY and an explicit USD
cap. Requests are journalled before transmission and are never retried silently.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.labels.openai_extract import request_body, request_hash, parse_response
from src.validation.proxy import ID, LABELS, sha, write_json_new

SOURCES = ["src/labels/openai_extract.py", "src/labels/llm_extract.py", "src/labels/ledger.py",
           "tools/openai_label_pilot.py", "docs/report_review_rubric.md"]


def write(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_inputs(manifest):
    for entry in manifest["inputs"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Pilot input bytes changed; prepare a new pilot")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Pilot source changed: {name}")
    reports = pd.read_csv(ROOT / manifest["inputs"]["reports"]["path"]).set_index(ID)
    rubric = (ROOT / "docs/report_review_rubric.md").read_text()
    return reports, rubric


def prepare(out, config, reports, roles, labels):
    out = Path(out)
    settings = json.loads(Path(config).read_text())
    if settings["model"] != "gpt-6-sol" or settings["reasoning_effort"] != "low":
        raise ValueError("This first pilot freezes GPT-6 Sol / low effort and its verified pricing")
    if not 1 <= settings["pilot_studies"] <= 200 or settings["max_output_tokens"] != 6000:
        raise ValueError("Pilot size/output limit differs from the bounded design")
    split = pd.read_csv(roles)
    if split[ID].duplicated().any() or not {ID, "role", "component"} <= set(split):
        raise ValueError("Need unique study roles and duplicate components")
    eligible = split[split.role == "train"].copy()
    forbidden = set(split.loc[split.role != "train", "component"])
    if set(eligible.component) & forbidden:
        raise ValueError("A training component overlaps another role")
    eligible["draw"] = [hashlib.sha256(f"{settings['seed']}:{uid}".encode()).hexdigest() for uid in eligible[ID]]
    sample = eligible.sort_values("draw").drop_duplicates("component").head(settings["pilot_studies"])
    if len(sample) != settings["pilot_studies"]:
        raise ValueError("Not enough independent training components")
    source_reports = pd.read_csv(reports).set_index(ID)
    old_labels = pd.read_parquet(labels).set_index(ID)
    if source_reports.index.has_duplicates or old_labels.index.has_duplicates:
        raise ValueError("Duplicate source studies")
    rubric = (ROOT / "docs/report_review_rubric.md").read_text()
    rows = []
    for uid, component in zip(sample[ID], sample.component):
        if source_reports.loc[uid, LABELS].notna().any():
            raise ValueError("Pilot selection includes a gold-labelled study")
        report = source_reports.loc[uid, "Report"]
        body = request_body(report, rubric, settings["model"], settings["max_output_tokens"])
        rows.append({ID: uid, "component": component, "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
                     "request_sha256": request_hash(body)})
    if len({row["request_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate report requests remain across selected components")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "scope": "bounded operational pilot; no accuracy gain established",
        "config": settings, "study_count": len(rows), "studies": rows,
        "inputs": {name: {"path": str(Path(path).resolve().relative_to(ROOT)), "sha256": sha(path)}
                   for name, path in [("reports", reports), ("roles", roles), ("labels", labels)]},
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "sampling": "seeded random training studies; one representative per duplicate component",
        "evaluation_limit": "Not stratified or powered for rare findings; fresh blind review is still required",
        "api_calls_made": 0, "production_labels_changed": False}
    old_labels.loc[sample[ID], LABELS + [f"{label}_weight" for label in LABELS]].to_parquet(out / "baseline_labels.parquet")
    # Keep the reviewer file separate from all model outputs; no baseline scores included.
    with (out / "blind_review.jsonl").open("x") as handle:
        for row in rows:
            for finding in LABELS:
                handle.write(json.dumps({ID: row[ID], "finding": finding, "report_sha256": row["report_sha256"],
                    "report_states": "", "evidence_quote": "", "reviewer_id": ""}) + "\n")
    manifest["baseline_labels_sha256"] = sha(out / "baseline_labels.parquet")
    write_json_new(out / "pilot.json", manifest)
    return {"prepared_studies": len(rows), "api_calls_made": 0, "production_labels_changed": False}


def reserve_cost(body, settings):
    # Conservative UTF-8 byte count plus framing/schema margin; no tool charges.
    upper_input_tokens = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    if upper_input_tokens > 200000:
        raise ValueError("Request exceeds the pilot's short-context pricing bound")
    return (upper_input_tokens * settings["input_or_cache_write_usd_per_million"] +
            body["max_output_tokens"] * settings["output_usd_per_million"]) / 1e6


def usage_cost(response, settings):
    usage = response["usage"]
    values = [usage["input_tokens"], usage["output_tokens"]]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("Missing/invalid API token usage")
    # Charge the full output count, including reasoning; take no cache discounts.
    return (values[0] * settings["input_or_cache_write_usd_per_million"] +
            values[1] * settings["output_usd_per_million"]) / 1e6


def post(body, api_key):
    request = urllib.request.Request("https://api.openai.com/v1/responses",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI HTTP {exc.code}; no automatic retry") from None
    except urllib.error.URLError:
        raise RuntimeError("OpenAI transport failure; reconcile reserved cost before retrying") from None


def run(pilot, max_cost_usd, send=post):
    import fcntl
    pilot = Path(pilot)
    if not 0 < max_cost_usd <= 100:
        raise ValueError("Supply an explicit pilot cap between 0 and 100 USD")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Configure OPENAI_API_KEY locally; do not paste credentials into chat")
    lock = (pilot / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((pilot / "pilot.json").read_text())
    reports, rubric = load_inputs(manifest)
    settings = manifest["config"]
    attempts = pilot / "attempts"
    attempts.mkdir(exist_ok=True)
    prior = [json.loads(p.read_text()) for p in attempts.glob("*.json")]
    if any(p["status"] != "accepted" for p in prior):
        raise RuntimeError("An earlier request needs review; no automatic paid retry")
    permitted = {row["request_sha256"]: row for row in manifest["studies"]}
    for record in prior:
        row = permitted.get(record["request_sha256"])
        if (row is None or record["pilot_sha256"] != sha(pilot / "pilot.json") or
                record[ID] != row[ID] or record["report_sha256"] != row["report_sha256"]):
            raise ValueError("Cached request belongs to another pilot/report")
        if record["charged_usd_upper"] != usage_cost(record["response"], settings):
            raise ValueError("Cached token accounting changed")
        if record["assertions"] != parse_response(record["response"], reports.loc[row[ID], "Report"]):
            raise ValueError("Cached accepted assertions changed")
    spent = sum(p["charged_usd_upper"] for p in prior)
    for row in manifest["studies"]:
        report = reports.loc[row[ID], "Report"]
        body = request_body(report, rubric, settings["model"], settings["max_output_tokens"])
        digest = request_hash(body)
        if digest != row["request_sha256"]:
            raise ValueError("Request differs from the prepared pilot")
        path = attempts / (digest + ".json")
        if path.exists():
            continue
        reserved = reserve_cost(body, settings)
        if spent + reserved > max_cost_usd:
            return {"status": "budget_limit", "accepted": len(list(attempts.glob('*.json'))),
                    "charged_usd_upper": spent, "next_request_reservation_usd": reserved}
        record = {ID: row[ID], "request_sha256": digest, "report_sha256": row["report_sha256"],
                  "status": "pending", "charged_usd_upper": reserved, "reserved_usd": reserved,
                  "pilot_sha256": sha(pilot / "pilot.json")}
        write_json_new(path, record)
        try:
            response = send(body, key)
            # Persist charged usage even if semantic validation subsequently fails.
            record.update(response=response, charged_usd_upper=usage_cost(response, settings))
            if record["charged_usd_upper"] > reserved:
                raise RuntimeError("Observed usage exceeds reserved cost; pricing/usage review required")
            record["assertions"] = parse_response(response, report)
            record["status"] = "accepted"
        except Exception as exc:
            record.update(status="needs_review", error_type=type(exc).__name__)
            write(path, record)
            raise
        write(path, record)
        spent += record["charged_usd_upper"]
        print(json.dumps({"accepted": len(list(attempts.glob('*.json'))), "charged_usd_upper": spent}), flush=True)
    return {"status": "extraction_completed", "accepted": len(manifest["studies"]),
            "charged_usd_upper": spent, "accuracy_gain_established": False, "production_labels_changed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--config", type=Path, default=ROOT / "configs/openai_label_pilot.json")
    prep.add_argument("--reports", type=Path, default=ROOT / "data/meta/train.csv")
    prep.add_argument("--roles", type=Path, default=ROOT / "outputs/foundation_finetune_outer0_v1/baseline/roles.csv")
    prep.add_argument("--labels", type=Path, default=ROOT / "data/meta/train_labels.parquet")
    execute = sub.add_parser("run")
    execute.add_argument("--pilot", type=Path, required=True)
    execute.add_argument("--max-cost-usd", type=float, required=True)
    args = vars(p.parse_args())
    action = args.pop("action")
    print(json.dumps(prepare(**args) if action == "prepare" else run(**args), indent=2))
