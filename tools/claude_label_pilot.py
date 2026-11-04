"""Run the approved pilot with Claude, preserving the original report selection.

Uses ANTHROPIC_API_KEY from the environment, project .env, or claude.env. Never logs keys,
executes .env as code, changes training labels, or silently retries paid requests.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.labels.claude_extract import MODEL, request_body, parse_response, normalization_audit
from src.labels.openai_extract import request_hash
from src.validation.proxy import ID, sha, write_json_new
from tools.openai_label_pilot import write

SOURCES = ["src/labels/claude_extract.py", "tools/claude_label_pilot.py"]


def local_setting(name):
    """Read one literal setting only; never source shell code or expand variables."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    values = []
    for path in [ROOT / ".env", ROOT / "claude.env"]:
        for line in path.read_text().splitlines() if path.is_file() else []:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.removeprefix("export ").partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values.append(value)
    if len(values) > 1:
        raise ValueError(f"Duplicate {name} entries in credential files")
    value = values[0] if values else ""
    if any(c.isspace() for c in value):
        raise ValueError(f"Invalid whitespace in {name}; use a single literal value")
    return value


def load_inputs(manifest):
    for entry in manifest["inputs"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Pilot input bytes changed; prepare a new pilot")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Pilot source changed: {name}")
    reports = pd.read_csv(ROOT / manifest["inputs"]["reports"]["path"]).set_index(ID)
    return reports, (ROOT / "docs/report_review_rubric.md").read_text()


def reserve_cost(body, settings):
    upper_input_tokens = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    if upper_input_tokens > 200000:
        raise ValueError("Request exceeds pilot context bound")
    return (upper_input_tokens * settings["input_reservation_usd_per_million"] +
            body["max_tokens"] * settings["output_usd_per_million"]) / 1e6


def usage_cost(response, settings):
    usage = response["usage"]
    names = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
    counts = [usage.get(k, 0) if k.startswith("cache_") else usage[k] for k in names]
    if any(type(v) is not int or v < 0 for v in counts):
        raise ValueError("Invalid Claude token accounting")
    input_tokens, output_tokens, writes, reads = counts
    # Count cache writes at the more expensive one-hour rate and take no read discount.
    return ((input_tokens + reads) * settings["input_usd_per_million"] +
            writes * settings["cache_write_upper_usd_per_million"] +
            output_tokens * settings["output_usd_per_million"]) / 1e6


def carried_cost(manifest):
    total = 0.0
    seen = set()
    for entry in manifest.get("prior_cost_records", []):
        if entry["path"] in seen:
            raise ValueError("Duplicate prior cost record")
        seen.add(entry["path"])
        path = ROOT / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Prior cost record changed")
        value = json.loads(path.read_text())["charged_usd_upper"]
        if type(value) not in {float, int} or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid prior cost")
        if value != entry["charged_usd_upper"]:
            raise ValueError("Prior cost accounting changed")
        total += value
    return total


def prepare(out, source_pilot, config, carry_from=None):
    source_pilot, out = Path(source_pilot), Path(out)
    if list((source_pilot / "attempts").glob("*.json")):
        raise ValueError("Original API pilot has attempts; reconcile the shared budget before switching")
    manifest = json.loads((source_pilot / "pilot.json").read_text())
    reports, rubric = load_inputs(manifest)
    if sha(source_pilot / "baseline_labels.parquet") != manifest["baseline_labels_sha256"]:
        raise ValueError("Original baseline snapshot changed")
    settings = json.loads(Path(config).read_text())
    if (settings["model"] != MODEL or settings["reasoning_effort"] != "low" or
            settings["thinking"] != "adaptive" or settings["max_output_tokens"] != 6000 or
            settings["max_cost_usd"] != 10.0):
        raise ValueError("Settings differ from the approved Claude pilot")
    manifest = copy.deepcopy(manifest)
    manifest.update(provider="anthropic", config=settings,
                    parent_pilot={"path": str((source_pilot / "pilot.json").resolve().relative_to(ROOT)),
                                  "sha256": sha(source_pilot / "pilot.json")})
    manifest["source_sha256"].update({name: sha(ROOT / name) for name in SOURCES})
    manifest["prior_cost_records"] = [
        {"path": str(Path(path).resolve().relative_to(ROOT)), "sha256": sha(path),
         "charged_usd_upper": json.loads(Path(path).read_text())["charged_usd_upper"]}
        for path in (carry_from or [])]
    prior_cost = carried_cost(manifest)
    if prior_cost >= settings["max_cost_usd"]:
        raise ValueError("Earlier attempts have exhausted the shared cap")
    reservations = []
    for row in manifest["studies"]:
        body = request_body(reports.loc[row[ID], "Report"], rubric)
        row["request_sha256"] = request_hash(body)
        reservations.append(reserve_cost(body, settings))
    out.mkdir(parents=True, exist_ok=False)
    for name in ["baseline_labels.parquet", "blind_review.jsonl"]:
        shutil.copyfile(source_pilot / name, out / name)
    write_json_new(out / "pilot.json", manifest)
    costs = {"studies": len(reservations), "calls_made": 0,
             "all_requests_maximum_reservations_usd": sum(reservations),
             "earlier_requests_reserved_usd": prior_cost,
             "max_cost_usd": settings["max_cost_usd"],
             "note": "Conservative reservation, not measured spend; provider invoice is authoritative"}
    write_json_new(out / "cost_plan.json", costs)
    return costs


def post(body, api_key):
    headers = {"content-type": "application/json", "x-api-key": api_key,
               "anthropic-version": "2023-06-01"}
    workspace = local_setting("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    request = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body, ensure_ascii=False).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
            result["_request_id"] = response.headers.get("request-id")
            return result
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Anthropic HTTP {exc.code}; no automatic retry") from None
    except urllib.error.URLError:
        raise RuntimeError("Anthropic transport failure; reconcile reserved cost before retrying") from None


def check_access():
    """List accessible models without sending reports or making inference calls."""
    key = local_setting("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured locally")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    workspace = local_setting("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    request = urllib.request.Request("https://api.anthropic.com/v1/models?limit=100", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        ids = [model["id"] for model in result.get("data", [])]
        return {"model_access_verified": MODEL in ids, "requested_model": MODEL,
                "available_models": ids, "has_more": result.get("has_more", False),
                "report_inference_calls": 0}
    except urllib.error.HTTPError as exc:
        error = json.load(exc).get("error", {})
        message = str(error.get("message", "")).replace(key, "[redacted]")
        result = {"model_access_verified": False, "http_status": exc.code,
                "error_type": error.get("type"), "message": message[:500], "report_inference_calls": 0}
        if exc.code == 400 and "workspace" in message and not workspace:
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/organizations/workspaces?limit=100&include_archived=false",
                headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    available = json.load(response)
                result["workspaces"] = [{"id": item["id"], "name": item["name"]}
                                        for item in available.get("data", [])]
                result["workspaces_has_more"] = available.get("has_more", False)
            except urllib.error.HTTPError as workspace_error:
                result["workspace_lookup_http_status"] = workspace_error.code
        return result


def run(pilot, max_cost_usd=10.0, send=post):
    pilot = Path(pilot)
    manifest = json.loads((pilot / "pilot.json").read_text())
    if not 0 < max_cost_usd <= min(10.0, manifest["config"]["max_cost_usd"]):
        raise ValueError("Cap must be positive and cannot exceed the approved $10")
    key = local_setting("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Save ANTHROPIC_API_KEY in .env, claude.env, or the environment; never paste it in chat")
    with (pilot / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reports, rubric = load_inputs(manifest)
        parent = ROOT / manifest["parent_pilot"]["path"]
        if sha(parent) != manifest["parent_pilot"]["sha256"]:
            raise ValueError("Original pilot manifest changed")
        if list((parent.parent / "attempts").glob("*.json")):
            raise ValueError("Original API pilot has attempts; reconcile the shared budget")
        if sha(pilot / "baseline_labels.parquet") != manifest["baseline_labels_sha256"]:
            raise ValueError("Baseline snapshot changed")
        settings = manifest["config"]
        attempts = pilot / "attempts"
        attempts.mkdir(exist_ok=True)
        prior = [json.loads(p.read_text()) for p in attempts.glob("*.json")]
        if any(p["status"] not in {"accepted", "rejected"} for p in prior):
            raise RuntimeError("An earlier request needs review; no automatic paid retry")
        permitted = {row["request_sha256"]: row for row in manifest["studies"]}
        for record in prior:
            row = permitted.get(record["request_sha256"])
            if (row is None or record["pilot_sha256"] != sha(pilot / "pilot.json") or
                    record[ID] != row[ID] or record["report_sha256"] != row["report_sha256"]):
                raise ValueError("Cached request belongs to another pilot/report")
            if record["charged_usd_upper"] != usage_cost(record["response"], settings):
                raise ValueError("Cached token accounting changed")
            try:
                parsed = parse_response(record["response"], reports.loc[row[ID], "Report"])
            except ValueError as exc:
                if record["status"] != "rejected" or record.get("validation_error") != str(exc):
                    raise ValueError("Cached rejection changed") from None
                if record.get("assertions"):
                    raise ValueError("Rejected response contains accepted assertions")
            else:
                if record["status"] != "accepted" or record["assertions"] != parsed:
                    raise ValueError("Cached assertions changed")
                if record.get("evidence_normalizations", []) != normalization_audit(record["response"], parsed):
                    raise ValueError("Cached quote-normalization audit changed")
        spent = carried_cost(manifest) + sum(p["charged_usd_upper"] for p in prior)
        counts = Counter(p["status"] for p in prior)
        for row in manifest["studies"]:
            report = reports.loc[row[ID], "Report"]
            body = request_body(report, rubric)
            digest = request_hash(body)
            if digest != row["request_sha256"]:
                raise ValueError("Request differs from the prepared pilot")
            path = attempts / (digest + ".json")
            if path.exists():
                continue
            reserved = reserve_cost(body, settings)
            if spent + reserved > max_cost_usd:
                return {"status": "budget_limit", "charged_usd_upper": spent}
            record = {ID: row[ID], "request_sha256": digest, "report_sha256": row["report_sha256"],
                      "status": "pending", "charged_usd_upper": reserved, "reserved_usd": reserved,
                      "pilot_sha256": sha(pilot / "pilot.json")}
            write_json_new(path, record)
            try:
                record["response"] = send(body, key)
                record["charged_usd_upper"] = usage_cost(record["response"], settings)
                if record["charged_usd_upper"] > reserved:
                    raise RuntimeError("Observed usage exceeds reserved cost; review required")
                try:
                    parsed = parse_response(record["response"], report)
                except ValueError as exc:
                    # Keep and count a paid invalid response, exclude its labels,
                    # and continue the operational sample without retrying it.
                    record.update(status="rejected", validation_error=str(exc))
                else:
                    record["assertions"] = parsed
                    record["evidence_normalizations"] = normalization_audit(record["response"], parsed)
                    record["status"] = "accepted"
            except Exception as exc:
                record.update(status="needs_review", error_type=type(exc).__name__)
                write(path, record)
                raise
            write(path, record)
            spent += record["charged_usd_upper"]
            counts[record["status"]] += 1
            print(json.dumps({"accepted": counts["accepted"], "rejected": counts["rejected"],
                              "charged_usd_upper": spent}), flush=True)
        return {"status": "pilot_completed_with_rejections" if counts["rejected"] else "extraction_completed",
                "accepted": counts["accepted"], "rejected": counts["rejected"],
                "reports_processed": sum(counts.values()),
                "charged_usd_upper": spent, "accuracy_gain_established": False,
                "production_labels_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--source-pilot", type=Path, default=ROOT / "outputs/openai_label_pilot_v1")
    prep.add_argument("--config", type=Path, default=ROOT / "configs/claude_label_pilot.json")
    prep.add_argument("--carry-from", type=Path, action="append", default=[])
    execute = commands.add_parser("run")
    execute.add_argument("--pilot", type=Path, default=ROOT / "outputs/claude_label_pilot_v4")
    execute.add_argument("--max-cost-usd", type=float, default=10.0)
    commands.add_parser("check-access")
    args = vars(parser.parse_args())
    action = args.pop("action")
    action_fn = {"prepare": prepare, "run": run, "check-access": check_access}[action]
    print(json.dumps(action_fn(**args), indent=2))
