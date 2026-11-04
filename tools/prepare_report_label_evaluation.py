"""Prepare offline AI-assisted report adjudication and fresh evaluation packets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.labels.evaluation_sampling import digest, eligible_representatives, ordered, select_sample
from src.labels.review_packet import blank_annotations, render
from src.validation.proxy import ID, LABELS, sha, write_json_new
from tools.audit_labels import detect_language
from tools.opus_blind_review import load_review

CORE = {
    "reports": "data/meta/train.csv",
    "roles": "outputs/foundation_finetune_outer0_v1/baseline/roles.csv",
    "components": "outputs/validation_fallback_v2/components.csv",
    "component_manifest": "outputs/validation_fallback_v2/manifest.json",
    "labels": "data/meta/train_labels.parquet",
    "categorical": "data/meta/llm_labels.parquet",
    "probability": "data/meta/llm_probs.parquet",
    "rubric": "docs/report_review_rubric.md",
    "opus_manifest": "outputs/opus_blind_review_v1/pilot.json",
    "opus_comparison": "outputs/opus_blind_review_v1/comparison_summary.json",
    "opus_rows": "outputs/opus_blind_review_v1/all_comparisons.jsonl",
    "opus_queue": "outputs/opus_blind_review_v1/adjudication_queue.jsonl",
    "opus_candidates": "outputs/opus_blind_review_v1/baseline_candidate_review.jsonl",
    "baseline_diagnostic": "outputs/claude_label_pilot_v4/baseline_diagnostic_summary.json",
}
SOURCES = ["tools/prepare_report_label_evaluation.py", "src/labels/evaluation_sampling.py",
           "src/labels/review_packet.py", "tools/report_review_template.html", "tools/audit_labels.py",
           "src/labels/openai_extract.py", "src/labels/ledger.py", "src/config.py",
           "tools/validate_report_evaluation.py"]


def jsonl(path, rows):
    with path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_ids(path):
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=[ID])
    elif path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)[[ID]]
    else:
        frame = pd.read_csv(path, usecols=[ID])
    if frame[ID].isna().any():
        raise ValueError(f"Missing exposure ID: {path}")
    return set(frame[ID])


def exposure_inventory(root):
    required = ["outputs/label_review_v1/review_queue.jsonl",
                "outputs/validation_reset_20260924/pilot_custodian_key.csv"]
    required += ["outputs/label_quality/audit_v1/" + name + ".parquet" for name in
                 ["reserve", "sample", "exposure", "extra_controls", "secondary_queue", "ledger_assembled"]]
    keys = sorted((root / "outputs/label_quality/review/keys").glob("*.parquet"))
    if not keys:
        raise ValueError("Prior reader-key inventory is missing")
    paths = [root / name for name in required] + keys
    if any(not p.is_file() for p in paths):
        raise ValueError("Required exposure/reserve file is missing")
    return {str(p.relative_to(root)): read_ids(p) for p in paths}


def make_packet(rows, reports, rubric, name, title, seed):
    sequence = ordered(rows, seed, "review_order:" + name, "component")
    cases, keys = [], []
    for position, row in enumerate(sequence.to_dict("records"), 1):
        uid = row[ID]
        case_id = ("P" if name == "pilot" else "E") + f"{position:03d}"
        report = reports.loc[uid, "Report"]
        if not isinstance(report, str) or not report.strip():
            raise ValueError("Empty source report")
        cases.append({"case_id": case_id, "report": report, "report_sha256": digest(report)})
        keys.append({"case_id": case_id, **row, "report_sha256": digest(report)})
    packet = {"schema_version": 1, "packet_id": "report_label_evaluation_v1_" + name,
              "title": title, "review_mode": "AI-assisted only", "human_reference_available": False,
              "rubric": rubric, "rubric_sha256": digest(rubric), "findings": LABELS, "cases": cases}
    packet["packet_sha256"] = digest(json.dumps(packet, sort_keys=True, ensure_ascii=False))
    return packet, keys


def write_packet(out, packet, template):
    out.mkdir(parents=True)
    write_json_new(out / "packet.json", packet)
    write_json_new(out / "annotations.blank.json", blank_annotations(packet))
    (out / "review.html").write_text(render(packet, template))
    jsonl(out / "reports.jsonl", packet["cases"])


def old_context(keys, calls, probabilities, labels):
    def nullable(value):
        return None if pd.isna(value) else value
    rows = []
    for key in keys:
        uid = key[ID]
        for finding in LABELS:
            rows.append({"case_id": key["case_id"], ID: uid, "finding": finding,
                         "old_categorical_call": nullable(calls.loc[uid, finding]),
                         "old_evidence": nullable(calls.loc[uid, finding + "_evidence"]),
                         "old_severity_score": nullable(calls.loc[uid, finding + "_severity"]),
                         "old_confidence": nullable(calls.loc[uid, finding + "_confidence"]),
                         "old_probability": nullable(probabilities.loc[uid, finding]),
                         "old_fused_target": nullable(labels.loc[uid, finding]),
                         "old_weight": nullable(labels.loc[uid, finding + "_weight"])})
    return rows


def prepare(out, config, root=ROOT):
    out, root, config = Path(out), Path(root), Path(config)
    if out.exists():
        raise FileExistsError("Evaluation packets are frozen; use a new version instead of overwriting")
    settings = json.loads(config.read_text())
    if settings["review_mode"] != "AI-assisted only" or settings["human_reference_available"]:
        raise ValueError("This preparation follows the user's AI-assisted-only choice")
    manifest, _, _, records = load_review(root / "outputs/opus_blind_review_v1")
    if len(records) != manifest["study_count"]:
        raise ValueError("Opus review must finish before constructing follow-up packets")
    paths = {key: root / relative for key, relative in CORE.items()}
    component_manifest = json.loads(paths["component_manifest"].read_text())
    if sha(paths["components"]) != component_manifest["components.csv"]:
        raise ValueError("Global duplicate-component map changed")
    diagnostic = json.loads(paths["baseline_diagnostic"].read_text())
    for path, expected in diagnostic["dump_sha256"].items():
        if sha(root / path) != expected:
            raise ValueError("Old extractor dump changed")
    reports = pd.read_csv(paths["reports"])
    roles = pd.read_csv(paths["roles"])
    components = pd.read_csv(paths["components"])
    calls = pd.read_parquet(paths["categorical"]).set_index(ID)
    probabilities = pd.read_parquet(paths["probability"]).set_index(ID)
    labels = pd.read_parquet(paths["labels"]).set_index(ID)
    for frame in [calls, probabilities, labels]:
        if frame.index.has_duplicates:
            raise ValueError("Duplicate baseline rows")
    inventory = exposure_inventory(root)
    inventory["gold_labels_in_train_csv"] = set(reports.loc[reports[LABELS].notna().any(axis=1), ID])
    inventory["completed_40_report_pilot"] = {r[ID] for r in manifest["studies"]}
    excluded = set().union(*inventory.values())
    reps = eligible_representatives(reports, roles, components, excluded, settings["seed"])
    indexed_reports = reports.set_index(ID)
    languages = indexed_reports.Report.map(detect_language)
    fresh, shortfalls = select_sample(reps, calls, languages, settings)
    if not set(fresh[ID]) <= set(probabilities.index) & set(labels.index):
        raise ValueError("Missing baseline targets for the fresh sample")
    pilot = pd.DataFrame([{ID: r[ID], "component": r["component"], "cohort": "exposed_pilot_development"}
                          for r in manifest["studies"]])
    if set(fresh.component) & set(pilot.component):
        raise ValueError("Fresh sample overlaps a pilot component")
    rubric = paths["rubric"].read_text()
    pilot_packet, pilot_keys = make_packet(pilot, indexed_reports, rubric, "pilot",
                                          "Pilot report adjudication · 40 reports", settings["seed"])
    fresh_packet, fresh_keys = make_packet(fresh, indexed_reports, rubric, "fresh",
                                          f"Fresh report evaluation · {len(fresh)} reports", settings["seed"])
    out.mkdir(parents=True)
    custodian = out / "custodian"
    custodian.mkdir()
    template = (root / "tools/report_review_template.html").read_text()
    for name, packet, keys in [("pilot", pilot_packet, pilot_keys), ("fresh", fresh_packet, fresh_keys)]:
        write_packet(out / "reviewer" / name, packet, template)
        jsonl(custodian / (name + "_key.jsonl"), keys)
        jsonl(custodian / (name + "_old_extractor.jsonl"), old_context(keys, calls, probabilities, labels))
    by_uid = {r[ID]: r["case_id"] for r in pilot_keys}
    for name, path_key in [("pilot_model_comparison.jsonl", "opus_rows"),
                           ("pilot_adjudication_queue.jsonl", "opus_queue"),
                           ("pilot_old_extractor_candidates.jsonl", "opus_candidates")]:
        rows = [json.loads(line) for line in paths[path_key].read_text().splitlines()]
        jsonl(custodian / name, [{"case_id": by_uid[r[ID]], **r} for r in rows])
    mapping = components.set_index(ID).component
    excluded_components = set(mapping.loc[list(excluded)])
    exclusions = [{ID: uid, "component": mapping.loc[uid],
                   "reasons": [name for name, ids in inventory.items() if uid in ids]} for uid in sorted(excluded)]
    jsonl(custodian / "recorded_exclusions.jsonl", exclusions)
    jsonl(custodian / "eligible_representatives.jsonl", reps.to_dict("records"))
    coverage = {"eligible_training_components": len(reps), "recorded_excluded_studies": len(excluded),
                "recorded_excluded_components": len(excluded_components),
                "exclusion_sources": {name: len(ids) for name, ids in inventory.items()},
                "fresh_reports": len(fresh), "fresh_finding_slots": len(fresh) * len(LABELS),
                "cohorts": fresh.cohort.value_counts().to_dict(), "shortfalls": shortfalls,
                "language_is_heuristic": True,
                "by_cohort": {cohort: {"reports": len(group),
                    "language_heuristic": group.language_heuristic.value_counts().to_dict(),
                    "old_positive_counts": {finding: int(group[ID].map(calls[finding]).eq(1).sum()) for finding in LABELS}}
                    for cohort, group in fresh.groupby("cohort")},
                "pilot_reports": len(pilot), "pilot_finding_slots": len(pilot) * len(LABELS),
                "random_control_unit": "one seeded report representative per eligible duplicate component",
                "freshness_limit": "No recorded prior report-review/pilot exposure; unlogged historical readings cannot be excluded. Existing Qwen extraction predates this evaluation; historical image-model training is not excluded."}
    write_json_new(out / "coverage.json", coverage)
    prior_cost = json.loads(paths["opus_comparison"].read_text())["shared_total_accounted_cost_usd_upper"]
    protocol = {
        "status": "prepared_only", "review_mode": "AI-assisted only", "human_reference_available": False,
        "purpose": "Diagnose report extraction and evaluate AI consistency; prepare candidates for later controlled image-model tests.",
        "pilot": "All 40 exposed pilot reports, all 480 slots. Includes disagreements, corroborated candidates, agreements and the Sonnet-rejected report. Development adjudication only.",
        "fresh": "Freeze before new extraction or answer review. Random controls first; priority and coverage supplements are descriptive challenge sets, never pooled into a population error rate.",
        "missing_and_silent": "Keep invalid responses in failure/abstention denominators. Silence is unmentioned, not a negative. Preserve uncertainty and unresolved thresholds.",
        "blinding": "Reviewer files contain original reports, fixed rubric, case aliases and blank fields. Keys, sampling reasons and old/model answers reside only in custodian/. Do not open custodian results until reviewer annotations and candidate settings are frozen.",
        "fresh_recipe": "Freeze each candidate model/prompt/effort and parsing version before any fresh results are opened. Do not revise prompts on this fresh set and continue to call it confirmation.",
        "allowed_conclusions": ["AI-assisted report adjudication", "paired model agreement/disagreement", "evidence-span validity", "operational failure and abstention rates"],
        "unsupported_conclusions": ["independent extraction accuracy", "clinical image truth", "guaranteed image-model improvement"],
        "later_image_test": "Separate extraction repair from severity/fusion. Hold architecture, initialization, split and schedule fixed. Use unchanged evaluation labels. Outer0 is development; a passing screen needs unexposed confirmation and matched seeds.",
        "api_calls_made": 0, "new_cost_usd": 0.0, "production_labels_changed": False,
        "kaggle_jobs_changed": False, "candidate_labels_generated": False,
        "accounted_existing_shared_spend_usd_upper": prior_cost, "existing_shared_cap_usd": 10.0,
        "budget_note": "Preparing 200 reports does not authorize a larger paid extraction. No requests are submitted by this tool."}
    write_json_new(out / "protocol.json", protocol)
    inputs = {name: {"path": str(path.relative_to(root)), "sha256": sha(path)} for name, path in paths.items()}
    for path in inventory:
        if (root / path).is_file():
            inputs["exposure:" + path] = {"path": path, "sha256": sha(root / path)}
    inputs["config"] = {"path": str(config.resolve().relative_to(root)), "sha256": sha(config)}
    output_hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    frozen = {"schema_version": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
              "settings": settings, "inputs": inputs,
              "source_sha256": {path: sha(root / path) for path in SOURCES}, "output_sha256": output_hashes,
              "fresh_packet_sha256": fresh_packet["packet_sha256"], "pilot_packet_sha256": pilot_packet["packet_sha256"],
              "api_calls_made": 0, "production_labels_changed": False}
    write_json_new(out / "manifest.json", frozen)
    return {"pilot_reports": len(pilot), "fresh_reports": len(fresh), "fresh_cohorts": coverage["cohorts"],
            "fresh_finding_slots": coverage["fresh_finding_slots"], "shortfalls": shortfalls,
            "api_calls_made": 0, "new_cost_usd": 0.0, "human_reference_available": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/report_label_evaluation_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/report_label_evaluation.json")
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
