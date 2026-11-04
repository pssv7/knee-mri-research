"""Stage and evaluate a matched label-only ablation; never launch training."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.proxy import (ID, LABELS, sha, write_json_new, validate_roles,
                                  validate_prediction_provenance, paired_proxy_evaluation)


def validate_changes(control, candidate, roles, changes):
    """Require a complete, exact train-only ledger with every other field fixed."""
    validate_roles(roles)
    for frame in (control, candidate):
        if frame[ID].duplicated().any() or set(frame[ID]) != set(roles[ID]):
            raise ValueError("Label membership mismatch")
        values = frame[LABELS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid label values")
    if list(control.columns) != list(candidate.columns):
        raise ValueError("Label schema changed")
    a = control.set_index(ID).sort_index()
    b = candidate.set_index(ID).reindex(a.index)
    other = [c for c in a.columns if c not in LABELS]
    if not a[other].equals(b[other]):
        raise ValueError("Weights or other non-target fields changed")
    held = roles.loc[roles.role != "train", ID]
    if not a.loc[held].equals(b.loc[held]):
        raise ValueError("Selection or outer labels changed")
    keys = [(r[ID], r["finding"]) for r in changes]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate change entry")
    actual = {(uid, f) for f in LABELS for uid in a.index[a[f] != b[f]]}
    if actual != set(keys):
        raise ValueError("Change ledger does not exactly cover target differences")
    for r in changes:
        uid, f = r[ID], r["finding"]
        if r["old_target"] != a.loc[uid, f] or r["new_target"] != b.loc[uid, f]:
            raise ValueError("Change ledger target value mismatch")
    return {"changed_slots": len(actual), "changed_studies": len({u for u, _ in actual}),
            "held_out_labels_unchanged": True, "weights_unchanged": True}


def stage_preview(experiment, out):
    from tools.label_repair_experiment import verify_plan
    experiment, out = Path(experiment), Path(out)
    if out.exists():
        raise FileExistsError("Preserve the staged pair; choose a new output")
    plan = verify_plan(experiment)
    preview = experiment / "fresh_sample_preview"
    receipt = json.loads((preview / "receipt.json").read_text())
    if receipt["plan_sha256"] != sha(experiment / "plan.json"):
        raise ValueError("Preview plan changed")
    if receipt["candidate_sha256"] != sha(preview / "candidate_proxy_labels.parquet"):
        raise ValueError("Preview labels changed")
    extraction = ROOT / plan["extraction"]
    for name, expected in receipt["response_record_sha256"].items():
        if sha(extraction / "attempts" / name) != expected:
            raise ValueError("Extraction response changed")
    base = ROOT / plan["baseline"]
    spec = json.loads((base / "control/spec.json").read_text())
    roles = pd.read_csv(base / "control/roles.csv")
    control = pd.read_parquet(base / "control/proxy_labels.parquet")
    candidate = pd.read_parquet(preview / "candidate_proxy_labels.parquet")
    changes = [json.loads(line) for line in (preview / "changes.jsonl").read_text().splitlines()]
    checked = validate_changes(control, candidate, roles, changes)
    if checked["changed_slots"] != receipt["changed_finding_slots"]:
        raise ValueError("Preview receipt mismatch")
    out.mkdir(parents=True)
    for filename in ("exclusions.csv", "raw_counts.parquet"):
        shutil.copyfile(base / filename, out / filename)
    shutil.copyfile(preview / "changes.jsonl", out / "changes.jsonl")
    shutil.copyfile(preview / "receipt.json", out / "label_receipt.json")
    hashes = {}
    for arm in ("control", "candidate"):
        target = out / arm
        target.mkdir()
        shutil.copyfile(base / "control/roles.csv", target / "roles.csv")
        source = base / "control/proxy_labels.parquet" if arm == "control" else preview / "candidate_proxy_labels.parquet"
        shutil.copyfile(source, target / "proxy_labels.parquet")
        arm_spec = {**spec, "design": "outer0_report_label_repair_preview",
                    "label_arm": arm, "label_scope": receipt["status"],
                    "candidate_status": "experimental_preview_predictive_effect_unmeasured",
                    "proxy_labels_sha256": sha(target / "proxy_labels.parquet"),
                    "label_receipt_sha256": sha(out / "label_receipt.json"),
                    "scope": "Partial training relabel; same-model ablation against unchanged report-derived evaluation labels."}
        write_json_new(target / "spec.json", arm_spec)
        hashes[arm] = sha(target / "spec.json")
    pair = {"status": "staged_preview_no_training_launched", "scope": receipt["status"],
            "arm_spec_sha256": hashes, "changes_sha256": sha(out / "changes.jsonl"),
            "label_receipt_sha256": sha(out / "label_receipt.json"),
            "comparison_tool_sha256": sha(__file__), "adoption_gate": plan["adoption_gate"],
            "checks": checked, "kaggle_owner": "user", "training_launched": False,
            "note": "The legacy run-contract arm field describes the sampler; label_arm identifies this comparison."}
    write_json_new(out / "pair.json", pair)
    return pair


def load_pair(design):
    design = Path(design)
    pair = json.loads((design / "pair.json").read_text())
    if pair["comparison_tool_sha256"] != sha(__file__):
        raise ValueError("Frozen comparison source changed")
    for filename, key in [("changes.jsonl", "changes_sha256"), ("label_receipt.json", "label_receipt_sha256")]:
        if sha(design / filename) != pair[key]:
            raise ValueError("Pair evidence changed")
    specs, tables = [], []
    for arm in ("control", "candidate"):
        folder = design / arm
        if sha(folder / "spec.json") != pair["arm_spec_sha256"][arm]:
            raise ValueError("Arm spec changed")
        spec = json.loads((folder / "spec.json").read_text())
        if spec.get("label_arm") != arm or spec.get("sampling_policy") != "legacy":
            raise ValueError("Invalid label-only arm")
        for filename, key in [("roles.csv", "roles_sha256"), ("proxy_labels.parquet", "proxy_labels_sha256")]:
            if sha(folder / filename) != spec[key]:
                raise ValueError("Frozen split or labels changed")
        if sha(design / "exclusions.csv") != spec["protected_exclusions_sha256"]:
            raise ValueError("Protected exclusions changed")
        if spec.get("raw_counts_sha256") and sha(design / "raw_counts.parquet") != spec["raw_counts_sha256"]:
            raise ValueError("Header counts changed")
        specs.append(spec)
        tables.append(pd.read_parquet(folder / "proxy_labels.parquet"))
    allowed = {"label_arm", "proxy_labels_sha256"}
    if {k: v for k, v in specs[0].items() if k not in allowed} != {k: v for k, v in specs[1].items() if k not in allowed}:
        raise ValueError("Specs differ beyond training labels")
    roles = pd.read_csv(design / "control/roles.csv")
    excluded = pd.read_csv(design / "exclusions.csv")
    if set(roles[ID]) & set(excluded[ID]) or set(roles.component) & set(excluded.component):
        raise ValueError("Protected membership leaked")
    changes = [json.loads(line) for line in (design / "changes.jsonl").read_text().splitlines()]
    checked = validate_changes(*tables, roles, changes)
    return pair, specs, roles, tables, checked


def evaluate(design, control_dir, candidate_dir, out, draws=2000):
    design, out = Path(design), Path(out)
    if out.exists():
        raise FileExistsError("Do not overwrite an evaluation readout")
    pair, specs, roles, tables, checked = load_pair(design)
    outer = roles[roles.role == "outer"].sort_values(ID)
    predictions, contracts, sources = [], [], {}
    for arm, folder, spec in zip(("control", "candidate"), (control_dir, candidate_dir), specs):
        folder = Path(folder)
        cp = folder / "run_contract.json"
        contract = json.loads(cp.read_text())
        if any(contract.get(k) != v for k, v in spec.items()):
            raise ValueError("Run is not bound to its frozen label arm")
        pp, checkpoint = folder / "outer_predictions.parquet", folder / "selected.pt"
        if contract.get("outer_predictions_sha256") != sha(pp) or contract.get("checkpoint_sha256") != sha(checkpoint):
            raise ValueError("Prediction or checkpoint hash mismatch")
        script_hash = spec["runtime_code_sha256"].get("tools/train_nested_proxy_control.py")
        if not script_hash or contract.get("training_script_sha256") != script_hash:
            raise ValueError("Training script provenance mismatch")
        pred = pd.read_parquet(pp)
        validate_prediction_provenance(roles, contract, pred)
        predictions.append(pred.set_index(ID).loc[outer[ID], LABELS].to_numpy(float))
        contracts.append(contract)
        sources[arm] = {"contract_sha256": sha(cp), "prediction_sha256": sha(pp), "checkpoint_sha256": sha(checkpoint)}
    for key in ("seed", "training_recipe", "initial_weights_sha256", "runtime_versions", "input_config_sha256", "runtime_code_sha256"):
        if contracts[0].get(key) is None or contracts[0][key] != contracts[1].get(key):
            raise ValueError("Matched-control mismatch: " + key)
    truth = tables[0].set_index(ID).loc[outer[ID], LABELS].to_numpy(float)
    report = paired_proxy_evaluation((truth >= .5).astype(float), *predictions, outer.component.to_numpy(), draws=draws)
    gate = pair["adoption_gate"]
    deltas = [r["candidate"] - r["control"] for r in report["per_finding"].values()
              if r["candidate"] is not None and r["control"] is not None]
    ci = report["paired_sampling_95ci"]
    passed = (report["macro_delta"] is not None and report["macro_delta"] >= gate["macro_auc_delta_at_least"]
              and len(deltas) == len(LABELS) and min(deltas) >= gate["per_finding_point_delta_at_least"]
              and ci is not None and ci[0] > gate["paired_component_bootstrap_lower_bound_above"])
    report.update(label_scope=pair["scope"], checks=checked, sources=sources,
                  pair_sha256=sha(design / "pair.json"), label_ablation_screen_passed=passed,
                  confirmation_required=True, training_label_repair_only=True)
    write_json_new(out, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("stage-preview")
    s.add_argument("--experiment", type=Path, default=ROOT / "outputs/extraction_repair_experiment_v1")
    s.add_argument("--out", type=Path, default=ROOT / "outputs/extraction_repair_pair_preview_v1")
    e = sub.add_parser("evaluate")
    for key in ("design", "control-dir", "candidate-dir", "out"):
        e.add_argument("--" + key, type=Path, required=True)
    e.add_argument("--draws", type=int, default=2000)
    args = vars(p.parse_args()); action = args.pop("action")
    print(json.dumps(stage_preview(**args) if action == "stage-preview" else evaluate(**args), indent=2))
