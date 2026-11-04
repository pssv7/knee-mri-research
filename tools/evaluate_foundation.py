"""Compare two foundation arms, or explicitly label a historical Tiny comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.validation.proxy import (ID, LABELS, sha, write_json_new,
    validate_prediction_provenance, paired_proxy_evaluation)
from tools.train_foundation import verify_experiment


def matched_contracts(control, candidate, historical_tiny=False):
    common = ["roles_sha256", "proxy_labels_sha256", "protected_exclusions_sha256"]
    for key in common:
        if not control.get(key) or control[key] != candidate.get(key):
            raise ValueError(f"Evaluation data mismatch: {key}")
    if historical_tiny:
        if control.get("backbone") != "convnext_tiny.fb_in22k_ft_in1k":
            raise ValueError("Historical reference must be the preserved Tiny control")
        if candidate.get("arm") != "orthofoundation":
            raise ValueError("Historical comparison requires the OrthoFoundation candidate")
        return
    if [control.get("arm"), candidate.get("arm")] != ["dinov3", "orthofoundation"]:
        raise ValueError("Matched pair must be DINOv3 control and OrthoFoundation candidate")
    for key in ["foundation_schema_version", "stage", "seed", "training_recipe",
                "runtime_versions", "input_config_sha256", "experiment_sha256",
                "initial_head_sha256", "runtime_code_sha256"]:
        if control.get(key) is None or control[key] != candidate.get(key):
            raise ValueError(f"Matched foundation comparison differs: {key}")
    if control.get("encoder_sha256") == candidate.get("encoder_sha256"):
        raise ValueError("Control and candidate cannot use the same pretrained encoder")


def evaluate(experiment, control, candidate, out, historical_tiny=False, draws=2000):
    experiment, out = Path(experiment), Path(out)
    if out.exists():
        raise FileExistsError("Do not overwrite an evaluation")
    spec = verify_experiment(experiment)
    baseline = experiment / "baseline"
    for name, key in [("roles.csv", "roles_sha256"), ("proxy_labels.parquet", "proxy_labels_sha256")]:
        if sha(baseline / name) != spec[key]:
            raise ValueError("Evaluation labels or roles changed")
    roles = pd.read_csv(baseline / "roles.csv")
    outer = roles.loc[roles.role == "outer"].sort_values(ID)
    labels = pd.read_parquet(baseline / "proxy_labels.parquet").set_index(ID).loc[outer[ID]]
    predictions, contracts, sources = [], [], {}
    for label, directory in [("control", control), ("candidate", candidate)]:
        directory = Path(directory)
        contract = json.loads((directory / "run_contract.json").read_text())
        for name, key in [("selected.pt", "checkpoint_sha256"),
                          ("outer_predictions.parquet", "outer_predictions_sha256")]:
            if sha(directory / name) != contract[key]:
                raise ValueError("Prediction or selected-weight bytes changed")
        for key in ["roles_sha256", "proxy_labels_sha256"]:
            if contract[key] != spec[key]:
                raise ValueError("Run belongs to another design")
        pred = pd.read_parquet(directory / "outer_predictions.parquet")
        validate_prediction_provenance(roles, contract, pred)
        predictions.append(pred.set_index(ID).loc[outer[ID], LABELS].to_numpy(float))
        contracts.append(contract)
        sources[label] = sha(directory / "run_contract.json")
    if contracts[1].get("experiment_sha256") != sha(experiment / "experiment.json"):
        raise ValueError("Candidate belongs to another experiment")
    matched_contracts(*contracts, historical_tiny=historical_tiny)
    truth = (labels[LABELS].to_numpy(float) >= .5).astype(float)
    report = paired_proxy_evaluation(truth, *predictions, outer.component.to_numpy(), draws=draws)
    explicit = truth.copy()
    explicit[labels[[f"{name}_weight" for name in LABELS]].to_numpy() < .7] = np.nan
    report["explicit_label_diagnostic"] = paired_proxy_evaluation(
        explicit, *predictions, outer.component.to_numpy(), draws=draws)
    secondary = [LABELS.index(name) for name in spec["screen"]["secondary_findings"]]
    deltas = np.array([report["per_finding"][name]["candidate"] -
                       report["per_finding"][name]["control"] for name in LABELS])
    ci = report["paired_sampling_95ci"]
    passes = (report["macro_delta"] is not None and report["macro_delta"] >= .005
              and ci is not None and ci[0] > 0 and deltas.min() >= -.020)
    report.update(sources=sources, secondary_mean_delta=float(deltas[secondary].mean()),
        comparison_scope="historical Tiny; recipe/runtime/input representation differ" if historical_tiny
                         else "matched architecture and head; pretrained encoder differs",
        matched_pretraining_comparison=not historical_tiny,
        numerical_screen_pass=bool(passes),
        decision="confirmation_required" if passes else "screen_failed",
        independent_confirmation_required=True, automatic_followup_authorized=False)
    write_json_new(out, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["experiment", "control", "candidate", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--historical-tiny", action="store_true")
    p.add_argument("--draws", type=int, default=2000)
    a = p.parse_args()
    r = evaluate(a.experiment, a.control, a.candidate, a.out, a.historical_tiny, a.draws)
    print(json.dumps({k: r[k] for k in ["macro_delta", "paired_sampling_95ci", "decision", "comparison_scope"]}, indent=2))
