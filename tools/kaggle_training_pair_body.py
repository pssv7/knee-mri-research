"""Remote body embedded by prepare_kaggle_training_pair.py."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def screening_gate(delta, finding_deltas, required_macro_delta):
    """Inclusive allocation boundaries shared by decisions and receipts."""
    import math
    values = list(finding_deltas)
    return {"passed": bool(values) and math.isfinite(delta)
            and delta >= required_macro_delta
            and all(math.isfinite(value) and value >= -.020 for value in values),
            "macro_delta": delta, "required_macro_delta": required_macro_delta,
            "maximum_finding_drop": .020}


def validate_contracts(contracts, candidate, control="baseline"):
    if candidate not in ("protocol", "classwise", "convnextv2", "slice_transformer",
                         "spatial") or len(contracts) != 2:
        raise ValueError("Expected a registered baseline/candidate pair")
    contracts = [dict(contract) for contract in contracts]
    specs = [dict(contract.pop("experiment")) for contract in contracts]
    if [spec.pop("name") for spec in specs] != [control, candidate]:
        raise ValueError("Unexpected experiment ordering")
    if candidate in ("classwise", "convnextv2", "slice_transformer", "spatial"):
        for contract in contracts:
            if not contract.get("shared_initial_weights_sha256"):
                raise ValueError("Missing shared initialization hash")
            contract.pop("initial_weights_sha256")
    if candidate == "convnextv2":
        for contract in contracts:
            if (contract.get("aggregation_seed") != contract.get("seed")
                    or contract.get("aggregation_seed") is None
                    or contract.get("feature_dim") != 768
                    or contract.get("input_defaults") != {
                        "input_mean": None, "input_std": None, "backbone_lr_scale": 1.0}):
                raise ValueError("Missing or invalid backbone comparison controls")
    if contracts[0] != contracts[1]:
        raise ValueError("Training settings or initial weights differ between runs")
    if candidate == "slice_transformer":
        if any(spec["series_selection"] != "protocol" for spec in specs):
            raise ValueError("Slice comparison requires protocol selection in both arms")
        for spec in specs:
            spec.setdefault("slice_transformer", False)
    if candidate == "spatial":
        # Both arms are the 24-slice protocol model; only the 3D path differs.
        if any(spec["series_selection"] != "protocol" for spec in specs):
            raise ValueError("Spatial comparison requires protocol selection in both arms")
        if any(spec["slices"] != 24 for spec in specs):
            raise ValueError("Spatial comparison requires 24 slices in both arms")
        for spec in specs:
            spec.setdefault("spatial_encoder", False)
            if spec.pop("plane_adapters", False):
                raise ValueError("Plane adapters are a separate ablation")
    variable, expected = {
        "protocol": ("series_selection", ["uniform", "protocol"]),
        "classwise": ("classwise_pool", [False, True]),
        "convnextv2": ("backbone", ["convnext_tiny.fb_in22k_ft_in1k",
                                    "convnextv2_tiny.fcmae_ft_in22k_in1k"]),
        "slice_transformer": ("slice_transformer", [False, True]),
        "spatial": ("spatial_encoder", [False, True]),
    }[candidate]
    if [spec.pop(variable) for spec in specs] != expected:
        raise ValueError("Unexpected experiment variable values")
    if specs[0] != specs[1]:
        raise ValueError("More than the registered experiment variable changed")


def main():
    output = Path("/kaggle/working")
    started = time.monotonic()
    required_macro_delta = globals().get("REQUIRED_MACRO_DELTA", .005 if globals().get("CANDIDATE") == "spatial" else .003)
    progress = {"state": "running", "completed": [], "labels_sha256": LABELS_SHA256,
                "fold": FOLD, "required_macro_delta": required_macro_delta,
                "maximum_finding_drop": .020}

    def save_progress():
        progress["elapsed_seconds"] = time.monotonic() - started
        (output / "pair_progress.json").write_text(json.dumps(progress, indent=2) + "\n")

    save_progress()
    try:
        if RECOVERY:
            checkpoints = attached_files("baseline_lit_fold1.ckpt")
            assert len(checkpoints) == 1, "Recovery dataset is missing or ambiguous"
            with checkpoints[0].open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == RECOVERY["checkpoint_sha256"]
        for name, smoke_source in globals().get("SMOKE_SCRIPTS", {"ddp_smoke": SMOKE_SOURCE}).items():
            smoke = output / name
            smoke.mkdir(exist_ok=False)
            (smoke / "train.py").write_text(smoke_source)
            progress["active"] = name
            save_progress()
            subprocess.run([sys.executable, "-u", str(smoke / "train.py")],
                           cwd=smoke, check=True, timeout=SMOKE_TIMEOUT)
            measurement = smoke / "training_measurement.json"
            if measurement.exists():
                measured = json.loads(measurement.read_text())
                if not measured["training_memory_passed"]:
                    raise ValueError("Smoke failed the training memory gate")
                contract = json.loads((smoke / 'run_contract.json').read_text())
                train = measured['phases']['train'][-1]
                validation = measured['phases']['validation'][-1]
                studies_per_batch = contract['microbatch'] * contract['devices']
                measured['projected_12_epoch_hours'] = 12 * (
                    train['seconds'] / train['batches'] * (4349 - VALIDATION_STUDIES) / studies_per_batch
                    + validation['seconds'] / validation['batches'] * (VALIDATION_STUDIES + 58) / studies_per_batch) / 3600
                measured['projection_limitation'] = 'Short warm-up sample; excludes export, final OOF and session variability'
                progress.setdefault("smoke_measurements", {})[name] = measured
        progress["smoke_passed"] = True
        save_progress()
        if SMOKE_ONLY:
            progress.update(state="complete", active=None, smoke_only=True)
            return
        for name, source in TRAINING_SCRIPTS.items():
            directory = output / f"{name}_fold{FOLD}"
            directory.mkdir(exist_ok=False)
            script = directory / "train.py"
            script.write_text(source)
            progress["active"] = name
            save_progress()
            print(f"Starting {name}, fold {FOLD}, 12 epochs", flush=True)
            subprocess.run([sys.executable, "-u", str(script)], cwd=directory,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"},
                           check=True, timeout=RUN_TIMEOUT)
            progress["completed"].append(name)
            save_progress()

        sys.path.insert(0, str(output / f"{CONTROL}_fold{FOLD}"))
        import numpy as np
        import pandas as pd
        from src import config
        from src.model import artifacts

        paths = [output / f"{name}_fold{FOLD}" for name in TRAINING_SCRIPTS]
        contracts = [json.loads((p / "run_contract.json").read_text()) for p in paths]
        validate_contracts(contracts, CANDIDATE, CONTROL)
        metadata = [artifacts.load_metadata(p / f"convnext_fold{FOLD}.pt") for p in paths]
        for key in ("train_uids", "val_uids", "fold", "labels"):
            if metadata[0][key] != metadata[1][key]:
                raise ValueError(f"Different checkpoint provenance: {key}")
        folded = [pd.read_parquet(p / "folds.parquet") for p in paths]
        pd.testing.assert_frame_equal(folded[0], folded[1])
        label_paths = attached_files("train_labels.parquet")
        assert len(label_paths) == 1
        assert hashlib.sha256(label_paths[0].read_bytes()).hexdigest() == LABELS_SHA256
        labels = pd.read_parquet(label_paths[0])
        comparison_path = output / "compare_oof.py"
        comparison_path.write_text(COMPARE_SOURCE)
        namespace = {"__name__": "pair_comparison", "__file__": str(comparison_path)}
        exec(compile(COMPARE_SOURCE, str(comparison_path), "exec"), namespace)
        frames = [namespace["read_predictions"]([p / f"oof_fold{FOLD}.parquet"], folded[0]) for p in paths]
        expected = set(metadata[0]["val_uids"])
        assert len(expected) == VALIDATION_STUDIES
        assert expected == set(folded[0].loc[folded[0]["fold"] == FOLD, config.ID_COL])
        assert all(set(frame.index) == expected for frame in frames)
        table, summary = namespace["compare"](labels, *frames, draws=1000, seed=0)
        assert np.isfinite(table.to_numpy()).all(), "Undefined validation AUC"
        delta = summary["macro"]["candidate"] - summary["macro"]["baseline"]
        # Apply the regression guard to all twelve findings, conservatively.
        gate = screening_gate(delta, table.candidate_delta, required_macro_delta)
        passed = gate["passed"]
        summary["screening_gate"] = {
            **gate, "guard_applies_to": "all twelve findings",
            "next_step": (("repeat matched pair on fold 3" if FOLD == 1 else
                           "review both screening folds before the remaining folds")
                          if passed else "review before more training"),
            "note": "Compute allocation screen, not statistical significance or leaderboard gain.",
        }
        summary["checkpoint_sha256"] = [m["sha256"] for m in metadata]
        table.to_csv(output / "per_finding.csv", index_label="finding")
        (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(table.round(5).to_string(), flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        progress.update(state="complete", active=None, screening_passed=passed,
                        screening_gate=summary["screening_gate"])
    except BaseException as exc:
        progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_progress()


if __name__ == "__main__":
    main()
