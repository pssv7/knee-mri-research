"""Recompute local evidence for the September 28 literature review, without training.

Run from the repository root. Saved intervals are copied from historical
readouts, not recomputed or represented as new independent evidence.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ID_COL, LABELS


def main():
    out = ROOT / "outputs/accuracy_research_20260928"
    if out.exists():
        raise FileExistsError("Preserve the existing research snapshot")
    sources = {}

    def read(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".json":
            return json.loads(path.read_text())
        frame = pd.read_parquet(path)
        assert not frame[ID_COL].duplicated().any()
        return frame.set_index(ID_COL)

    labels = read("outputs/validation_fallback_v2/outer0/proxy_labels.parquet")
    control_path = "outputs/silence_screen_v1_results/control/outer_predictions.parquet"
    control = read(control_path).sort_index()
    assert set(control.index) == set(labels.index[labels.role == "outer"])
    labels = labels.loc[control.index]
    truth = (labels[LABELS].to_numpy() >= .5).astype(int)

    def aucs(pred):
        assert set(pred.index) == set(control.index)
        values = pred.loc[control.index, LABELS].to_numpy(float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert all(len(np.unique(truth[:, j])) == 2 for j in range(12))
        return np.array([roc_auc_score(truth[:, j], values[:, j]) for j in range(12)])

    baseline = aucs(control)
    diagnostics = []
    for j, name in enumerate(LABELS):
        weights = labels[name + "_weight"].to_numpy()
        explicit = weights >= .7
        diagnostics.append({
            "finding": name, "proxy_auc": float(baseline[j]),
            "positive": int(truth[:, j].sum()),
            "negative": int((1 - truth[:, j]).sum()),
            "explicit_proxy_auc": float(roc_auc_score(truth[explicit, j], control[name].to_numpy()[explicit])),
            "explicit_positive": int(truth[explicit, j].sum()),
            "explicit_negative": int((1 - truth[explicit, j]).sum()),
            "silent_fraction": float(np.isclose(weights, .25).mean()),
        })
    experiments = {}
    for name, directory, evaluation in [
        ("distinct_sampling", "kaggle_distinct_training_pair_v1_results", "outer_proxy_evaluation.json"),
        ("convnext_small", "colab_small_224_outer0_results", "reproduced_evaluation.json"),
        ("mirror", "mirror_screen_v1_results", "outer_proxy_evaluation.json"),
        ("silence_mask", "silence_screen_v1_results", "outer_proxy_evaluation.json"),
    ]:
        base = f"outputs/{directory}"
        relative = f"{base}/candidate/outer_predictions.parquet"
        candidate = read(relative)
        contract = read(f"{base}/candidate/run_contract.json")
        assert sources[relative] == contract["outer_predictions_sha256"]
        assert set(contract["outer_uids"]) == set(control.index)
        assert not set(contract["train_uids"]) & set(control.index)
        assert not set(contract["selection_uids"]) & set(control.index)
        measured = aucs(candidate)
        stored = read(f"{base}/{evaluation}")
        np.testing.assert_allclose(measured.mean(), stored["candidate_macro"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(baseline.mean(), stored["control_macro"], atol=1e-12, rtol=0)
        for j, finding in enumerate(LABELS):
            np.testing.assert_allclose(measured[j], stored["per_finding"][finding]["candidate"], atol=1e-12, rtol=0)
        if name != "convnext_small":
            repeated = read(f"{base}/control/outer_predictions.parquet")
            np.testing.assert_array_equal(control[LABELS], repeated.loc[control.index, LABELS])
        experiments[name] = {
            "recomputed_macro": float(measured.mean()),
            "recomputed_delta": float(measured.mean() - baseline.mean()),
            "historical_sampling_95ci_not_recomputed": stored["paired_sampling_95ci"],
            "per_finding_delta": dict(zip(LABELS, (measured - baseline).tolist())),
        }
    explicit_saved = read("outputs/silence_screen_v1_results/explicit_evaluation.json")
    for row in diagnostics:
        np.testing.assert_allclose(row["explicit_proxy_auc"], explicit_saved["per_finding"][row["finding"]]["explicit_control"], atol=1e-12, rtol=0)
    evidence = {
        "date": "2026-09-28", "n_studies": len(control),
        "metric": "macro ROC AUC against mined report labels, not image truth",
        "control_macro": float(baseline.mean()),
        "diagnostics": diagnostics, "experiments": experiments,
        "silence_mask_historical_primary": explicit_saved["decision"],
        "limits": [
            "Outer0 has now been used repeatedly for model development.",
            "Explicit means weight >= 0.7, not independent image annotation.",
            "Subsets have different case mix; AUC differences between them are not causal effects.",
            "Historical bootstrap intervals condition on fixed models and labels.",
            "No GPU run, image review, new seed, new test set or checkpoint-byte verification was performed.",
        ],
        "source_sha256": sources,
    }
    out.mkdir(parents=True)
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(diagnostics).sort_values("proxy_auc").to_csv(out / "finding_diagnostics.csv", index=False)
    print(f"Verified all 48 candidate finding AUCs; control macro={baseline.mean():.9f}; n={len(control)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
