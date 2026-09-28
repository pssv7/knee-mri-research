"""Silence masking for label-driven findings, and its explicit-label endpoint.

The OOF loss analysis (docs/worklog.md, 2026-09-25) found that for OA,
Synovitis, Baker's and Contusion, 40-75% of mis-ordered pairs involve a
"silent" negative: a report that never mentions the finding, carried as target
~0.02 at weight 0.25. The label audit found PF OA labels miss ~22% of positives,
all by omission. Silent cells therefore teach "absent" where the report is
merely quiet. This screen gives those cells weight 0 in *training rows only*,
so the loss learns from what reports state. Nothing is relabelled; targets are
untouched; validation (checkpoint selection) reads targets, not weights.

Scoring against the same mined labels would penalise a model that finds
unmentioned positives, so the declared primary endpoint is AUC on explicitly
labelled cells (weight >= 0.7: corroborated or single-source), averaged over
the masked findings.
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src import config

MASKED = ("Medial OA", "Lateral OA", "PF OA", "Synovitis", "Baker's", "Contusion")
SILENT_WEIGHT = 0.25
EXPLICIT_MIN_WEIGHT = 0.7
GATE = {"explicit_mean_delta_min": 0.005, "unmasked_finding_delta_min": -0.02}


def mask_silence(labels: pd.DataFrame, roles: pd.DataFrame, findings: Sequence[str] = MASKED):
    """Copy of `labels` with silent cells of `findings` weighted 0 in training rows."""
    out = labels.copy()
    train = out[config.ID_COL].isin(roles.loc[roles.role == "train", config.ID_COL])
    counts = {}
    for finding in findings:
        column = f"{finding}_weight"
        silent = train & np.isclose(out[column], SILENT_WEIGHT)
        out.loc[silent, column] = 0.0
        counts[finding] = int(silent.sum())
    return out, counts


def explicit_truth(labels: pd.DataFrame, studies) -> pd.DataFrame:
    """0/1 truth where the cell is explicit (weight >= 0.7), NaN elsewhere."""
    frame = labels.set_index(config.ID_COL).loc[studies]
    truth = (frame[config.LABELS] >= 0.5).astype(float)
    weights = frame[[f"{f}_weight" for f in config.LABELS]].to_numpy()
    return truth.where(weights >= EXPLICIT_MIN_WEIGHT - 1e-9)


def explicit_evaluation(labels: pd.DataFrame, roles: pd.DataFrame, control: np.ndarray,
                        candidate: np.ndarray, draws: int = 2000, seed: int = 20260924) -> Dict:
    """Paired component bootstrap of explicit-label AUC over the masked findings.

    `control` and `candidate` are (outer studies sorted by ID) x twelve.
    """
    from src.validation.proxy import strict_aucs

    outer = roles[roles.role == "outer"].sort_values(config.ID_COL)
    truth = explicit_truth(labels, outer[config.ID_COL]).to_numpy(float)
    full = (labels.set_index(config.ID_COL).loc[outer[config.ID_COL], config.LABELS] >= 0.5).to_numpy(float)
    masked = [config.LABELS.index(f) for f in MASKED]
    unmasked = [j for j in range(len(config.LABELS)) if j not in masked]
    components = outer.component.to_numpy()
    units = [np.flatnonzero(components == g) for g in sorted(set(components))]
    a, _ = strict_aucs(truth, control)
    b, _ = strict_aucs(truth, candidate)
    fa, _ = strict_aucs(full, control)
    fb, _ = strict_aucs(full, candidate)
    rng = np.random.default_rng(seed)
    boot = []
    for r in rng.integers(len(units), size=(draws, len(units))):
        ix = np.concatenate([units[i] for i in r])
        d = strict_aucs(truth[ix], candidate[ix])[0] - strict_aucs(truth[ix], control[ix])[0]
        if np.isfinite(d[masked]).all():
            boot.append(d)
    boot = np.asarray(boot)
    mean_delta = float(np.mean(b[masked] - a[masked]))
    ci = np.quantile(boot[:, masked].mean(axis=1), [0.025, 0.975]).tolist()
    guard = {config.LABELS[j]: float(fb[j] - fa[j]) for j in unmasked}
    decision = {
        "explicit_mean_delta": mean_delta, "explicit_mean_delta_95ci": ci,
        "point_screen_passed": mean_delta >= GATE["explicit_mean_delta_min"]
                               and min(guard.values()) >= GATE["unmasked_finding_delta_min"],
        "paired_lower_bound_positive": ci[0] > 0,
        "promotion_allowed": False, "automatic_followup_training": False, "gate": GATE}
    decision["eligible_for_separate_confirmation"] = (decision["point_screen_passed"]
                                                      and decision["paired_lower_bound_positive"])
    per = {config.LABELS[j]: {
        "explicit_control": float(a[j]), "explicit_candidate": float(b[j]),
        "explicit_delta": float(b[j] - a[j]),
        "explicit_delta_95ci": np.quantile(boot[:, j], [0.025, 0.975]).tolist(),
        "explicit_positive": int(np.nansum(truth[:, j] == 1)), "explicit_negative": int(np.nansum(truth[:, j] == 0)),
        "full_delta": float(fb[j] - fa[j])} for j in range(len(config.LABELS))}
    return {"masked": list(MASKED), "valid_draws": len(boot), "requested_draws": draws,
            "decision": decision, "unmasked_full_delta_guard": guard, "per_finding": per}
