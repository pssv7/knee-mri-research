"""Isolated categorical-call repairs with unchanged control target mapping/weights."""
import numpy as np

from src.labels import fuse, mode_fusion
from src.labels.llm_extract import SEVERITY_SCORES
from src.labels.openai_extract import validate
from src.validation.proxy import ID, LABELS, validate_roles


def frozen_map(query, source, reference):
    source, reference = np.sort(np.asarray(source, float)), np.sort(np.asarray(reference, float))
    if len(source) < 2 or len(source) != len(reference):
        raise ValueError("Insufficient aligned training calibration")
    query = np.asarray(query, float)
    rank = (np.searchsorted(source, query, side="left") + np.searchsorted(source, query, side="right") - 1) / (2 * (len(source)-1))
    rank = np.clip(rank, 0, 1)
    return (np.quantile(reference, rank) + .02 * rank) / 1.02


def repair(control, roles, cat, prob, reports, assertions_by_uid):
    validate_roles(roles)
    for frame in (control, cat, prob):
        if frame[ID].duplicated().any():
            raise ValueError("Duplicate label IDs")
    ids = set(roles[ID])
    if set(control[ID]) != ids or not ids <= set(cat[ID]) & set(prob[ID]) & set(reports.index):
        raise ValueError("Incomplete label/split membership")
    train_ids = sorted(roles.loc[roles.role == "train", ID])
    if not set(assertions_by_uid) <= set(train_ids):
        raise ValueError("Attempt to repair a nontraining or protected study")
    cat, prob = [f[f[ID].isin(ids)].copy() for f in (cat, prob)]
    lex = fuse.lexicon_frame(reports.loc[cat[ID], "Report"], cat[ID])
    raw = fuse.fuse_scores(lex, mode_fusion.combine(cat, prob)).set_index(ID)
    ref = fuse.fuse_scores(lex, mode_fusion.from_categorical(cat)).set_index(ID)
    original = control.set_index(ID)
    for finding in LABELS:
        reproduced = frozen_map(raw.loc[original.index, finding], raw.loc[train_ids, finding], ref.loc[train_ids, finding])
        if not np.allclose(reproduced, original[finding], atol=1e-12, rtol=0):
            raise ValueError("Original control target mapping does not reproduce: " + finding)
    changed_cat = cat.set_index(ID).copy(deep=True)
    proposals, keeps = [], []
    for uid, assertions in assertions_by_uid.items():
        validate(assertions, reports.loc[uid, "Report"])
        for finding, item in assertions.items():
            state, old = item["report_states"], changed_cat.loc[uid, finding]
            if state not in {"present", "negated"} or item["certainty"] != "definite":
                keeps.append({ID: uid, "finding": finding, "reason": "ambiguous_silent_or_threshold_state"})
                continue
            new = 1.0 if state == "present" else 0.0
            if old == new:
                # Crucial: do not replace severity for an already-positive call.
                keeps.append({ID: uid, "finding": finding, "reason": "same_categorical_call_preserve_existing_severity"})
                continue
            severity = SEVERITY_SCORES.get(item["severity_norm"], .85) if new else 0.0
            changed_cat.loc[uid, finding] = new
            changed_cat.loc[uid, finding + "_severity"] = severity
            proposals.append({ID: uid, "finding": finding, "old_call": None if np.isnan(old) else float(old),
                              "new_call": new, "new_categorical_score": severity,
                              "evidence_quote": item["evidence_quote"], "report_state": state})
    proposed_raw = fuse.fuse_scores(lex, mode_fusion.combine(changed_cat.reset_index(), prob)).set_index(ID)
    candidate = original.copy(deep=True)
    changes = []
    for proposal in proposals:
        uid, finding = proposal[ID], proposal["finding"]
        target = float(frozen_map([proposed_raw.loc[uid, finding]], raw.loc[train_ids, finding], ref.loc[train_ids, finding])[0])
        if abs(target - original.loc[uid, finding]) <= 1e-12:
            keeps.append({ID: uid, "finding": finding, "reason": "categorical_update_does_not_change_frozen_numeric_target"})
            continue
        candidate.loc[uid, finding] = target
        changes.append({**proposal, "old_target": float(original.loc[uid, finding]), "new_target": target,
                        "old_raw": float(raw.loc[uid, finding]), "new_raw": float(proposed_raw.loc[uid, finding]),
                        "weight": float(original.loc[uid, finding + "_weight"])})
    held = roles.loc[roles.role != "train", ID]
    if not candidate.loc[held].equals(original.loc[held]):
        raise AssertionError("Selection or outer labels changed")
    if not candidate[[f+"_weight" for f in LABELS]].equals(original[[f+"_weight" for f in LABELS]]):
        raise AssertionError("Sample weights changed")
    changed_keys = {(r[ID],r['finding']) for r in changes}
    for finding in LABELS:
        actual = {(uid,finding) for uid in original.index if candidate.loc[uid,finding] != original.loc[uid,finding]}
        if actual != {p for p in changed_keys if p[1] == finding}:
            raise AssertionError("Unlogged target change")
    return candidate.reset_index()[control.columns], changes, keeps
