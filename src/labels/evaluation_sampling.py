"""Deterministic component sampling; controls precede every enriched draw."""
import hashlib

import pandas as pd

from src.validation.proxy import ID, LABELS


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ordered(frame, seed, purpose, column):
    return frame.assign(_draw=frame[column].map(lambda x: digest(f"{seed}|{purpose}|{x}"))).sort_values(
        ["_draw", column], kind="stable").drop(columns="_draw")


def eligible_representatives(reports, roles, components, excluded_ids, seed):
    for name, frame in [("reports", reports), ("roles", roles), ("components", components)]:
        if frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"Invalid unique IDs in {name}")
    if set(reports[ID]) != set(components[ID]) or components.component.isna().any():
        raise ValueError("Global component coverage differs from reports")
    if not set(roles[ID]) <= set(components[ID]) or not set(excluded_ids) <= set(components[ID]):
        raise ValueError("Unknown role/exclusion study")
    mapping = components.set_index(ID).component
    if not roles[ID].map(mapping).equals(roles.component):
        raise ValueError("Frozen roles disagree with global duplicate components")
    if (roles.groupby("component").role.nunique() > 1).any():
        raise ValueError("A component crosses train/selection/outer roles")
    if set(roles.role) != {"train", "selection", "outer"}:
        raise ValueError("Unexpected or missing split role")
    protected = set(mapping.loc[list(excluded_ids)]) | set(roles.loc[roles.role != "train", "component"])
    candidates = roles[(roles.role == "train") & ~roles.component.isin(protected)].copy()
    if candidates.empty:
        raise ValueError("No eligible training components")
    representatives = ordered(candidates, seed, "representative", ID).drop_duplicates("component")
    return representatives.sort_values("component").reset_index(drop=True)


def select_sample(representatives, calls, languages, settings):
    seed = settings["seed"]
    if representatives.component.duplicated().any() or representatives[ID].duplicated().any():
        raise ValueError("Need one representative per component")
    work = representatives.copy()
    if not set(work[ID]) <= set(calls.index):
        raise ValueError("Missing old extractor call for an eligible representative")
    work["language_heuristic"] = work[ID].map(languages).fillna("unknown")
    selected, shortfalls = [], []
    used = set()

    def draw(pool, count, purpose, cohort, reason):
        pool = pool[~pool.component.isin(used)]
        chosen = ordered(pool, seed, purpose, "component").head(count).copy()
        if len(chosen) < count:
            shortfalls.append({"reason": reason, "requested": count, "available": len(pool)})
        chosen["cohort"] = cohort
        chosen["sampling_reason"] = reason
        chosen["draw_pool_size"] = len(pool)
        chosen["draw_count"] = len(chosen)
        # For enriched cases this is a conditional draw probability, not an
        # overall inclusion weight and never a population error-rate weight.
        chosen["conditional_draw_probability"] = len(chosen) / len(pool) if len(pool) else None
        used.update(chosen.component)
        selected.append(chosen)
        return len(chosen)

    n = settings["random_controls"]
    if len(work) < n:
        raise ValueError("Insufficient components for the prespecified random controls")
    draw(work, n, "controls", "random_control", "uniform_component_random_control")
    for finding in settings["priority_findings"]:
        positive = work[ID].map(calls[finding]).eq(1)
        for side, mask, key in [("old_positive", positive, "priority_positive_per_finding"),
                                ("old_nonpositive_or_silent", ~positive, "priority_nonpositive_per_finding")]:
            draw(work[mask], settings[key], finding + side, "priority_challenge", finding + ":" + side)
    budget = settings["coverage_cases"]

    def selected_frame():
        return pd.concat(selected, ignore_index=True)

    for finding in LABELS:
        have = selected_frame()[ID].map(calls[finding]).eq(1).sum()
        needed = max(0, settings["minimum_old_positive_per_finding"] - int(have))
        if needed and budget:
            budget -= draw(work[work[ID].map(calls[finding]).eq(1)], min(needed, budget),
                           "coverage:" + finding, "coverage_challenge", "old_positive_coverage:" + finding)
    for language in sorted(set(work.language_heuristic)):
        have = selected_frame().language_heuristic.eq(language).sum()
        needed = max(0, settings["minimum_per_language_heuristic"] - int(have))
        if needed and budget:
            budget -= draw(work[work.language_heuristic.eq(language)], min(needed, budget),
                           "language:" + language, "coverage_challenge", "language_coverage:" + language)
    if budget:
        draw(work, budget, "coverage_fill", "coverage_challenge", "random_supplement_after_enrichment")
    result = selected_frame()
    if result.component.duplicated().any():
        raise AssertionError("Sampled component twice")
    for finding in LABELS:
        available = int(result[ID].map(calls[finding]).eq(1).sum())
        if available < settings["minimum_old_positive_per_finding"]:
            shortfalls.append({"reason": "final_old_positive_coverage:" + finding,
                               "requested": settings["minimum_old_positive_per_finding"], "available": available})
    for language in sorted(set(work.language_heuristic)):
        available = int(result.language_heuristic.eq(language).sum())
        if available < settings["minimum_per_language_heuristic"]:
            shortfalls.append({"reason": "final_language_coverage:" + language,
                               "requested": settings["minimum_per_language_heuristic"], "available": available})
    return result, shortfalls
