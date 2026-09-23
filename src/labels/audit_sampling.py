"""Sampling for the solo report audit: reserve first, then controls, then strata.

The order is the design. The extraction-validation reserve is drawn before
anything else, from studies no one has read, and every reserved study is then
removed from discovery for *all* findings -- a report read for one finding has
been read. Random controls come next, uniformly from what remains, so their
error rate estimates the population's. Only then are the enriched strata
filled, and a pair already drawn as a control is never drawn again: enriched
cases are selected for looking wrong and must never be pooled into a rate.

Model disagreement only selects cases here. No threshold, rule or weight is
fitted to it.

Pure functions over frames: no file access and no report text. The caller
supplies each study's language code, never the report.
"""

from dataclasses import asdict, dataclass
import zlib

import numpy as np
import pandas as pd

from src import config

PRIORITY = ("Lateral Meniscus", "Contusion", "PF OA")
ID = config.ID_COL


@dataclass(frozen=True)
class Settings:
    """Fixed before sampling (decision D3); changing any of them is a new version."""
    seed: int = 20260923
    discovery_folds: tuple = (0, 2, 4)
    reserve_per_finding: int = 40
    controls: int = 50
    per_stratum: int = 50
    positive: float = 0.5
    fp_max_oof: float = 0.3
    miss_min_oof: float = 0.7

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def rng_for(settings, purpose, finding):
    """An independent, reproducible stream per purpose and finding.

    Drawing one finding's sample can never shift another's, and adding a
    purpose later cannot change what earlier purposes drew.
    """
    return np.random.default_rng([settings.seed, zlib.crc32(f"{purpose}|{finding}".encode())])


def finding_pairs(labels, folds, oof, languages, finding):
    """One row per study in the frame: target, weight, fold, protocol OOF, language."""
    frame = labels[[ID, finding, finding + "_weight"]].rename(
        columns={finding: "target", finding + "_weight": "weight"})
    frame = frame.merge(folds[[ID, "fold"]], on=ID, how="inner", validate="one_to_one")
    frame["oof"] = frame[ID].map(oof[finding])
    frame["language"] = frame[ID].map(languages).fillna("unknown")
    if frame.oof.isna().any():
        raise ValueError(f"{finding}: {int(frame.oof.isna().sum())} studies lack protocol OOF")
    frame.insert(1, "finding", finding)
    return frame.sort_values(ID, kind="stable").reset_index(drop=True)


def suspect_flags(pairs, settings):
    positive = pairs.target >= settings.positive
    return pd.DataFrame({
        "meets_false_positive": positive & (pairs.oof <= settings.fp_max_oof),
        "meets_missed_positive": ~positive & (pairs.oof >= settings.miss_min_oof)},
        index=pairs.index)


def proportional(counts, total):
    """Largest-remainder allocation of `total` across cells, capped at each cell's size."""
    counts = counts[counts > 0]
    if counts.sum() <= total:
        return counts.copy()
    exact = counts / counts.sum() * total
    alloc = np.floor(exact).astype(int)
    order = (exact - alloc).sort_values(ascending=False, kind="stable").index
    for cell in order[: total - alloc.sum()]:
        alloc[cell] += 1
    # A cell can be allocated more than it holds only through rounding; move the excess on.
    while (alloc > counts).any():
        over = (alloc - counts).clip(lower=0)
        alloc -= over
        room = (counts - alloc)[lambda s: s > 0].sort_values(ascending=False, kind="stable")
        for cell in room.index[: int(over.sum())]:
            alloc[cell] += 1
    return alloc


def draw_reserve(pairs_by_finding, exposed, settings):
    """The extraction-validation reserve, stratified by target side and language.

    Only discovery-fold studies with no known prior exposure are eligible.
    Returns (reserve rows, shortfalls).
    """
    rows, shortfalls = [], []
    half = settings.reserve_per_finding // 2
    for finding in PRIORITY:
        pairs = pairs_by_finding[finding]
        eligible = pairs[pairs.fold.isin(settings.discovery_folds) & ~pairs[ID].isin(exposed)]
        rng = rng_for(settings, "reserve", finding)
        for side, quota in (("target_positive", half), ("target_negative", settings.reserve_per_finding - half)):
            side_pairs = eligible[(eligible.target >= settings.positive) == (side == "target_positive")]
            if len(side_pairs) < quota:
                shortfalls.append({"finding": finding, "stratum": side,
                                   "wanted": quota, "available": len(side_pairs)})
            alloc = proportional(side_pairs.language.value_counts().sort_index(), quota)
            for language, n in alloc.items():
                cell = side_pairs[side_pairs.language == language]
                chosen = cell.iloc[np.sort(rng.choice(len(cell), size=int(n), replace=False))]
                rows.append(chosen.assign(role="validation_reserve", stratum=side,
                                          inclusion_probability=n / len(cell),
                                          provenance="random_draw"))
    reserve = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return reserve, shortfalls


def draw_audit(pairs_by_finding, reserved_studies, prior_pairs, settings):
    """Controls, then the two enriched strata, per priority finding.

    `prior_pairs` maps (study, finding) -> earlier review id for the existing
    label_review_v1 queue. Those pairs fill enriched quotas first, marked
    non-random, and count as controls only if the new control draw picks them.
    Returns (sample rows, shortfalls, population sizes).
    """
    rows, shortfalls, population = [], [], {}
    for finding in PRIORITY:
        pairs = pairs_by_finding[finding]
        pop = pairs[pairs.fold.isin(settings.discovery_folds) & ~pairs[ID].isin(reserved_studies)]
        pop = pop.join(suspect_flags(pop, settings))
        pop["prior_review_id"] = [prior_pairs.get((uid, finding)) for uid in pop[ID]]
        population[finding] = len(pop)

        rng = rng_for(settings, "controls", finding)
        n = min(settings.controls, len(pop))
        control_idx = pop.index[np.sort(rng.choice(len(pop), size=n, replace=False))]
        rows.append(pop.loc[control_idx].assign(role="control", provenance="random_draw",
                                                inclusion_probability=n / len(pop)))
        if n < settings.controls:
            shortfalls.append({"finding": finding, "stratum": "control",
                               "wanted": settings.controls, "available": len(pop)})

        remaining = pop.drop(index=control_idx)
        for role, flag in (("suspected_false_positive", "meets_false_positive"),
                           ("suspected_missed_positive", "meets_missed_positive")):
            eligible = remaining[remaining[flag]]
            prior = eligible[eligible.prior_review_id.notna()]
            rng = rng_for(settings, role, finding)
            if len(prior) > settings.per_stratum:
                prior = prior.iloc[np.sort(rng.choice(len(prior), size=settings.per_stratum, replace=False))]
            rows.append(prior.assign(role=role, provenance="label_review_v1",
                                     inclusion_probability=np.nan))
            fresh = eligible.drop(index=prior.index)
            fresh = fresh[fresh.prior_review_id.isna()]
            k = min(settings.per_stratum - len(prior), len(fresh))
            if k:
                chosen = fresh.iloc[np.sort(rng.choice(len(fresh), size=k, replace=False))]
                rows.append(chosen.assign(role=role, provenance="random_draw",
                                          inclusion_probability=k / len(fresh)))
            if len(prior) + k < settings.per_stratum:
                shortfalls.append({"finding": finding, "stratum": role, "wanted": settings.per_stratum,
                                   "available": len(prior) + len(fresh)})
    sample = pd.concat(rows, ignore_index=True)
    if sample.duplicated([ID, "finding"]).any():
        raise AssertionError("a pair was drawn twice")
    return sample, shortfalls, population


def coverage(frame, extra):
    """Counts by finding and role over the given descriptive columns; no text."""
    out = {}
    for column in extra:
        table = frame.groupby(["finding", "role", column], dropna=False).size()
        out[column] = {f"{f} | {r} | {v}": int(n) for (f, r, v), n in table.items()}
    return out
