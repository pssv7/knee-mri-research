"""Labels for perturbation variants, and the probe scored against them.

Kept apart from `perturb.py`, whose bytes are pinned by the Kaggle probe's run
contract. Labels are the study's existing report-mined labels carried through
each transform -- never new readings of the image.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config
from src.model import augment
from src.model.perturb import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, VARIANTS, _wide

# How each variant's label is written from its study's existing mined label.
# Photometric, slice and covariate variants leave the anatomy unchanged, so the
# study's label carries over. A mirrored knee has two candidate readings: the
# repository convention swaps medial/lateral labels ("flip_lr:swap"); the
# anatomical reading keeps them, since a mirrored right knee is a left knee whose
# medial side is still medial ("flip_lr:keep"). Scoring both is the test.
LABEL_RULES = {name: "inherit" for name in VARIANTS}
FLIP_RULES = ("swap", "keep")


def variant_labels(truth: pd.DataFrame, name: str, flip_rule: Optional[str] = None) -> pd.DataFrame:
    """Labels for one variant, written from the unmodified study's labels.

    `truth` is (study x twelve findings), 0/1. Returns the same shape. These are
    the original report-mined labels carried through a known transform, not new
    readings of the image.
    """
    if name not in VARIANTS:
        raise KeyError(name)
    if name == "flip_lr":
        if flip_rule not in FLIP_RULES:
            raise ValueError("flip_lr needs flip_rule 'swap' or 'keep'")
        if flip_rule == "swap":
            swapped = truth[config.LABELS].to_numpy()[:, augment.lateral_swap_order()]
            return pd.DataFrame(swapped, index=truth.index, columns=config.LABELS)
    elif flip_rule is not None:
        raise ValueError("flip_rule applies only to flip_lr")
    return truth[config.LABELS].copy()


def labelled_arms():
    """(arm name, variant, flip rule) for every scored arm, identity first."""
    arms = []
    for name in VARIANTS:
        if name == "flip_lr":
            arms += [(f"flip_lr:{rule}", name, rule) for rule in FLIP_RULES]
        else:
            arms.append((name, name, None))
    return arms


def score_against_labels(predictions: pd.DataFrame, truth: pd.DataFrame, components,
                         draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> Dict:
    """AUC of every variant against its written labels, paired with identity.

    Raw (unaligned) variant outputs are scored against that variant's labels;
    for the flip, swapping the labels is equivalent to re-permuting outputs.
    The bootstrap resamples link components, the same units as the outer0
    screen, and applies each draw to identity and variant together.
    """
    from src.validation.proxy import strict_aucs

    studies = list(truth.index)
    components = np.asarray(pd.Series(components, index=truth.index).loc[studies])
    units = [np.flatnonzero(components == g) for g in sorted(set(components))]
    draws_ix = [np.concatenate([units[i] for i in r]) for r in
                np.random.default_rng(seed).integers(len(units), size=(draws, len(units)))]
    base_truth = variant_labels(truth, "identity").to_numpy(float)
    base_scores = _wide(predictions, "identity", studies)
    base_aucs, base_macro = strict_aucs(base_truth, base_scores)
    boot_base = [strict_aucs(base_truth[ix], base_scores[ix])[0] for ix in draws_ix]
    arms = {}
    for arm, name, rule in labelled_arms():
        if not (predictions.variant == name).any():
            continue
        t = variant_labels(truth, name, rule).to_numpy(float)
        s = _wide(predictions, name, studies)
        aucs, macro = strict_aucs(t, s)
        deltas = np.array([strict_aucs(t[ix], s[ix])[0] - b for ix, b in zip(draws_ix, boot_base)])
        arms[arm] = {
            "variant": name, "label_rule": rule or LABEL_RULES[name], "macro": macro,
            "macro_delta": macro - base_macro,
            "macro_delta_95ci": np.quantile(deltas.mean(axis=1), [.025, .975]).tolist(),
            "per_finding": {f: {"auc": float(aucs[j]), "delta": float(aucs[j] - base_aucs[j]),
                                "delta_95ci": np.quantile(deltas[:, j], [.025, .975]).tolist()}
                            for j, f in enumerate(config.LABELS)},
        }
    return {"studies": len(studies), "components": len(units), "draws": draws, "seed": seed,
            "identity_macro": base_macro, "labels": "original report-mined labels carried through "
            "each transform; not new image readings", "arms": arms}
