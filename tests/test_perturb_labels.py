"""Labels written for perturbation variants, and scoring against them.

The mirror is the case that matters: a flip read with the wrong label rule
looks like a model failure. The scorer must tell the two readings apart.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import perturb_labels


def long_frame(values_by_variant, studies):
    rows = []
    for name, values in values_by_variant.items():
        for uid, p in zip(studies, values):
            rows.append({config.ID_COL: uid, "variant": name, **dict(zip(config.LABELS, p))})
    return pd.DataFrame(rows)


def truth_frame(n=40, seed=3):
    rng = np.random.default_rng(seed)
    studies = [f"s{i:02d}" for i in range(n)]
    values = np.zeros((n, len(config.LABELS)), dtype=int)
    values[: n // 2] = 1
    for j in range(len(config.LABELS)):
        rng.shuffle(values[:, j])
    return pd.DataFrame(values, index=pd.Index(studies, name=config.ID_COL), columns=config.LABELS)


def test_variant_labels_inherit_and_the_flip_is_written_both_ways():
    truth = truth_frame()
    assert perturb_labels.variant_labels(truth, "noise_5").equals(truth)
    keep = perturb_labels.variant_labels(truth, "flip_lr", "keep")
    swap = perturb_labels.variant_labels(truth, "flip_lr", "swap")
    assert keep.equals(truth)
    for medial, lateral in config.LATERAL_LABEL_PAIRS:
        assert (swap[medial] == truth[lateral]).all() and (swap[lateral] == truth[medial]).all()
    assert (swap["MCL"] == truth["MCL"]).all()
    with pytest.raises(ValueError, match="swap"):
        perturb_labels.variant_labels(truth, "flip_lr")
    with pytest.raises(ValueError, match="only to flip_lr"):
        perturb_labels.variant_labels(truth, "noise_5", "keep")
    arms = [a for a, _, _ in perturb_labels.labelled_arms()]
    assert arms[0] == "identity" and {"flip_lr:swap", "flip_lr:keep"} <= set(arms) and len(arms) == 16


def test_label_scoring_detects_which_flip_reading_the_scores_follow():
    truth = truth_frame()
    rng = np.random.default_rng(4)
    base = truth.to_numpy(float) * 0.6 + rng.uniform(0, 0.4, truth.shape)
    # An anatomy-reading model: the mirrored scan scores like the original.
    frame = long_frame({"identity": base, "flip_lr": base, "noise_5": base[:, ::-1]},
                       list(truth.index))
    report = perturb_labels.score_against_labels(frame, truth, list(truth.index), draws=200)
    arms = report["arms"]
    assert arms["flip_lr:keep"]["macro_delta"] == pytest.approx(0)
    assert arms["flip_lr:swap"]["macro_delta"] < -0.1
    swap = arms["flip_lr:swap"]["per_finding"]
    assert swap["Medial Meniscus"]["delta_95ci"][1] < 0 and swap["ACL"]["delta"] == pytest.approx(0)
    assert arms["noise_5"]["macro_delta"] < 0
    assert report["identity_macro"] == pytest.approx(1.0)
