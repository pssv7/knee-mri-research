"""Tests for the perturbation probe's variants and summary.

The properties that matter are the ones that would otherwise fail silently:
a flip that moves pixels without re-permuting medial/lateral outputs, a side
toggle that invents a side for an unknown series, a variant that mutates the
shared input, and a summary that compares misaligned studies.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import augment, perturb
from src.model.dataset import META_COLUMNS

SIZE = 6


def study(planes=("Sagittal", "Coronal"), known=(1, 1), right=(0, 1)):
    rng = np.random.default_rng(0)
    meta = np.zeros((len(planes), len(META_COLUMNS)), dtype=np.float32)
    for i, plane in enumerate(planes):
        meta[i, META_COLUMNS.index("is_sagittal")] = plane == "Sagittal"
        meta[i, META_COLUMNS.index("is_coronal")] = plane == "Coronal"
        meta[i, perturb.SIDE_KNOWN] = known[i]
        meta[i, perturb.IS_RIGHT] = right[i]
    return {config.ID_COL: "s0",
            "volumes": rng.integers(0, 256, (len(planes), perturb.DEPTH, SIZE, SIZE), dtype=np.uint8),
            "meta": meta,
            "targets": np.arange(len(config.LABELS), dtype=np.float32),
            "weights": np.ones(len(config.LABELS), dtype=np.float32)}


def test_the_declared_grid_is_fifteen_passes_in_order():
    assert list(perturb.VARIANTS) == [
        "identity", "shift_-20", "shift_-10", "shift_-5", "shift_+5", "shift_+10", "shift_+20",
        "noise_2", "noise_5", "noise_10", "slices_even12", "slices_alternate8",
        "slices_central12", "flip_lr", "side_toggle"]


@pytest.mark.parametrize("name", list(perturb.VARIANTS))
def test_no_variant_mutates_the_shared_input(name):
    item = study()
    volumes, meta = item["volumes"].copy(), item["meta"].copy()
    out = perturb.apply(name, item, "s0")
    assert np.array_equal(item["volumes"], volumes) and np.array_equal(item["meta"], meta)
    assert out["volumes"].dtype == np.uint8 and out["volumes"] is not item["volumes"]


def test_flip_uses_the_plane_axis_and_twice_is_identity():
    item = study()
    flipped = perturb.flip(item)
    # Sagittal: medial-lateral is the slice axis. Coronal: the column axis.
    assert np.array_equal(flipped["volumes"][0], item["volumes"][0][::-1])
    assert np.array_equal(flipped["volumes"][1], item["volumes"][1][:, :, ::-1])
    assert np.array_equal(perturb.flip(flipped)["volumes"], item["volumes"])
    # The side covariate stays as acquired, as in the training augmentation.
    assert np.array_equal(flipped["meta"], item["meta"])


def test_flipped_outputs_are_realigned_and_the_permutation_is_an_involution():
    p = np.arange(len(config.LABELS), dtype=float)[None]
    aligned = perturb.align("flip_lr", p)[0]
    index = {name: i for i, name in enumerate(config.LABELS)}
    for medial, lateral in config.LATERAL_LABEL_PAIRS:
        assert aligned[index[medial]] == index[lateral]
        assert aligned[index[lateral]] == index[medial]
    assert aligned[index["MCL"]] == index["MCL"]
    assert np.array_equal(perturb.align("flip_lr", perturb.align("flip_lr", p)), p)
    assert np.array_equal(perturb.align("shift_+5", p), p)
    assert augment.lateral_swap_order() == list(np.asarray(aligned, int))


def test_side_toggle_inverts_known_sides_only_and_leaves_pixels():
    item = study(known=(1, 0), right=(1, 0))
    out = perturb.toggle_side(item)
    assert out["meta"][0, perturb.IS_RIGHT] == 0.0
    assert out["meta"][1, perturb.IS_RIGHT] == 0.0 and out["meta"][1, perturb.SIDE_KNOWN] == 0.0
    assert np.array_equal(out["volumes"], item["volumes"])
    other = [c for c in range(len(META_COLUMNS)) if c != perturb.IS_RIGHT]
    assert np.array_equal(out["meta"][:, other], item["meta"][:, other])


def test_shift_clips_at_both_ends():
    item = study()
    item["volumes"][:] = 250
    assert (perturb.shift_intensity(item, 20)["volumes"] == 255).all()
    item["volumes"][:] = 3
    assert (perturb.shift_intensity(item, -5)["volumes"] == 0).all()
    assert (perturb.shift_intensity(item, 5)["volumes"] == 8).all()


def test_noise_is_fixed_per_study_and_variant_and_zero_sigma_is_identity():
    item = study()
    a = perturb.apply("noise_5", item, "s0")["volumes"]
    assert np.array_equal(a, perturb.apply("noise_5", item, "s0")["volumes"])
    assert not np.array_equal(a, perturb.apply("noise_5", item, "s1")["volumes"])
    assert not np.array_equal(a, perturb.apply("noise_10", item, "s0")["volumes"])
    assert np.array_equal(perturb.add_noise(item, 0.0, 1)["volumes"], item["volumes"])


def test_slice_removal_keeps_the_same_indices_in_every_series():
    item = study()
    even = perturb.apply("slices_even12", item, "s0")["volumes"]
    keep = np.linspace(0, 15, 12).round().astype(int)
    assert even.shape[1] == 12 and len(set(keep)) == 12
    assert np.array_equal(even, item["volumes"][:, keep])
    assert np.array_equal(perturb.apply("slices_alternate8", item, "s0")["volumes"],
                          item["volumes"][:, ::2])
    assert np.array_equal(perturb.apply("slices_central12", item, "s0")["volumes"],
                          item["volumes"][:, 2:14])
    with pytest.raises(ValueError, match="ascending"):
        perturb.drop_slices(item, [3, 1])


def test_apply_refuses_the_wrong_depth():
    item = study()
    item["volumes"] = item["volumes"][:, :8]
    with pytest.raises(ValueError, match="16"):
        perturb.apply("identity", item, "s0")


def long_frame(values_by_variant, studies):
    rows = []
    for name, values in values_by_variant.items():
        for uid, p in zip(studies, values):
            rows.append({config.ID_COL: uid, "variant": name, **dict(zip(config.LABELS, p))})
    return pd.DataFrame(rows)


def test_summary_is_zero_on_identical_predictions_and_realigns_the_flip():
    rng = np.random.default_rng(1)
    studies = [f"s{i}" for i in range(30)]
    base = rng.uniform(0.05, 0.95, (30, len(config.LABELS)))
    # A flip pass that is exactly the base once re-permuted: a perfectly
    # consistent model. Unaligned comparison would show large deltas.
    flip_raw = base[:, augment.lateral_swap_order()]
    frame = long_frame({"identity": base, "shift_+5": base, "flip_lr": flip_raw}, studies[::-1])
    reference = pd.DataFrame(base, columns=config.LABELS).assign(**{config.ID_COL: studies[::-1]})
    summary = perturb.summarise(frame, reference)
    assert summary["studies"] == 30 and summary["reproduction"]["passed"]
    assert summary["declared"]["labels_used"] is False
    rows = pd.DataFrame(summary["rows"])
    assert set(rows.variant) == {"shift_+5", "flip_lr"} and len(rows) == 24
    assert (rows.p95_abs_dp == 0).all() and (rows.mean_dlogit == 0).all()
    assert np.allclose(rows.spearman, 1.0) and not rows.sensitive.any()
    notes = rows[rows.note != ""]
    assert list(zip(notes.variant, notes.finding)) == [("flip_lr", "MCL")]


def test_summary_flags_a_shift_above_the_floor_and_fails_reproduction():
    studies = [f"s{i}" for i in range(20)]
    base = np.full((20, len(config.LABELS)), 0.5)
    moved = base.copy()
    moved[:, 0] = 0.7
    frame = long_frame({"identity": base, "noise_5": moved}, studies)
    reference = pd.DataFrame(base + 0.02, columns=config.LABELS).assign(**{config.ID_COL: studies})
    summary = perturb.summarise(frame, reference)
    assert summary["reproduction"]["passed"] is False
    assert summary["floor_p95_abs_dp"]["ACL"] == pytest.approx(0.02)
    rows = pd.DataFrame(summary["rows"]).set_index("finding")
    assert rows.loc["ACL", "sensitive"] and rows.loc["ACL", "mean_dlogit"] > 0
    assert rows.loc["ACL", "dlogit_95ci"][0] > 0
    assert not rows.drop("ACL").sensitive.any()


def test_summary_rejects_a_variant_missing_a_study():
    studies = ["a", "b"]
    base = np.full((2, len(config.LABELS)), 0.5)
    frame = long_frame({"identity": base, "shift_+5": base[:1]}, studies)
    with pytest.raises(ValueError, match="exactly once"):
        perturb.summarise(frame)
