import numpy as np
import pandas as pd
import pytest

from src import config
from src.labels import audit_sampling as s

ID = config.ID_COL
SMALL = s.Settings(reserve_per_finding=8, controls=10, per_stratum=6)


def make_pairs(n=300, seed=1):
    rng = np.random.default_rng(seed)
    uids = [f"u{i:04}" for i in range(n)]
    folds = pd.DataFrame({ID: uids, "fold": rng.integers(0, 5, n)})
    labels = pd.DataFrame({ID: uids})
    oof = pd.DataFrame(index=uids)
    for f in s.PRIORITY:
        labels[f] = rng.random(n)
        labels[f + "_weight"] = rng.choice([1.0, 0.7, 0.4], n)
        oof[f] = rng.random(n)
    languages = pd.Series(rng.choice(["en", "el", "de", "tr"], n, p=[.5, .2, .2, .1]), index=uids)
    return {f: s.finding_pairs(labels, folds, oof, languages, f) for f in s.PRIORITY}


@pytest.fixture
def drawn():
    pairs = make_pairs()
    exposed = {f"u{i:04}" for i in range(0, 300, 7)}
    prior = {(uid, "Contusion"): f"R{i}" for i, uid in enumerate(sorted(exposed))}
    reserve, reserve_short = s.draw_reserve(pairs, exposed, SMALL)
    sample, short, population = s.draw_audit(pairs, set(reserve[ID]), prior, SMALL)
    return pairs, exposed, prior, reserve, sample, short, population


def test_reserve_is_fresh_discovery_and_split_by_target(drawn):
    pairs, exposed, _, reserve, *_ = drawn
    assert not set(reserve[ID]) & exposed
    assert set(reserve.fold) <= set(SMALL.discovery_folds)
    for f in s.PRIORITY:
        part = reserve[reserve.finding == f]
        assert len(part) == 8
        assert (part[part.stratum == "target_positive"].target >= .5).all()
        assert (part[part.stratum == "target_negative"].target < .5).all()


def test_reserved_studies_leave_discovery_for_every_finding(drawn):
    _, _, _, reserve, sample, *_ = drawn
    assert not set(sample[ID]) & set(reserve[ID])


def test_controls_are_uniform_and_carry_their_probability(drawn):
    _, _, _, _, sample, _, population = drawn
    for f in s.PRIORITY:
        controls = sample[(sample.finding == f) & (sample.role == "control")]
        assert len(controls) == 10
        assert np.allclose(controls.inclusion_probability, 10 / population[f])
        assert (controls.provenance == "random_draw").all()


def test_control_suspect_overlaps_stay_controls(drawn):
    _, _, _, _, sample, *_ = drawn
    controls = sample[sample.role == "control"]
    assert {"meets_false_positive", "meets_missed_positive"} <= set(controls.columns)
    enriched = sample[sample.role != "control"]
    assert not set(zip(controls[ID], controls.finding)) & set(zip(enriched[ID], enriched.finding))


def test_strata_meet_their_criteria(drawn):
    _, _, _, _, sample, *_ = drawn
    fp = sample[sample.role == "suspected_false_positive"]
    miss = sample[sample.role == "suspected_missed_positive"]
    assert ((fp.target >= .5) & (fp.oof <= .3)).all()
    assert ((miss.target < .5) & (miss.oof >= .7)).all()


def test_prior_reviews_fill_enriched_quotas_first_and_are_marked(drawn):
    _, _, prior, _, sample, *_ = drawn
    marked = sample[sample.provenance == "label_review_v1"]
    assert len(marked) and (marked.finding == "Contusion").all()
    assert marked.inclusion_probability.isna().all()
    assert all((u, f) in prior for u, f in zip(marked[ID], marked.finding))
    assert (marked.role != "control").all()


def test_prior_reviews_count_as_controls_only_when_drawn():
    pairs = make_pairs()
    everything = {(u, "Contusion"): "R" for u in pairs["Contusion"][ID]}
    sample, *_ = s.draw_audit(pairs, set(), everything, SMALL)
    controls = sample[(sample.finding == "Contusion") & (sample.role == "control")]
    assert (controls.provenance == "random_draw").all() and controls.inclusion_probability.notna().all()


def test_shortfall_is_recorded_not_filled():
    pairs = make_pairs(n=40)
    sample, short, _ = s.draw_audit(pairs, set(), {}, s.Settings(controls=10, per_stratum=50))
    assert short and all(item["available"] < item["wanted"] for item in short)
    assert not sample.duplicated([ID, "finding"]).any()


def test_draws_are_deterministic_and_per_finding_independent():
    a = make_pairs()
    first = s.draw_audit(a, set(), {}, SMALL)[0]
    again = s.draw_audit(make_pairs(), set(), {}, SMALL)[0]
    pd.testing.assert_frame_equal(first, again)
    # Changing one finding's data leaves the others' draws untouched.
    b = make_pairs()
    b["PF OA"] = b["PF OA"].assign(target=1 - b["PF OA"].target)
    other = s.draw_audit(b, set(), {}, SMALL)[0]
    for f in ("Lateral Meniscus", "Contusion"):
        pd.testing.assert_frame_equal(first[first.finding == f].reset_index(drop=True),
                                      other[other.finding == f].reset_index(drop=True))


def test_proportional_allocation():
    counts = pd.Series({"en": 50, "el": 30, "tr": 1})
    alloc = s.proportional(counts, 10)
    assert alloc.sum() == 10 and (alloc <= counts).all()
    assert (s.proportional(pd.Series({"en": 3, "el": 2}), 10) == pd.Series({"en": 3, "el": 2})).all()


def test_missing_oof_is_refused():
    pairs = make_pairs(n=10)
    labels = pd.DataFrame({ID: ["a"], "ACL": [1.0], "ACL_weight": [1.0]})
    with pytest.raises(ValueError, match="lack protocol OOF"):
        s.finding_pairs(labels, pd.DataFrame({ID: ["a"], "fold": [0]}),
                        pd.DataFrame({"ACL": [.5]}, index=["b"]), pd.Series(dtype=str), "ACL")
