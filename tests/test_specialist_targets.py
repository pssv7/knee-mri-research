import pandas as pd
import pytest

from src import config
from src.labels.specialist_targets import (
    PFOA_GROUP,
    SYNOVITIS_GROUP,
    assertion_target,
    build_specialist_targets,
    severity_band,
)


def item(state="unmentioned", certainty="", norm="", text=""):
    return {
        "report_states": state,
        "certainty": certainty,
        "severity_norm": norm,
        "severity_text": text,
    }


def inventory(**overrides):
    values = {name: item() for name in config.LABELS}
    values.update(overrides)
    return values


def base(uid="s1"):
    row = {config.ID_COL: uid}
    for name in config.LABELS:
        row[name] = 0.1
        row[name + "_weight"] = 0.7
    return pd.DataFrame([row])


def test_severity_preserves_minimal_mild_moderate_and_severe():
    assert severity_band(item("present", "definite", "minimal", "trace")) == "minimal"
    assert severity_band(item("present", "definite", "minimal", "mild")) == "mild"
    assert severity_band(item("present", "definite", "moderate", "")) == "moderate"
    assert severity_band(item("present", "definite", "severe", "")) == "severe"


def test_uncertain_synovitis_is_masked_and_pf_below_threshold_is_soft():
    assert assertion_target("Synovitis", item("uncertain", "possible")) == (0.5, False)
    assert assertion_target("Synovitis", item("unmentioned")) == (0.5, False)
    assert assertion_target("PF OA", item("present_below_threshold", "definite")) == (0.3, True)


def test_specialist_masks_other_labels_and_uses_agreement_for_weight():
    assertions = {"s1": inventory(**{
        "Synovitis": item("present", "definite", "moderate", "moderate"),
        "Effusion": item("negated", "definite", "not stated", ""),
        "Baker's": item("unmentioned"),
    })}
    labels = base()
    labels.loc[0, "Synovitis"] = 0.9
    labels.loc[0, "Effusion"] = 0.9  # Deliberate disagreement.
    targets, audit = build_specialist_targets(assertions, labels, SYNOVITIS_GROUP)
    row = targets.iloc[0]
    assert row["Synovitis_weight"] == 1.0
    assert row["Effusion_weight"] == 0.4
    assert row["Baker's_weight"] == 0.0
    assert row["ACL"] == 0.5 and row["ACL_weight"] == 0.0
    assert set(audit.agreement) == {"agree", "disagree", "masked"}


def test_pf_group_has_only_three_active_heads():
    assertions = {"s1": inventory(**{
        name: item("present", "definite", "minimal", "mild")
        for name in PFOA_GROUP
    })}
    targets, _ = build_specialist_targets(assertions, base(), PFOA_GROUP)
    for name in PFOA_GROUP:
        assert targets.at[0, name + "_weight"] > 0
    for name in set(config.LABELS) - set(PFOA_GROUP):
        assert targets.at[0, name + "_weight"] == 0


def test_rejects_incomplete_assertion_inventory():
    with pytest.raises(ValueError, match="Incomplete"):
        build_specialist_targets({"s1": {"Synovitis": item()}}, base(), SYNOVITIS_GROUP)
