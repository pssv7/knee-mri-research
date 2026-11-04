import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.labels.evaluation_sampling import eligible_representatives, select_sample
from src.labels.review_packet import blank_annotations, render
from src.validation.proxy import ID, LABELS
from tools.prepare_report_label_evaluation import make_packet, exposure_inventory
from tools.validate_report_evaluation import annotation_status, validate_packet


def settings():
    return {"seed": 23, "random_controls": 12, "priority_findings": ["PF OA", "Contusion"],
            "priority_positive_per_finding": 2, "priority_nonpositive_per_finding": 2,
            "coverage_cases": 5, "minimum_old_positive_per_finding": 2, "minimum_per_language_heuristic": 2}


def population(n=100):
    reps = pd.DataFrame({ID: [f"uid-{i}" for i in range(n)], "component": [f"component-{i}" for i in range(n)]})
    calls = pd.DataFrame({f: [float((i+j) % 4 == 0) for i in range(n)] for j,f in enumerate(LABELS)}, index=reps[ID])
    languages = pd.Series(["en" if i % 5 else "es" for i in range(n)], index=reps[ID])
    return reps, calls, languages


def test_sampling_reproducible_and_controls_not_changed_by_enrichment():
    reps, calls, languages = population()
    a, _ = select_sample(reps, calls, languages, settings())
    b, _ = select_sample(reps.sample(frac=1, random_state=1), calls.iloc[::-1], languages.iloc[::-1], settings())
    assert a.to_dict("records") == b.to_dict("records")
    changed = settings(); changed["priority_findings"] = list(reversed(changed["priority_findings"]))
    c, _ = select_sample(reps, calls, languages, changed)
    assert list(a.loc[a.cohort == "random_control", ID]) == list(c.loc[c.cohort == "random_control", ID])
    assert len(a) == 25 and a.component.nunique() == 25
    assert a.loc[a.cohort == "random_control", "conditional_draw_probability"].eq(.12).all()


def test_protected_sibling_excludes_whole_component():
    reports = pd.DataFrame({ID: ["gold", "sibling", "ok", "inner", "outer"]})
    components = reports.assign(component=["linked", "linked", "ok", "inner", "outer"])
    roles = components.iloc[1:].copy().assign(role=["train", "train", "selection", "outer"])
    result = eligible_representatives(reports, roles, components, {"gold"}, 7)
    assert list(result[ID]) == ["ok"]
    with pytest.raises(ValueError, match="Unknown"):
        eligible_representatives(reports, roles, components, {"unlogged"}, 7)
    bad = roles.copy(); bad.loc[bad[ID] == "inner", "component"] = "linked"
    with pytest.raises(ValueError, match="disagree"):
        eligible_representatives(reports, bad, components, {"gold"}, 7)


def test_component_split_leakage_rejected():
    reports = pd.DataFrame({ID: ["a", "b", "c"]})
    components = reports.assign(component=["same", "same", "other"])
    roles = components.assign(role=["train", "selection", "outer"])
    with pytest.raises(ValueError, match="crosses"):
        eligible_representatives(reports, roles, components, set(), 7)


def test_rare_positive_shortfall_is_recorded_not_fabricated():
    reps, calls, languages = population()
    calls["Fracture"] = 0
    result, shortfalls = select_sample(reps, calls, languages, settings())
    assert len(result) == 25
    assert any(r["reason"] == "final_old_positive_coverage:Fracture" and r["available"] == 0 for r in shortfalls)
    small = settings(); small["random_controls"] = 101
    with pytest.raises(ValueError, match="Insufficient"):
        select_sample(reps, calls, languages, small)


def packet():
    reps, _, _ = population(2)
    reps["cohort"] = "PRIVATE_STRATUM_CANARY"
    reports = pd.DataFrame({"Report": ["The ACL is intact.", "No joint effusion. </script><script>bad()</script>"]}, index=reps[ID])
    return make_packet(reps, reports, "Fixed rubric", "fresh", "Review", 1)


def test_blind_packet_drops_all_custodian_metadata_and_escapes_report_html():
    p, keys = packet()
    serialized = json.dumps(p)
    assert "PRIVATE_STRATUM_CANARY" not in serialized and "uid-" not in serialized and "component-" not in serialized
    assert all(row["cohort"] == "PRIVATE_STRATUM_CANARY" for row in keys)
    validate_packet(p)
    template = (Path(__file__).parents[1] / "tools/report_review_template.html").read_text()
    page = render(p, template)
    assert "</script><script>bad()" not in page
    assert "PRIVATE_STRATUM_CANARY" not in page and "uid-" not in page
    assert "__PAYLOAD__" not in page and "__TITLE__" not in page


def test_blank_is_not_a_completed_reference_and_full_ai_annotations_still_not_truth():
    p, _ = packet()
    blank = blank_annotations(p)
    assert annotation_status(p, blank)["structurally_complete_reports"] == 0
    for row in blank["rows"]:
        row["report_states"] = "unmentioned"
    blank["reviewer"] = "test AI; intentionally not an adjudicated reference"
    result = annotation_status(p, blank)
    assert result["status"] == "complete_unverified_ai_annotations"
    assert not result["independent_accuracy_established"]


def test_changed_packet_or_dropped_or_duplicate_annotation_is_rejected():
    p, _ = packet()
    altered = copy.deepcopy(p); altered["cases"][0]["report"] += " changed"
    with pytest.raises(ValueError, match="hash"):
        validate_packet(altered)
    blank = blank_annotations(p)
    blank["rows"].pop()
    with pytest.raises(ValueError, match="Missing"):
        annotation_status(p, blank)
    blank["rows"].append(blank["rows"][0])
    with pytest.raises(ValueError, match="duplicated"):
        annotation_status(p, blank)


def test_missing_exposure_inventory_blocks_freshness_claim(tmp_path):
    with pytest.raises(ValueError, match="inventory"):
        exposure_inventory(tmp_path)
