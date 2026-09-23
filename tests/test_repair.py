import pytest

from src.labels import repair

# Synthetic sentences written for these tests, not report text.


@pytest.mark.parametrize("report", [
    "Grade III chondromalacia of the patella.",
    "Retropatellare Chondromalazie Grad II.",
    "Hondromalacija patele II. stupnja.",
    "Patellar kıkırdakta kondromalazi izlendi.",
    "Condromalacia rotuliana grado III.",
    "Fissuring of the trochlear cartilage.",
    "Patellofemoral osteoarthritis with osteophytes.",
    "Cartilage defect of the lateral patellar facet.",
])
def test_present_readings(report):
    assert repair.read_pf_oa(report)[0] == "present"


@pytest.mark.parametrize("report, state", [
    ("No chondromalacia of the patella.", "negated"),
    ("Patellar cartilage without defects.", "negated"),
    ("Possible early chondromalacia of the patella.", "hedged"),
    ("Grade I chondromalacia of the patella.", "below_threshold"),
    ("Chondromalacia patellae grado 1.", "below_threshold"),
    ("Mild signal change of the patellar cartilage.", "below_threshold"),
    ("Medial tibiofemoral osteoarthritis.", None),
    ("The patella is normally positioned.", None),
    ("Normal patellar cartilage.", None),
])
def test_non_present_readings(report, state):
    assert repair.read_pf_oa(report)[0] == state


def test_grade_one_to_two_is_not_below_threshold():
    assert repair.read_pf_oa("Chondromalacia of the patella grade I-II.")[0] == "present"


def test_proposals_only_raise_low_targets():
    report = "Grade III chondromalacia of the patella."
    assert repair.propose(report, 0.01)["mechanism"] == "missed_mention"
    assert repair.propose(report, 0.42)["mechanism"] == "severity_definition"
    assert repair.propose(report, 0.8) is None
    assert repair.propose("No chondromalacia of the patella.", 0.01) is None


def test_clause_boundaries_keep_compartments_apart():
    report = "Medial femoral condyle cartilage defect. Patella normal."
    assert repair.read_pf_oa(report)[0] is None
