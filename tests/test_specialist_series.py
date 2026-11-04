import pandas as pd

from src.model.specialist import select_specialist_series


def rows():
    return pd.DataFrame([
        {"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0, "Fat_Suppression": 0},
        {"Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        {"Anatomical_Plane": "Axial", "Fluid_Sensitive": 0, "Fat_Suppression": 0},
        {"Anatomical_Plane": "Axial", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        {"Anatomical_Plane": "Coronal", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        {"Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 0, "Fat_Suppression": 0},
    ])


def test_synovitis_prefers_fluid_axial_sagittal_and_preserves_planes():
    selected = rows().iloc[select_specialist_series(rows(), 3, "synovitis")]
    assert set(selected.Anatomical_Plane) == {"Axial", "Sagittal", "Coronal"}
    assert selected.Fluid_Sensitive.eq(1).all()


def test_pfoa_prioritises_axial_and_keeps_sagittal_context():
    selected = rows().iloc[select_specialist_series(rows(), 3, "pfoa")]
    assert (selected.Anatomical_Plane == "Axial").sum() == 2
    assert (selected.Anatomical_Plane == "Sagittal").sum() == 1
