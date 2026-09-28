import json

import numpy as np
import pandas as pd
import pytest

from tools import prepare_combined_submission as cs

REPORT15 = {"members": 15, "fallbacks": 0}
REPORT5 = {"members": 5, "fallbacks": 0}


def frame(values, ids=("a", "b", "c")):
    return pd.DataFrame({"StudyInstanceUID": list(ids), "ACL": values[0], "MCL": values[1]})


def test_blend_weights_each_model_equally():
    p15 = frame([[0.2, 0.4, 0.6], [0.1, 0.1, 0.1]])
    p5 = frame([[0.6, 0.8, 1.0], [0.5, 0.5, 0.5]])
    out = cs.blend({"part15": p15, "part384": p5}, {"part15": REPORT15, "part384": REPORT5})
    assert np.allclose(out.ACL, 0.75 * p15.ACL + 0.25 * p5.ACL)
    assert out.StudyInstanceUID.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("change, message", [
    ({"part384": {"members": 4, "fallbacks": 0}}, "members"),
    ({"part15": {"members": 15, "fallbacks": 2}}, "fallback"),
])
def test_blend_refuses_bad_reports(change, message):
    reports = {"part15": REPORT15, "part384": REPORT5, **change}
    with pytest.raises(ValueError, match=message):
        cs.blend({"part15": frame([[.1] * 3, [.1] * 3]), "part384": frame([[.1] * 3, [.1] * 3])}, reports)


def test_blend_refuses_misaligned_studies_and_bad_values():
    good = frame([[.1] * 3, [.1] * 3])
    with pytest.raises(ValueError, match="order"):
        cs.blend({"part15": good, "part384": frame([[.1] * 3, [.1] * 3], ids=("b", "a", "c"))},
                 {"part15": REPORT15, "part384": REPORT5})
    with pytest.raises(ValueError, match="outside"):
        cs.blend({"part15": good, "part384": frame([[1.5] * 3, [.1] * 3])}, {"part15": REPORT15, "part384": REPORT5})


def test_each_part_changes_only_manifest_selection_and_output_paths():
    for name in cs.PARTS:
        script, manifest_hash = cs.part_script(name)
        original = "\n\n".join(cs.code_cells(cs.ROOT / cs.PARTS[name]["staged"] / "submission.ipynb"))
        restored = script.replace(f"/kaggle/working/{name}/", "/kaggle/working/")
        restored = restored.replace(
            'manifests = [p for p in attached_files("ensemble.json")\n'
            f"             if hashlib.sha256(p.read_bytes()).hexdigest() == {manifest_hash!r}]",
            'manifests = attached_files("ensemble.json")')
        assert restored == original
        assert "/kaggle/working/submission.csv" not in script


def test_staged_kernel_compiles_and_attaches_both_model_datasets():
    out, receipt = cs.prepare()
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert {"pssv27/knee-ensemble-protocol-private", "pssv27/knee-resolution384-five-fold-private"} <= set(meta["dataset_sources"])
    assert meta["is_private"] and not meta["enable_internet"]
    assert receipt["projected_1300_study_hours"] < 9
    source = "".join(json.loads((out / "submission.ipynb").read_text())["cells"][1]["source"])
    compile(source, "kernel", "exec")
