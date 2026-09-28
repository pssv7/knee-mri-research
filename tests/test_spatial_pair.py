"""Staging checks for the spatial screening pair: one variable, 24 slices, A100."""

import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.experiments import get_experiment
from tools.kaggle_training_pair_body import validate_contracts
from tools.prepare_kaggle_training_pair import ROOT, prepare, training_source

ARMS = ("spatial_control", "spatial")


def notebook():
    return json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())


@pytest.fixture
def staged(tmp_path):
    """Stage with the real notebook and labels, into a throwaway output tree."""
    for path in ("notebooks", "data", "tools", "outputs/kaggle_sources"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    prepare(root=tmp_path, candidate="spatial", fold=1, no_timeout=True)
    return tmp_path / "outputs/kaggle_spatial_fold1"


@pytest.mark.parametrize("fold", [1, 3])
@pytest.mark.parametrize("arm", ARMS)
def test_both_arms_are_the_same_24_slice_protocol_recipe(arm, fold):
    source = training_source(notebook(), arm, "hash", fold=fold)
    ast.parse(source)
    assert "MATCH_SPATIAL = True" in source
    assert "EXPECTED_SLICES = 24" in source
    assert "CONTROL_EXPERIMENT = 'spatial_control'" in source
    assert "/kaggle/working/spatial_control_fold" in source
    assert "spatial_encoder=RUN.spatial_encoder" in source
    assert "('model.spatial_encoder.',)" in source
    assert f"os.environ['KNEE_EXPERIMENT'] = {arm!r}" in ast.unparse(ast.parse(source))


def test_the_control_is_the_24_slice_model_not_the_16_slice_incumbent(staged):
    source = (staged / "train_pair.py").read_text()
    tree = ast.parse(source)
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("TRAINING_SCRIPTS", "CANDIDATE", "CONTROL", "FOLD")}
    assert tuple(values["TRAINING_SCRIPTS"]) == ARMS
    assert (values["CONTROL"], values["CANDIDATE"], values["FOLD"]) == (*ARMS, 1)
    for arm, script in values["TRAINING_SCRIPTS"].items():
        assert f"/kaggle/working/{arm}_fold1" in script
    receipt = json.loads((staged / "staging_receipt.json").read_text())
    assert receipt["kernel"] == "pssv27/knee-spatial-fold1-pair"


def test_the_screening_contract_permits_only_the_3d_path_to_differ():
    contracts = [{"experiment": get_experiment(arm).to_dict(), "seed": 0,
                  "initial_weights_sha256": arm,
                  "shared_initial_weights_sha256": "same"} for arm in ARMS]
    validate_contracts(contracts, "spatial", "spatial_control")

    broken = json.loads(json.dumps(contracts))
    broken[0]["experiment"]["slices"] = 16
    with pytest.raises(ValueError, match="24 slices"):
        validate_contracts(broken, "spatial", "spatial_control")

    broken = json.loads(json.dumps(contracts))
    broken[0]["experiment"]["series_selection"] = "uniform"
    with pytest.raises(ValueError, match="protocol selection"):
        validate_contracts(broken, "spatial", "spatial_control")

    broken = json.loads(json.dumps(contracts))
    broken[1]["experiment"]["plane_adapters"] = True
    with pytest.raises(ValueError, match="separate ablation"):
        validate_contracts(broken, "spatial", "spatial_control")

    broken = json.loads(json.dumps(contracts))
    broken[1]["experiment"]["max_series"] = 7
    with pytest.raises(ValueError, match="registered experiment variable"):
        validate_contracts(broken, "spatial", "spatial_control")

    broken = json.loads(json.dumps(contracts))
    broken[1]["shared_initial_weights_sha256"] = "drift"
    with pytest.raises(ValueError, match="initial weights"):
        validate_contracts(broken, "spatial", "spatial_control")

    with pytest.raises(ValueError, match="experiment ordering"):
        validate_contracts(contracts, "spatial")


def test_spatial_staging_refuses_to_be_combined_with_another_change():
    for kwargs in ({"match_classwise": True}, {"match_backbone": True},
                   {"match_slice": True}, {"complete_protocol": True}):
        with pytest.raises(ValueError):
            training_source(notebook(), "spatial", "hash", fold=1, **kwargs)
    with pytest.raises(ValueError, match="Spatial comparison"):
        training_source(notebook(), "baseline", "hash", fold=1, match_spatial=True)
    with pytest.raises(ValueError, match="screening folds"):
        training_source(notebook(), "spatial", "hash", fold=2)
    with pytest.raises(ValueError, match="control arm"):
        prepare(root=ROOT, candidate="spatial_control", fold=1)


def test_the_colab_notebook_states_the_gate_and_starts_no_session(tmp_path):
    from tools.prepare_colab_spatial_pair import build
    for path in ("notebooks", "data", "tools", "outputs/kaggle_sources"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    (tmp_path / "notebooks").unlink()
    (tmp_path / "notebooks").mkdir()
    for name in ("08_finetune.ipynb",):
        (tmp_path / "notebooks" / name).symlink_to(ROOT / "notebooks" / name)

    destination = build(root=tmp_path, fold=1)
    written = json.loads(destination.read_text())
    assert written["metadata"]["colab"]["gpuType"] == "A100"
    sources = ["".join(cell["source"]) for cell in written["cells"]]
    assert any("+0.005" in text and "-0.020" in text for text in sources)
    assert any("PAIR_SOURCE = " in text for text in sources)
    assert not any("kaggle kernels push" in text for text in sources)
    for cell in written["cells"]:
        if cell["cell_type"] == "code":
            body = "\n".join(line for line in "".join(cell["source"]).splitlines()
                             if not line.lstrip().startswith(("%", "!")))
            ast.parse(body)


PROBE_LAUNCHER = """import os, pathlib, runpy, sys
# Google Drive does not bump a directory's mtime when an entry is created in
# it. Reproduce that: every mkdir/write under the probe restores the cwd's
# mtime, so Python's cached finder for the cwd never notices `src/` appearing.
cwd = pathlib.Path.cwd()
stat = os.stat(cwd)
def freeze():
    os.utime(cwd, ns=(stat.st_atime_ns, stat.st_mtime_ns))
mkdir, write_text = pathlib.Path.mkdir, pathlib.Path.write_text
def frozen_mkdir(self, *args, **kwargs):
    mkdir(self, *args, **kwargs); freeze()
def frozen_write_text(self, *args, **kwargs):
    result = write_text(self, *args, **kwargs); freeze(); return result
pathlib.Path.mkdir, pathlib.Path.write_text = frozen_mkdir, frozen_write_text
script = str(cwd / "spatial_smoke.py")
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
"""


def test_the_representative_probe_imports_its_sources_from_a_drive_like_cwd(tmp_path):
    """The first live Colab run died on "No module named 'src'" at this point."""
    import subprocess
    from tools.prepare_colab_spatial_pair import build
    root = tmp_path / "root"
    for path in ("notebooks", "data", "tools", "outputs/kaggle_sources"):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    (root / "notebooks").unlink()
    (root / "notebooks").mkdir()
    (root / "notebooks/08_finetune.ipynb").symlink_to(ROOT / "notebooks/08_finetune.ipynb")
    written = json.loads(build(root=root, fold=1).read_text())
    cell = next("".join(c["source"]) for c in written["cells"]
                if "REPRESENTATIVE_SOURCE = " in "".join(c["source"]))
    source = next(ast.literal_eval(node.value) for node in ast.parse(cell).body
                  if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], "id", None) == "REPRESENTATIVE_SOURCE")
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "spatial_smoke.py").write_text(source)
    (probe / "run_probe.py").write_text(PROBE_LAUNCHER)
    result = subprocess.run([sys.executable, "-u", "run_probe.py"], cwd=probe,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--full-series" in result.stdout
