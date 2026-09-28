"""Completion jobs require successful screens and unchanged training inputs."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.experiments import get_experiment
from tools.prepare_kaggle_protocol_completion import prepare


@pytest.fixture
def staged_root(tmp_path):
    def put(path, data):
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(data)
    notebook = (ROOT / "notebooks/08_finetune.ipynb").read_text()
    put("notebooks/08_finetune.ipynb", notebook)
    put("data/meta/train_labels.parquet", "fixture-labels")
    for fold in (1, 3):
        directory = f"outputs/kaggle_protocol_fold{fold}_results"
        put(f"{directory}/comparison.json", json.dumps({"folds": [fold],
            "screening_gate": {"passed": True}}))
        put(f"{directory}/pair_progress.json", json.dumps({"state": "complete",
            "labels_sha256": hashlib.sha256(b"fixture-labels").hexdigest()}))
    put("outputs/kaggle_protocol_fold3/staging_receipt.json", json.dumps({
        "notebook_sha256": hashlib.sha256(notebook.encode()).hexdigest()}))
    put("outputs/kaggle_protocol_fold3_results/protocol_fold3/run_contract.json",
        json.dumps({"experiment": get_experiment("protocol").to_dict()}))
    put("outputs/kaggle_sources/knee-finetune/kernel-metadata.json", json.dumps({
        "competition_sources": ["fixture-competition"], "kernel_sources": ["fixture-cache"],
        "dataset_sources": ["fixture-labels"], "docker_image": "fixture-pinned-image"}))
    return tmp_path


def test_stages_only_requested_protocol_fold_without_baseline_training(staged_root):
    prepare(staged_root, folds=(2,))
    output = staged_root / "outputs/kaggle_protocol_fold2_completion"
    tree = ast.parse((output / "train.py").read_text())
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert values["FOLD"] == 2 and values["VALIDATION_STUDIES"] == 870
    assert set(values["SCRIPTS"]) == {"ddp_smoke", "protocol_fold2"}
    assert all("REFERENCE_CONTRACT = " in source for source in values["SCRIPTS"].values())
    assert "DDP_SMOKE_PASSED" in values["SCRIPTS"]["ddp_smoke"]
    assert "oof.to_parquet" in values["SCRIPTS"]["protocol_fold2"]
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert metadata["docker_image"] == "fixture-pinned-image"


def test_fold4_has_no_background_stack_dump_but_keeps_parent_time_limits(staged_root):
    import types

    prepare(staged_root, folds=(4,))
    tree = ast.parse((staged_root / "outputs/kaggle_protocol_fold4_completion/train.py").read_text())
    scripts = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and node.targets[0].id == "SCRIPTS")
    for source in scripts.values():
        calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        diagnostics = [node for node in calls if ast.unparse(node.func).startswith("faulthandler.")]
        enabled = []
        handler = types.SimpleNamespace(enable=lambda: enabled.append(True))
        # Execute the generated diagnostic calls: an accidental timer or thread
        # dump fails here instead of first surfacing during a paid GPU run.
        for call in diagnostics:
            exec(compile(ast.Expression(call), "staged_diagnostics", "eval"),
                 {"faulthandler": handler})
        assert enabled == [True]
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "subprocess.run")
    timeout = next(keyword.value for keyword in run.keywords if keyword.arg == "timeout")
    for name, seconds in (("ddp_smoke", 900), ("protocol_fold4", 18000)):
        assert eval(compile(ast.Expression(timeout), "runner_timeout", "eval"), {"name": name}) == seconds


@pytest.mark.parametrize("problem", ["gate", "labels", "notebook", "variant"])
def test_rejects_changed_or_failed_screening(staged_root, problem):
    if problem == "gate":
        path = staged_root / "outputs/kaggle_protocol_fold1_results/comparison.json"
        data = json.loads(path.read_text())
        data["screening_gate"]["passed"] = False
        path.write_text(json.dumps(data))
    elif problem == "labels":
        (staged_root / "data/meta/train_labels.parquet").write_text("different")
    elif problem == "notebook":
        path = staged_root / "notebooks/08_finetune.ipynb"
        path.write_text(path.read_text() + "\n")
    else:
        path = staged_root / "outputs/kaggle_protocol_fold3_results/protocol_fold3/run_contract.json"
        path.write_text(json.dumps({"experiment": get_experiment("classwise").to_dict()}))
    with pytest.raises(ValueError):
        prepare(staged_root)
