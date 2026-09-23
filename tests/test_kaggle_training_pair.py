"""Check isolation and compute guards before spending remote training time."""

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_kaggle_training_pair import INPUT_SEARCH, ROOT, training_source
from tools.kaggle_training_pair_body import validate_contracts


@pytest.mark.parametrize("experiment", ["baseline", "protocol", "classwise", "convnextv2"])
@pytest.mark.parametrize("fold", [1, 3])
def test_staged_training_preserves_modules_and_isolates_artifacts(experiment, fold):
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    source = training_source(notebook, experiment, "fixture-label-hash", fold=fold)
    tree = ast.parse(source)
    original = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    def files(tree):
        return next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "FILES")
    assert files(tree) == files(ast.parse(original[1]))
    paths = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and n.value.startswith("/kaggle/working")]
    assert paths and all(p.startswith(f"/kaggle/working/{experiment}_fold{fold}")
                         or p == f"/kaggle/working/baseline_fold{fold}/run_contract.json"
                         for p in paths)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "FOLD"]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == fold
    assert source.index("Matched-run preflight passed") < source.index("trainer.fit(")
    assert "fixture-label-hash" in source
    assert "KNEE_EXPERIMENT" in source
    assert "BASELINE_CONDITIONAL" not in source
    assert "oof.to_parquet" in source and "artifacts.write_metadata" in source
    assert "!pip install" not in source
    assert "STRATEGY = 'ddp' if DEVICES > 1 else 'auto'" in source
    assert source.index("if not trainer.is_global_zero:") < source.index("n_tensors = knee_lightning.export_from_checkpoint")


def test_staging_refuses_unreviewed_experiments_or_notebook_structure():
    with pytest.raises(ValueError, match="screening folds"):
        training_source({}, "baseline", "hash", fold=2)
    with pytest.raises(ValueError, match="recovery checkpoint"):
        training_source({}, "baseline", "hash", recovery={"fixture": True}, fold=3)
    with pytest.raises(ValueError, match="authorized"):
        training_source({}, "resample", "hash")
    with pytest.raises(ValueError, match="structure"):
        training_source({"cells": []}, "baseline", "hash")


def test_recovery_cannot_retrain_and_smoke_checks_end_of_fit_exports():
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    recovery = training_source(notebook, "baseline", "label-hash", recovery={
        "checkpoint_sha256": "checkpoint-hash", "contract": {}})
    tree = ast.parse(recovery)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "trainer.fit" not in calls
    assert "checkpoint-hash" in recovery and "Recovered baseline settings changed" in recovery
    assert "oof.to_parquet" in recovery
    smoke = training_source(notebook, "protocol", "label-hash", smoke=True)
    ast.parse(smoke)
    assert "limit_train_batches=8" in smoke and "max_epochs=1" in smoke
    assert "DDP_SMOKE_PASSED" in smoke and "oof.to_parquet" not in smoke


def test_classwise_pair_checks_shared_initialization_without_changing_input_selection():
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    for experiment in ("baseline", "classwise"):
        source = training_source(notebook, experiment, "hash", match_classwise=True)
        assert "MATCH_CLASSWISE = True" in source
        assert "shared_initial_weights_sha256" in source
        assert "('protocol' if EXPERIMENT == 'protocol' else 'uniform')" in source
        assert "RUN.classwise_pool == (EXPERIMENT == 'classwise')" in source
        assert "('model.study_pool.', 'model.head.')" in source
    with pytest.raises(ValueError, match="fresh native-DDP baseline"):
        training_source(notebook, "baseline", "hash", recovery={"fixture": True},
                        match_classwise=True)


@pytest.mark.parametrize("candidate", ["protocol", "classwise", "convnextv2"])
def test_contract_guard_rejects_settings_and_initialization_drift(candidate):
    from src.model.experiments import get_experiment
    contracts = [{"experiment": get_experiment(name).to_dict(), "seed": 0,
                  "initial_weights_sha256": "full", "shared_initial_weights_sha256": "shared"}
                 for name in ("baseline", candidate)]
    if candidate == "convnextv2":
        for contract in contracts:
            contract.update(aggregation_seed=0, feature_dim=768, input_defaults={
                "input_mean": None, "input_std": None, "backbone_lr_scale": 1.0})
    original = json.dumps(contracts, sort_keys=True)
    validate_contracts(contracts, candidate)
    assert json.dumps(contracts, sort_keys=True) == original
    contracts[1]["initial_weights_sha256"] = "different"
    if candidate == "protocol":
        with pytest.raises(ValueError, match="initial weights"):
            validate_contracts(contracts, candidate)
        contracts[1]["initial_weights_sha256"] = "full"
    else:
        validate_contracts(contracts, candidate)
    contracts[1]["shared_initial_weights_sha256"] = "different"
    with pytest.raises(ValueError, match="initial weights"):
        validate_contracts(contracts, candidate)
    contracts[1]["shared_initial_weights_sha256"] = "shared"
    contracts[1]["experiment"]["slices"] = 24
    with pytest.raises(ValueError, match="More than"):
        validate_contracts(contracts, candidate)
    contracts[1]["experiment"]["slices"] = 16
    contracts[1]["seed"] = 1
    with pytest.raises(ValueError, match="Training settings|comparison controls"):
        validate_contracts(contracts, candidate)


def test_backbone_adapter_reseeds_between_trunk_and_aggregation():
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    for experiment in ("baseline", "convnextv2"):
        source = training_source(notebook, experiment, "hash", match_backbone=True)
        assert "MATCH_BACKBONE = True" in source
        assert "('model.backbone.',)" in source
        start = source.index("_matched_backbone = net.build_backbone(BACKBONE)")
        seed = source.index("pl.seed_everything(SEED, workers=True)", start)
        module = source.index("module = knee_lightning.KneeLitModule(", start)
        assert start < seed < module
        assert "backbone=_matched_backbone" in source[module:]
        assert "aggregation_seed" in source and "FEATURE_DIM == 768" in source
    with pytest.raises(ValueError, match="fresh baseline"):
        training_source(notebook, "baseline", "hash", recovery={"fixture": True},
                        match_backbone=True)
    with pytest.raises(ValueError, match="cannot include"):
        training_source(notebook, "classwise", "hash", match_backbone=True)


def test_v2_contract_requires_explicit_aggregation_controls():
    from src.model.experiments import get_experiment
    contracts = [{"experiment": get_experiment(n).to_dict(), "seed": 0,
                  "initial_weights_sha256": n, "shared_initial_weights_sha256": "shared"}
                 for n in ("baseline", "convnextv2")]
    with pytest.raises(ValueError, match="comparison controls"):
        validate_contracts(contracts, "convnextv2")


def test_actual_v1_v2_trunks_share_seeded_aggregation_and_feature_shape():
    import torch
    from src.model import net
    from src.model.lightning import KneeLitModule
    from src.model.experiments import get_experiment
    states = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for name in ("baseline", "convnextv2"):
            trunk = net.build_backbone(get_experiment(name).backbone, pretrained=False)
            assert trunk.num_features == 768
            trunk.eval()
            with torch.no_grad():
                features = trunk(torch.zeros(1, 3, 32, 32))
            assert features.shape == (1, 768) and torch.isfinite(features).all()
            torch.manual_seed(0)
            module = KneeLitModule(trunk, 768)
            states.append({k: v.clone() for k, v in module.state_dict().items()
                           if not k.startswith("model.backbone.")})
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    finally:
        torch.set_num_threads(previous_threads)


def test_classwise_shared_initialization_is_actually_identical():
    import torch
    from src.model.lightning import KneeLitModule
    states = []
    for classwise in (False, True):
        torch.manual_seed(0)
        module = KneeLitModule(torch.nn.Linear(8, 8), feature_dim=8,
                               classwise_pool=classwise)
        states.append(module.state_dict())
    upstream = [key for key in states[0]
                if not key.startswith(("model.study_pool.", "model.head."))]
    assert upstream and all(torch.equal(states[0][key], states[1][key]) for key in upstream)
    assert not torch.equal(states[0]["model.head.2.weight"], states[1]["model.head.2.weight"])


def test_input_search_skips_raw_dicom_tree(tmp_path):
    for folder in ["competitions", "rsna-knee-abnormality-detection", "datasets/owner/labels"]:
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        (directory / "train_labels.parquet").touch()
    tree = ast.parse(INPUT_SEARCH)
    class RootPath(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "/kaggle/input":
                return ast.copy_location(ast.Constant(str(tmp_path)), node)
            return node
    namespace = {}
    exec(compile(ast.fix_missing_locations(RootPath().visit(tree)), "input_search", "exec"), namespace)
    assert namespace["attached_files"]("train_labels.parquet") == [tmp_path / "datasets/owner/labels/train_labels.parquet"]


@pytest.mark.parametrize("fold,count", [(0, 871), (2, 870), (4, 868)])
def test_completion_source_keeps_protocol_settings_and_exact_fold(fold, count):
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    source = training_source(notebook, "protocol", "hash", fold=fold, complete_protocol=True)
    assert f"EXPECTED_FOLD = {fold}" in source and f"EXPECTED_VAL = {count}" in source
    assert f"FOLD = {fold}" in source
    assert "Completion settings or initialization differ from screened protocol" in source
    assert source.index("REFERENCE_CONTRACT.items()") < source.index("trainer.fit(")
    assert "MATCH_CLASSWISE = False" in source and "MATCH_BACKBONE = False" in source


@pytest.mark.parametrize("experiment,fold", [("classwise", 0), ("convnextv2", 2),
                                            ("baseline", 4), ("protocol", 1)])
def test_completion_rejects_unreviewed_variants_and_screening_folds(experiment, fold):
    with pytest.raises(ValueError, match="Completion is only"):
        training_source({}, experiment, "hash", fold=fold, complete_protocol=True)


def test_smoke_only_runner_exits_before_full_training(tmp_path, monkeypatch):
    from tools import kaggle_training_pair_body as body
    import types
    original_path = Path
    monkeypatch.setattr(body, "Path", lambda value: tmp_path if value == "/kaggle/working" else original_path(value))
    for name, value in {"FOLD": 1, "LABELS_SHA256": "hash", "RECOVERY": None,
                        "SMOKE_ONLY": True, "SMOKE_SOURCE": "print(1)", "SMOKE_TIMEOUT": 900,
                        "TRAINING_SCRIPTS": {"must_not_run": "raise RuntimeError"}}.items():
        monkeypatch.setattr(body, name, value, raising=False)
    calls = []
    monkeypatch.setattr(body, "subprocess", types.SimpleNamespace(run=lambda *a, **kw: calls.append((a, kw))))
    body.main()
    assert len(calls) == 1 and calls[0][1]["timeout"] == 900
    assert not (tmp_path / "must_not_run_fold1").exists()
    progress = json.loads((tmp_path / "pair_progress.json").read_text())
    assert progress["state"] == "complete" and progress["smoke_only"]
    assert progress["completed"] == []
