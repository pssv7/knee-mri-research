"""Foundation geometry, checkpoint, cache and comparison boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.model import foundation as fm, net, dataset
from src.validation.proxy import ID, LABELS, sha
from tools.train_foundation import (extract_cache, FeatureDataset, make_module,
                                    load_selected, verify_smoke)
from tools.evaluate_foundation import matched_contracts


class TinyOfficialEncoder(nn.Module):
    """A differentiable surrogate for interface/gradient tests, never an accuracy model."""
    def __init__(self):
        super().__init__()
        self.map = nn.Linear(3, fm.ENCODER_DIM)
        self.last_input = None

    def forward_features(self, x):
        self.last_input = x.detach().clone()
        return {"x_norm_clstoken": self.map(x.mean((-1, -2)))}


def image_model(frozen=True, chunk=16, checkpoint=False):
    return fm.FoundationKneeNet(fm.FoundationEncoder(
        TinyOfficialEncoder(), frozen=frozen, chunk_size=chunk, checkpoint_blocks=checkpoint), head_dim=16)


def args():
    return (torch.randint(0, 256, (1, 2, 3, 224, 224), dtype=torch.uint8),
            torch.randn(1, 2, len(dataset.META_COLUMNS)), torch.tensor([[True, False]]))


def test_grayscale_replication_normalization_and_output_shape():
    m = image_model().eval()
    volumes, meta, mask = args()
    logits = m(volumes, meta, mask)
    seen = m.backbone.encoder.last_input
    recovered = seen * m.input_std + m.input_mean
    expected = volumes.flatten(0, 2).float() / 255
    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
    for channel in range(3):
        torch.testing.assert_close(recovered[:, channel], expected)
    assert "foundation_signature" in m.state_dict()


def test_cached_and_image_paths_agree_and_head_receives_gradients():
    m = image_model().eval()
    v, meta, mask = args()
    features = m.extract_slices(v.flatten(0, 1)).reshape(1, 2, 3, fm.ENCODER_DIM)
    head = fm.FoundationKneeNet(cached=True, head_dim=16).eval()
    head.load_state_dict(fm.head_state(m), strict=True)
    torch.testing.assert_close(m(v, meta, mask), head(features, meta, mask))
    loss = net.weighted_bce(head(features, meta, mask), torch.rand(1, 12), torch.ones(1, 12))
    loss.backward()
    assert head.project.weight.grad is not None
    assert all(p.grad is None for p in m.backbone.parameters())


def test_frozen_encoder_stays_eval_when_head_trains():
    m = image_model().train()
    assert m.training and m.head.training and not m.backbone.encoder.training
    assert not any(p.requires_grad for p in m.backbone.parameters())


def test_chunking_and_checkpointed_finetuning_preserve_outputs_and_gradients():
    torch.manual_seed(10)
    first = image_model(frozen=False, chunk=1).train()
    second = image_model(frozen=False, chunk=4, checkpoint=True).train()
    second.load_state_dict(first.state_dict())
    x = torch.randn(5, 3, 224, 224)
    a, b = first.backbone(x), second.backbone(x)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(first.backbone.encoder.map.weight.grad, second.backbone.encoder.map.weight.grad)


@pytest.mark.parametrize("defect", ["missing", "changed", "legacy"])
def test_incompatible_preprocessing_checkpoint_is_rejected(defect):
    m = image_model()
    state = m.state_dict()
    if defect == "missing":
        del state["foundation_signature"]
    elif defect == "changed":
        state["foundation_signature"] = state["foundation_signature"].clone()
        state["foundation_signature"][-1] = 2
    else:
        state = net.KneeNet(None, 16).state_dict()
    with pytest.raises(RuntimeError):
        m.load_state_dict(state, strict=True)


def test_head_initialization_independent_of_encoder_rng_and_plain_state_unchanged():
    torch.manual_seed(21)
    plain = net.KneeNet(None, 16)
    assert "foundation_signature" not in plain.state_dict()
    assert "input_mean" not in plain.state_dict()
    a = image_model().eval()
    fresh = image_model().eval()
    fresh.load_state_dict(a.state_dict(), strict=True)
    x = args()
    torch.testing.assert_close(a(*x), fresh(*x), rtol=0, atol=0)


def test_encoder_state_rejects_unknown_wrapping_and_non_dinov3():
    state = {"backbone.storage_tokens": torch.zeros(1, 4, 1024),
             "backbone.rope_embed.periods": torch.zeros(16)}
    assert set(fm.extract_encoder_state(state, "orthofoundation")) == {"storage_tokens", "rope_embed.periods"}
    for bad in [{"state_dict": state}, {**state, "head.weight": torch.zeros(1)}, {"backbone.pos_embed": torch.zeros(1)}]:
        with pytest.raises(ValueError):
            fm.extract_encoder_state(bad, "orthofoundation")


def test_source_lock_rejects_changes_and_extra_importable_source(tmp_path):
    root = tmp_path / "dinov3"
    root.mkdir()
    code = root / "__init__.py"
    code.write_text("# fixture\n")
    lock = {"dinov3_commit": fm.DINOV3_COMMIT, "source_sha256": {"dinov3/__init__.py": sha(code)}}
    fm.verify_source(tmp_path, lock)
    code.write_text("# changed\n")
    with pytest.raises(ValueError, match="source missing or changed"):
        fm.verify_source(tmp_path, lock)
    code.write_text("# fixture\n")
    (root / "extra.py").write_text("")
    with pytest.raises(ValueError, match="inventory"):
        fm.verify_source(tmp_path, lock)


def test_cache_binds_features_and_metadata_and_detects_tampering(tmp_path):
    m = image_model().eval()
    source = [{ID: "study_a", "volumes": np.full((1, 16, 224, 224), 128, dtype=np.uint8),
               "meta": np.ones((1, len(dataset.META_COLUMNS)), dtype=np.float32),
               "targets": np.arange(12, dtype=np.float32) / 12, "weights": np.ones(12, dtype=np.float32)}]
    binding = {"encoder_sha256": "one", "role": "train"}
    result = extract_cache(m, source, tmp_path / "cache", binding, device="cpu")
    item = result[0]
    assert item[ID] == "study_a" and item["volumes"].shape == (1, 16, 1024)
    np.testing.assert_equal(item["targets"], source[0]["targets"])
    with pytest.raises(ValueError, match="another encoder"):
        FeatureDataset(tmp_path / "cache", {**binding, "role": "outer"})
    with (tmp_path / "cache/features.npy").open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"\xff")
    with pytest.raises(ValueError, match="bytes changed"):
        FeatureDataset(tmp_path / "cache", binding)


def test_finetune_optimizer_separates_encoder_and_head_rates():
    m = image_model(frozen=False)
    recipe = {"learning_rate": 1e-4, "weight_decay": 1e-4, "backbone_lr_scale": 1/30,
              "max_lr_multiplier": 3.0, "schedule": "onecycle", "epochs": 12, "warmup_epochs": 3.6}
    module = make_module(m, recipe, 120)
    optimizer = module.configure_optimizers()["optimizer"]
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["max_lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["max_lr"] == pytest.approx(3e-4)
    trunk = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert id(m.project.weight) not in trunk
    assert id(m.backbone.encoder.map.weight) in trunk


def contracts():
    control = dict(roles_sha256="roles", proxy_labels_sha256="labels", protected_exclusions_sha256="exclusions",
        foundation_schema_version=1, stage="frozen", seed=0, training_recipe={"same": True},
        runtime_versions={"same": True}, input_config_sha256="inputs", experiment_sha256="experiment",
        initial_head_sha256="head", runtime_code_sha256={"same": True}, encoder_sha256="base", arm="dinov3")
    return control, {**copy.deepcopy(control), "arm": "orthofoundation", "encoder_sha256": "knee"}


@pytest.mark.parametrize("key", ["stage", "seed", "training_recipe", "runtime_versions", "input_config_sha256",
                                  "initial_head_sha256", "runtime_code_sha256", "roles_sha256"])
def test_comparison_refuses_unmatched_runs(key):
    c, a = contracts()
    matched_contracts(c, a)
    a[key] = "different"
    with pytest.raises(ValueError):
        matched_contracts(c, a)


def test_historical_comparison_requires_explicit_flag_and_correct_model():
    c, a = contracts()
    c.update(arm="control", backbone="convnext_tiny.fb_in22k_ft_in1k", training_recipe={"legacy": True})
    with pytest.raises(ValueError):
        matched_contracts(c, a)
    matched_contracts(c, a, historical_tiny=True)


def test_training_requires_matching_gpu_smoke():
    spec = {"stage": "frozen"}
    receipt = dict(status="passed", arm="orthofoundation", stage="frozen", experiment_sha256="spec",
        input_config_sha256="inputs", encoder_sha256="weights", runtime_versions={"gpu": "T4"},
        optimizer_steps=2, roundtrip_max_abs=0.0)
    arguments = (spec, "spec", "inputs", "weights", "orthofoundation", {"gpu": "T4"})
    verify_smoke(receipt, *arguments)
    for bad in [None, {**receipt, "stage": "finetune"}, {**receipt, "runtime_versions": {"gpu": "A100"}},
                {**receipt, "experiment_sha256": "other"}, {**receipt, "roundtrip_max_abs": .01}]:
        with pytest.raises(ValueError):
            verify_smoke(bad, *arguments)


def test_real_lightning_fit_select_export_and_image_head_reload(tmp_path):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    from src.model import lightning as knee
    head = fm.FoundationKneeNet(cached=True)
    items = [{ID: f"s{i}", "volumes": np.full((1, 2, 1024), i + .2, dtype=np.float32),
              "meta": np.zeros((1, len(dataset.META_COLUMNS)), dtype=np.float32),
              "targets": np.full(12, float(i % 2), dtype=np.float32),
              "weights": np.ones(12, dtype=np.float32)} for i in range(4)]
    loader = DataLoader(items, batch_size=2, collate_fn=dataset.collate_studies)
    recipe = {"learning_rate": 1e-4, "weight_decay": 1e-4, "backbone_lr_scale": 1.,
              "max_lr_multiplier": 3., "schedule": "onecycle", "epochs": 12, "warmup_epochs": 3.6}
    module = make_module(head, recipe, 24)
    callback = pl.callbacks.ModelCheckpoint(dirpath=tmp_path / "checkpoints", monitor="val_macro", mode="max")
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                         callbacks=[callback], enable_model_summary=False, enable_progress_bar=False,
                         num_sanity_val_steps=0)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert callback.best_model_path
    selected = tmp_path / "selected.pt"
    knee.export_from_checkpoint(callback.best_model_path, str(selected))
    encoder = fm.FoundationEncoder(TinyOfficialEncoder())
    restored_image = fm.FoundationKneeNet(encoder).eval()
    load_selected(restored_image, selected, "frozen")
    restored_head = fm.FoundationKneeNet(cached=True).eval()
    restored_head.load_state_dict(torch.load(selected, weights_only=True), strict=True)
    batch = next(iter(loader))
    expected = restored_head(batch["volumes"], batch["meta"], batch["mask"])
    actual = restored_image.forward_features(batch["volumes"], batch["meta"], batch["mask"])
    torch.testing.assert_close(actual, expected)


def test_generated_notebook_and_frozen_bundle_are_reproducible(tmp_path):
    import ast
    import io
    import zipfile
    import nbformat
    ROOT = Path(__file__).resolve().parents[1]
    experiment = ROOT / "outputs/foundation_frozen_outer0_v2"
    repo = ROOT / "outputs/foundation_vendor" / ("dinov3-" + fm.DINOV3_COMMIT)
    if not experiment.is_dir() or not repo.is_dir():
        pytest.skip("Local prepared artifacts are not available in this checkout")
    from tools.prepare_colab_foundation import build, payload
    generated = build(experiment, repo, tmp_path / "foundation.ipynb")
    checked_in = ROOT / "notebooks/23_orthofoundation_frozen_outer0_colab.ipynb"
    assert generated.read_bytes() == checked_in.read_bytes()
    notebook = nbformat.read(generated, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse("\n".join(x for x in cell.source.splitlines() if not x.startswith("%")))
    archive, _ = payload(experiment, repo)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read("bundle_manifest.json"))
        import hashlib
        for name, digest in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
        assert not any(name.endswith((".pth", ".pt", ".npy")) for name in bundle.namelist())
        assert "vendor/LICENSE.md" in bundle.namelist()
        roles = pd.read_csv(io.BytesIO(bundle.read("design/baseline/roles.csv")))
        assert roles.role.value_counts().to_dict() == {"train": 2208, "outer": 691, "selection": 553}
        spec = json.loads(bundle.read("design/experiment.json"))
        assert spec["stage"] == "frozen" and spec["recipe"]["augmentation"] == "none"
        for name, digest in spec["runtime_code_sha256"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest


def test_evaluator_runs_on_bound_artifacts_and_rejects_tampered_predictions(tmp_path):
    from tools.evaluate_foundation import evaluate, ROOT
    experiment = ROOT / "outputs/foundation_frozen_outer0_v2"
    if not experiment.is_dir():
        pytest.skip("Prepared nested design is not available")
    spec = json.loads((experiment / "experiment.json").read_text())
    baseline = json.loads((experiment / "baseline/spec.json").read_text())
    roles = pd.read_csv(experiment / "baseline/roles.csv")
    labels = pd.read_parquet(experiment / "baseline/proxy_labels.parquet").set_index(ID)
    uids = sorted(roles.loc[roles.role == "outer", ID])
    truth = (labels.loc[uids, LABELS].to_numpy() >= .5).astype(float)
    for arm, values in [("dinov3", .9 - .8 * truth), ("orthofoundation", .1 + .8 * truth)]:
        folder = tmp_path / arm
        folder.mkdir()
        predictions = pd.DataFrame(values, columns=LABELS)
        predictions.insert(0, ID, uids)
        predictions.to_parquet(folder / "outer_predictions.parquet", index=False)
        (folder / "selected.pt").write_bytes(b"test fixture; not model weights")
        contract = {**baseline, "arm": arm, "encoder_sha256": arm,
            "foundation_schema_version": 1, "stage": "frozen", "training_recipe": spec["recipe"],
            "runtime_versions": {"test": True}, "initial_head_sha256": "same-head",
            "input_config_sha256": "same-inputs", "experiment_sha256": sha(experiment / "experiment.json"),
            "checkpoint_sha256": sha(folder / "selected.pt"),
            "outer_predictions_sha256": sha(folder / "outer_predictions.parquet"),
            **{f"{role}_uids": sorted(roles.loc[roles.role == role, ID]) for role in ["train", "selection", "outer"]}}
        (folder / "run_contract.json").write_text(json.dumps(contract))
    result = evaluate(experiment, tmp_path / "dinov3", tmp_path / "orthofoundation",
                      tmp_path / "comparison.json", draws=100)
    assert result["macro_delta"] == pytest.approx(1.)
    assert result["secondary_mean_delta"] == pytest.approx(1.)
    assert result["numerical_screen_pass"]
    assert result["decision"] == "confirmation_required"
    assert result["image_gain_demonstrated"] is False
    with pytest.raises(FileExistsError):
        evaluate(experiment, tmp_path / "dinov3", tmp_path / "orthofoundation",
                 tmp_path / "comparison.json", draws=100)
    path = tmp_path / "orthofoundation/outer_predictions.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        evaluate(experiment, tmp_path / "dinov3", tmp_path / "orthofoundation",
                 tmp_path / "other.json", draws=100)
