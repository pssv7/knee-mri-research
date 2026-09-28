"""Regression tests for input selection, model variants and artifact identity."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.model import artifacts, dataset, experiments, lightning, net
from tools import compare_oof, package_16_24_ensemble


def protocols():
    return pd.DataFrame({"SeriesInstanceUID": [f"s{i}" for i in range(7)],
                         "Anatomical_Plane": ["Sagittal", "Axial", "Sagittal",
                                              "Coronal", "Coronal", "Sagittal", "Coronal"],
                         "Fluid_Sensitive": [0, 1, 1, 0, 1, 0, 0],
                         "Fat_Suppression": [0, 1, 1, 0, 1, 0, 0]})


class Trunk(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        return self.layer(x).mean((2, 3))


@pytest.mark.parametrize("side", [None, np.nan, pd.NA, "", "U", "B", "unknown"])
def test_unknown_side_never_mirrors(side):
    volume = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(dataset.mirror_volume(volume, side, "Coronal"), volume)


@pytest.mark.parametrize("value,expected", [(np.nan, 0), (pd.NA, 0), ("False", 0),
                                           ("0", 0), ("true", 1), (1, 1)])
def test_sequence_flags_are_finite(value, expected):
    meta = dataset.series_metadata(pd.Series({"Fluid_Sensitive": value}))
    assert np.isfinite(meta).all()
    assert meta[0] == expected


def test_side_metadata_matches_mirror_normalization():
    meta = dataset.series_metadata(pd.Series({"side_predicted": " r "}))
    assert meta[-2:].tolist() == [1, 1]


def test_protocol_selection_keeps_unique_plane_and_determinism():
    rows = protocols()
    assert "Axial" not in set(rows.iloc[dataset.select_series(rows, 5)].Anatomical_Plane)
    for rng in [None, np.random.default_rng(3)]:
        chosen = dataset.select_series(rows, 5, "protocol", rng)
        assert len(chosen) == len(set(chosen)) == 5
        assert np.array_equal(chosen, np.sort(chosen))
        assert set(rows.iloc[chosen].Anatomical_Plane) == set(rows.Anatomical_Plane)


@pytest.mark.parametrize("classwise,gated", [(False, False), (False, True), (True, False), (True, True)])
def test_pooling_export_roundtrip_and_padding(classwise, gated):
    torch.manual_seed(3)
    model = net.KneeNet(Trunk(), 8, classwise_pool=classwise, gated_pool=gated).eval()
    volumes = torch.randint(0, 256, (1, 2, 3, 8, 8), dtype=torch.uint8)
    meta = torch.zeros(1, 2, len(dataset.META_COLUMNS))
    mask = torch.ones(1, 2, dtype=torch.bool)
    expected = model(volumes, meta, mask)
    assert expected.shape == (1, 12)
    state = model.state_dict()
    restored = net.KneeNet(Trunk(), 8, **net.aggregation_defaults(state)).eval()
    restored.load_state_dict(state, strict=True)
    assert torch.allclose(expected, restored(volumes, meta, mask))
    padded = torch.cat([volumes, torch.zeros_like(volumes[:, :1])], 1)
    meta_pad = torch.cat([meta, torch.ones_like(meta[:, :1]) * 99], 1)
    mask_pad = torch.tensor([[True, True, False]])
    assert torch.allclose(expected, model(padded, meta_pad, mask_pad), atol=1e-6)
    loss = net.weighted_bce(expected, torch.ones_like(expected), torch.ones_like(expected))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_empty_attention_is_finite():
    x = torch.randn(2, 3, 8)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    for pool in [net.AttentionPool(8), net.ClassAttentionPool(8, 12)]:
        out = pool(x, mask)
        assert torch.isfinite(out).all() and torch.count_nonzero(out) == 0


def test_exact_optimizer_step_count():
    assert lightning.optimiser_steps(1000, 2, 2, 1, 12) == 3000
    assert lightning.optimiser_steps(1001, 2, 2, 2, 12) == 1512


def test_weight_average_samples_after_updates(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    trainer = SimpleNamespace(is_global_zero=True, global_step=0)
    module = SimpleNamespace(model=model)
    average = lightning.WeightAverage(str(tmp_path / "avg"), decay=.5,
                                      swa_fraction=1, total_steps=2)
    average.on_train_start(trainer, module)
    average.on_train_batch_end(trainer, module, None, None, 0)
    for step in [1, 2]:
        model.weight.data.fill_(step)
        trainer.global_step = step
        average.on_train_batch_end(trainer, module, None, None, step)
        average.on_train_batch_end(trainer, module, None, None, step)
    assert average.average.steps == 2
    assert average.average.swa_state()["weight"].item() == pytest.approx(1.5)


def test_explicit_ensemble_matches_dataset_and_rejects_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_SIZE", 8)
    monkeypatch.setattr(config, "CACHE_SLICES", 4)
    monkeypatch.setattr(net, "build_backbone", lambda *a, **kw: Trunk())
    rows = protocols().assign(StudyInstanceUID="s", cache_path="cache",
                              position=np.arange(7), side_predicted="L")
    volumes = np.random.default_rng(1).integers(0, 256, (7, 4, 8, 8), dtype=np.uint8)
    paths = []
    for variant in ["baseline", "classwise"]:
        path = tmp_path / (variant + "_fold0.pt")
        spec = experiments.get_experiment(variant).to_dict()
        spec.update(slices=3, series_selection="protocol")
        model = net.KneeNet(Trunk(), 8, classwise_pool=spec["classwise_pool"]).eval()
        torch.save(model.state_dict(), path)
        artifacts.write_metadata(path, spec, 0, ["train"], ["s"])
        paths.append({"checkpoint": path.name, "weight": 1})
    manifest = tmp_path / "ensemble.json"
    manifest.write_text(json.dumps({"members": paths}))
    members = artifacts.load_ensemble(manifest)
    shuffled = rows.sample(frac=1, random_state=7)
    actual = artifacts.predict_members(members, volumes[shuffled.index], shuffled)
    meta_cols = ["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    index = dataset.StudyIndex(shuffled.drop(columns=meta_cols[1:] + ["side_predicted"]),
                                rows[meta_cols], sides=rows[["SeriesInstanceUID", "side_predicted"]],
                                max_series=5, series_selection="protocol")
    item = dataset.load_study(index, "s", {"cache": volumes})
    keep = np.linspace(0, 3, 3).round().astype(int)
    tensor = torch.from_numpy(item["volumes"][:, keep]).unsqueeze(0)
    meta = torch.from_numpy(item["meta"]).unsqueeze(0)
    mask = torch.ones(1, len(item["volumes"]), dtype=torch.bool)
    with torch.no_grad():
        expected = np.stack([m["model"](tensor, meta, mask).sigmoid().numpy()[0] for m in members])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    manifest.write_text(json.dumps({"members": paths + paths[:1]}))
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.load_ensemble(manifest)
    with open(tmp_path / paths[0]["checkpoint"], "ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        artifacts.load_metadata(tmp_path / paths[0]["checkpoint"])


def test_oof_rejects_wrong_folds_and_duplicate_studies(tmp_path):
    folded = pd.DataFrame({config.ID_COL: ["s0", "s1"], "fold": [0, 1]})
    frame = folded.copy()
    for name in config.LABELS:
        frame[name] = [.2, .8]
    path = tmp_path / "oof.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        compare_oof.read_predictions([path, path], folded)
    frame["fold"] = [1, 0]
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="folds disagree"):
        compare_oof.read_predictions([path], folded)


def test_legacy_packaging_and_verified_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(net, "build_backbone", lambda *a, **kw: Trunk())
    monkeypatch.setattr(package_16_24_ensemble.splits, "load_gold_uids", lambda *a: set())
    labels = pd.DataFrame({config.ID_COL: [f"study{i}" for i in range(10)]})
    for name in config.LABELS:
        labels[name] = [.2, .8] * 5
    (tmp_path / "data/meta").mkdir(parents=True)
    labels.to_parquet(tmp_path / "data/meta/train_labels.parquet")
    for directory in ["models_train", "experiment /slice_24"]:
        for fold in range(5):
            source = tmp_path / directory / f"fold{fold}" / f"lit_fold{fold}.ckpt"
            source.parent.mkdir(parents=True)
            torch.save(net.KneeNet(Trunk(), 8).state_dict(), source)
    output = tmp_path / "package"
    package_16_24_ensemble.package(tmp_path, output)
    manifest = json.loads((output / "ensemble.json").read_text())
    assert len(manifest["members"]) == 10
    assert sum(m["weight"] for m in manifest["members"]) == pytest.approx(1)
    members = artifacts.load_ensemble(output / "ensemble.json")
    assert [m["spec"]["slices"] for m in members] == [16] * 5 + [24] * 5
    assert all(m["metadata"]["train_uids"] == [] for m in members)
    with pytest.raises(FileExistsError):
        package_16_24_ensemble.package(tmp_path, output)
    package_16_24_ensemble.package(tmp_path, output, resume=True)
    dest = output / manifest["members"][0]["checkpoint"]
    with dest.open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        package_16_24_ensemble.package(tmp_path, output, resume=True)
