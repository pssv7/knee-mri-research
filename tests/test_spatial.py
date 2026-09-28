"""Behaviour checks for the spatial volume encoder and the plane adapters.

These are the local tests the multi-plane plan requires before any GPU time:
gradients reach the new parameters, missing planes and padded series are
masked rather than read as zeros, and a checkpoint reloads into a model
rebuilt from its own weights.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.model import net
from src.model.dataset import META_COLUMNS
from src.model.experiments import get_experiment
from src.model.lightning import KneeLitModule

PLANE_INDEX = {name: META_COLUMNS.index(f"is_{name}")
               for name in ("sagittal", "coronal", "axial")}


class TinyTrunk(torch.nn.Module):
    """A stand-in with timm's two-stage interface: feature maps, then a head.

    `forward` must equal `forward_head(forward_features(x))` exactly, which is
    the property the spatial path relies on to reduce to its 2D control.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.num_features = dim
        self.conv = torch.nn.Conv2d(net.SLICE_CONTEXT, dim, 3, padding=1)
        self.norm = torch.nn.LayerNorm(dim)

    def forward_features(self, x):
        return self.conv(x)

    def forward_head(self, maps, pre_logits=False):
        return self.norm(maps.mean(dim=(2, 3)))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def study(batch=2, series=3, depth=4, size=8, planes=("sagittal", "coronal", "axial")):
    """Volumes, one-hot plane metadata and a mask, with the last series padded."""
    volumes = torch.randint(0, 255, (batch, series, depth, size, size), dtype=torch.uint8)
    meta = torch.zeros(batch, series, len(META_COLUMNS))
    for i in range(series):
        plane = planes[i % len(planes)] if planes else None
        if plane is not None:
            meta[:, i, PLANE_INDEX[plane]] = 1.0
    mask = torch.ones(batch, series, dtype=torch.bool)
    mask[-1, -1] = False
    return volumes, meta, mask


def build(**kwargs):
    trunk = TinyTrunk()
    return net.KneeNet(trunk, trunk.num_features, **kwargs)


def test_zero_initialised_spatial_blocks_reproduce_the_2d_control_exactly():
    """The screening arms must differ by the 3D path and nothing else."""
    torch.manual_seed(0)
    control = build().eval()
    torch.manual_seed(0)
    candidate = build(spatial_encoder=True).eval()
    volumes, meta, mask = study()
    with torch.no_grad():
        assert torch.allclose(control(volumes, meta, mask),
                              candidate(volumes, meta, mask), atol=1e-6)

    shared = {k: v for k, v in candidate.state_dict().items()
              if not k.startswith("spatial_encoder.")}
    assert set(shared) == set(control.state_dict())
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in shared.items())


def test_the_spatial_blocks_are_a_3d_computation_that_trains():
    torch.manual_seed(1)
    encoder = net.SpatialVolumeEncoder(4, width=8, blocks=1).train()
    maps = torch.randn(2, 5, 4, 6, 6)
    assert encoder(maps).shape == maps.shape

    optimiser = torch.optim.SGD(encoder.parameters(), lr=0.5)
    encoder(maps).square().mean().backward()
    assert encoder.output.weight.grad.abs().sum() > 0
    # The interior of a zero-initialised residual branch is reached only once
    # its output convolution has moved -- the first step is what unsticks it.
    assert encoder.blocks[0][1].weight.grad.abs().sum() == 0
    optimiser.step()
    optimiser.zero_grad()

    encoder(maps).square().mean().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert encoder.blocks[0][1].weight.grad.abs().sum() > 0

    # Depth is a real axis: the output at a slice must depend on its
    # neighbours, not only on itself.
    optimiser.step()
    encoder.eval()
    with torch.no_grad():
        original = encoder(maps)
        changed = maps.clone()
        changed[:, 0] += 1.0
        assert not torch.allclose(original[:, 1], encoder(changed)[:, 1])
        assert not torch.allclose(original, encoder(maps.flip(1)).flip(1))


def test_the_spatial_encoder_rejects_geometry_it_cannot_be_given():
    encoder = net.SpatialVolumeEncoder(4, width=8, blocks=1)
    with pytest.raises(ValueError, match="feature maps"):
        encoder(torch.randn(2, 5, 3, 6, 6))
    with pytest.raises(ValueError, match="feature maps"):
        encoder(torch.randn(2, 5, 4, 6))
    with pytest.raises(ValueError, match="divisible"):
        net.SpatialVolumeEncoder(4, width=12, blocks=1, groups=8)


def test_the_frozen_embedding_path_cannot_silently_skip_the_spatial_blocks():
    with pytest.raises(ValueError, match="forward_features"):
        net.KneeNet(None, 8, spatial_encoder=True)
    with pytest.raises(ValueError, match="forward_features"):
        net.KneeNet(torch.nn.Linear(8, 8), 8, spatial_encoder=True)


def test_plane_membership_is_one_group_per_series_and_excludes_padding():
    volumes, meta, mask = study()
    groups = net.PlaneAdapterPool.plane_masks(meta, mask)
    assert groups.shape == (2, 3, len(net.PLANE_GROUPS))
    assert torch.equal(groups.sum(dim=-1), mask.long())
    assert groups[0, :, :3].diagonal().all()          # sagittal, coronal, axial
    assert not groups[..., 3].any()                    # nothing unknown here

    _, blank, blank_mask = study(planes=())
    unknown = net.PlaneAdapterPool.plane_masks(blank, blank_mask)
    assert unknown[..., 3].equal(blank_mask)
    assert not unknown[..., :3].any()


def test_a_missing_plane_is_masked_rather_than_pooled_as_a_zero_vector():
    torch.manual_seed(2)
    pool = net.PlaneAdapterPool(8, hidden=4, dropout=0.0).eval()
    series = torch.randn(1, 3, 8)
    present = torch.ones(1, 3, dtype=torch.bool)

    meta = torch.zeros(1, 3, len(META_COLUMNS))
    meta[0, :, PLANE_INDEX["sagittal"]] = 1.0
    sagittal_only = pool(series, meta, present)

    # The same series, with an axial slot that was never acquired: a model
    # reading a zero vector for it would move; a masked one must not.
    padded = torch.cat([series, torch.zeros(1, 1, 8)], dim=1)
    padded_meta = torch.cat([meta, torch.zeros(1, 1, len(META_COLUMNS))], dim=1)
    padded_meta[0, 3, PLANE_INDEX["axial"]] = 1.0
    padded_mask = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        assert torch.allclose(sagittal_only, pool(padded, padded_meta, padded_mask),
                              atol=1e-6)

    # An acquired axial series, by contrast, has to change the answer.
    with torch.no_grad():
        acquired = pool(padded, padded_meta, torch.ones(1, 4, dtype=torch.bool))
    assert not torch.allclose(sagittal_only, acquired, atol=1e-5)


def test_a_study_with_no_usable_series_pools_to_zero_instead_of_nan():
    pool = net.PlaneAdapterPool(8, hidden=4, dropout=0.0).eval()
    series = torch.randn(1, 2, 8)
    meta = torch.zeros(1, 2, len(META_COLUMNS))
    with torch.no_grad():
        pooled = pool(series, meta, torch.zeros(1, 2, dtype=torch.bool))
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(pooled) == 0


def test_plane_adapters_train_and_every_group_receives_gradient():
    torch.manual_seed(3)
    model = build(spatial_encoder=True, plane_adapters=True).train()
    # Five series over three planes, so sagittal and coronal carry two each:
    # attention over a single series is a softmax over one element and has no
    # gradient to give, which is arithmetic rather than a wiring fault.
    volumes, meta, mask = study(series=5)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)
    logits = model(volumes, meta, mask)
    assert logits.shape == (2, len(config.LABELS))
    logits.square().mean().backward()
    assert model.spatial_encoder.output.weight.grad.abs().sum() > 0
    optimiser.step()
    optimiser.zero_grad()

    model(volumes, meta, mask).square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    # Every acquired plane's own adapter and pool trains, including the
    # fallback group, which is what carries series with no usable plane tag.
    for group in net.PLANE_GROUPS[:3]:
        assert model.study_pool.adapters[group][1].weight.grad.abs().sum() > 0, group
    for group in ("sagittal", "coronal"):
        assert model.study_pool.pools[group].score[0].weight.grad.abs().sum() > 0, group
    assert model.study_pool.fuse.score[0].weight.grad.abs().sum() > 0
    assert model.study_pool.plane_embedding.grad.abs().sum() > 0
    assert model.spatial_encoder.blocks[0][1].weight.grad.abs().sum() > 0


def test_padded_series_cannot_change_a_prediction():
    torch.manual_seed(4)
    model = build(spatial_encoder=True, plane_adapters=True).eval()
    volumes, meta, mask = study(batch=1, series=2)
    mask[:] = True
    padding = torch.randint(0, 255, (1, 1, *volumes.shape[2:]), dtype=torch.uint8)
    with torch.no_grad():
        plain = model(volumes, meta, mask)
        padded = model(torch.cat([volumes, padding], dim=1),
                       torch.cat([meta, torch.zeros(1, 1, len(META_COLUMNS))], dim=1),
                       torch.cat([mask, torch.zeros(1, 1, dtype=torch.bool)], dim=1))
    assert torch.allclose(plain, padded, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"spatial_encoder": True},
                                    {"spatial_encoder": True, "plane_adapters": True}])
def test_a_checkpoint_rebuilds_from_its_own_weights_and_scores_the_same(kwargs, tmp_path):
    torch.manual_seed(5)
    model = build(**kwargs).eval()
    volumes, meta, mask = study()
    with torch.no_grad():
        expected = model(volumes, meta, mask)

    path = tmp_path / "spatial.pt"
    torch.save(model.state_dict(), path)
    state = net.load_knee_state(path)
    defaults = net.aggregation_defaults(state)
    assert defaults["spatial_encoder"] is True
    assert defaults["plane_adapters"] is kwargs.get("plane_adapters", False)

    trunk = TinyTrunk()
    clone = net.KneeNet(trunk, trunk.num_features, **defaults).eval()
    clone.load_state_dict(state, strict=True)
    with torch.no_grad():
        assert torch.allclose(expected, clone(volumes, meta, mask), atol=1e-6)


def test_a_plain_checkpoint_still_rebuilds_as_a_plain_model():
    torch.manual_seed(6)
    state = build().state_dict()
    defaults = net.aggregation_defaults(state)
    assert not defaults["spatial_encoder"] and not defaults["plane_adapters"]
    assert not any("spatial_encoder" in key or "adapters" in key for key in state)


def test_plane_adapters_and_classwise_pooling_are_separate_experiments():
    with pytest.raises(ValueError, match="separate experiments"):
        build(plane_adapters=True, classwise_pool=True)


def test_the_wrapper_shares_every_unchanged_weight_with_its_control():
    """A matched pair records a shared-initialisation hash; keep it shareable."""
    states = []
    for kwargs in ({}, {"spatial_encoder": True}):
        torch.manual_seed(7)
        states.append(KneeLitModule(TinyTrunk(), 8, **kwargs).state_dict())
    shared = {k for k in states[1] if not k.startswith("model.spatial_encoder.")}
    assert shared == set(states[0])
    assert all(torch.equal(states[0][k], states[1][k]) for k in shared)


def test_the_presets_change_one_thing_at_a_time():
    control = get_experiment("spatial_control")
    spatial = get_experiment("spatial")
    planes = get_experiment("spatial_planes")
    assert control.slices == spatial.slices == 24
    assert control.series_selection == spatial.series_selection == "protocol"
    assert not control.spatial_encoder and spatial.spatial_encoder
    assert not (control.plane_adapters or spatial.plane_adapters)
    assert planes.spatial_encoder and planes.plane_adapters

    differences = {k for k, v in vars(spatial).items() if vars(control)[k] != v}
    assert differences == {"name", "spatial_encoder"}
    differences = {k for k, v in vars(planes).items() if vars(spatial)[k] != v}
    assert differences == {"name", "plane_adapters"}

    # Existing contracts and checkpoint sidecars must not gain keys.
    assert "spatial_encoder" not in get_experiment("baseline").to_dict()
    assert spatial.to_dict()["spatial_encoder"] is True
    assert "plane_adapters" not in spatial.to_dict()


def test_the_smoke_batch_has_the_cache_geometry_and_a_masked_series():
    from tools.spatial_smoke import synthetic_batch
    batch = synthetic_batch(2, 5, 24, 32, torch.device("cpu"))
    assert batch["volumes"].shape == (2, 5, 24, 32, 32)
    assert batch["volumes"].dtype == torch.uint8
    assert batch["meta"].shape == (2, 5, len(META_COLUMNS))
    # One plane flag per series, and one study short a series so the masking
    # the model depends on is exercised on every measured run.
    assert torch.equal(batch["meta"][..., 2:5].sum(-1), torch.ones(2, 5))
    assert batch["mask"].sum() == 9 and not batch["mask"][0, -1]
    assert batch["targets"].shape == batch["weights"].shape == (2, len(config.LABELS))


@pytest.mark.parametrize('variant', ['spatial', 'spatial_planes'])
@pytest.mark.parametrize('plane', ['Sagittal', None])
def test_sidecar_ensemble_export_agrees_with_missing_planes(tmp_path, monkeypatch, variant, plane):
    import json
    import numpy as np
    import pandas as pd
    from src.model import artifacts, dataset
    monkeypatch.setattr(config, 'CACHE_SIZE', 8)
    monkeypatch.setattr(config, 'CACHE_SLICES', 4)
    monkeypatch.setattr(net, 'build_backbone', lambda *a, **kw: TinyTrunk())
    spec = get_experiment(variant).to_dict()
    spec['slices'] = 4
    model = build(spatial_encoder=True, plane_adapters=variant == 'spatial_planes').eval()
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    artifacts.write_metadata(checkpoint, spec, 1, ['train'], ['val'])
    manifest = tmp_path / 'ensemble.json'
    manifest.write_text(json.dumps({'members': [{'checkpoint': 'model.pt', 'weight': 1}]}))
    rows = pd.DataFrame({'SeriesInstanceUID': ['b', 'a'], 'Anatomical_Plane': [plane, plane],
                         'side_predicted': ['R', 'R'], 'Fluid_Sensitive': [1, 0], 'Fat_Suppression': [1, 0]})
    volumes = [np.full((4, 8, 8), n, np.uint8) for n in (50, 100)]
    direct = artifacts.predict_members([{'model': model, 'spec': spec}], volumes, rows)
    reloaded = artifacts.predict_members(artifacts.load_ensemble(manifest), volumes, rows)
    assert reloaded.shape == (1, 12) and np.isfinite(reloaded).all()
    np.testing.assert_allclose(direct, reloaded, atol=1e-6, rtol=0)


def test_representative_smoke_accumulation_export_and_measurement(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from tools import spatial_smoke
    monkeypatch.setattr(config, 'CACHE_SIZE', 8)
    monkeypatch.setattr(config, 'CACHE_SLICES', 4)
    monkeypatch.setattr(net, 'build_backbone', lambda *a, **kw: TinyTrunk())
    spec = replace(get_experiment('spatial'), slices=4)
    monkeypatch.setattr(spatial_smoke.experiments, 'get_experiment', lambda name: spec)
    args = SimpleNamespace(studies=1, series=2, seed=0, pretrained=False,
                           full_series=True, precision='32', accumulation=2)
    result = spatial_smoke.run_arm('spatial', args, torch.device('cpu'), tmp_path)
    assert result['effective_batch'] == 2 and result['accumulation'] == 2
    assert result['reload_max_logit_gap'] < 1e-4
    assert result['training_memory_gate_passed'] is None, 'CPU must not establish GPU feasibility'
