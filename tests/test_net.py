"""Tests for the study-level network.

The property most worth guarding is masking. A padded series must not change
the output for a study, and that failure is invisible: the model still trains,
still converges, and quietly dilutes every study with fewer series than the
batch maximum.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import net  # noqa: E402
from src.model.dataset import META_COLUMNS  # noqa: E402

FEATURES = 32
DEPTH, SIZE = 4, 16


class TinyBackbone(nn.Module):
    """Stand-in for timm: (N, 3, H, W) -> (N, FEATURES)."""

    def __init__(self, dim=FEATURES):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


def make_model(**kwargs):
    torch.manual_seed(0)
    return net.KneeNet(TinyBackbone(), FEATURES, **kwargs).eval()


def make_batch(batch=2, series=3):
    torch.manual_seed(1)
    volumes = torch.randint(0, 255, (batch, series, DEPTH, SIZE, SIZE),
                            dtype=torch.uint8)
    meta = torch.rand(batch, series, len(META_COLUMNS))
    mask = torch.ones(batch, series, dtype=torch.bool)
    return volumes, meta, mask


class TestStack2p5d:
    def test_shape(self):
        volumes = torch.rand(2, DEPTH, SIZE, SIZE)
        assert net.stack_2p5d(volumes).shape == (2 * DEPTH, 3, SIZE, SIZE)

    def test_middle_channel_is_the_slice_itself(self):
        volumes = torch.rand(1, DEPTH, SIZE, SIZE)
        out = net.stack_2p5d(volumes)
        assert torch.allclose(out[2, 1], volumes[0, 2])

    def test_edges_repeat_rather_than_zero_pad(self):
        # A black plane at a volume edge is a feature the model would learn.
        volumes = torch.rand(1, DEPTH, SIZE, SIZE)
        out = net.stack_2p5d(volumes)
        assert torch.allclose(out[0, 0], volumes[0, 0])
        assert torch.allclose(out[DEPTH - 1, 2], volumes[0, DEPTH - 1])


class TestAttentionPool:
    def test_masked_entries_are_ignored(self):
        torch.manual_seed(0)
        pool = net.AttentionPool(FEATURES).eval()
        x = torch.rand(1, 4, FEATURES)
        mask = torch.tensor([[True, True, False, False]])

        full = pool(x, mask)
        # Change only the masked positions; the result must not move.
        x2 = x.clone()
        x2[0, 2:] = torch.rand(2, FEATURES) * 100
        assert torch.allclose(full, pool(x2, mask), atol=1e-6)

    def test_weights_sum_to_one_over_real_entries(self):
        pool = net.AttentionPool(FEATURES).eval()
        x = torch.ones(1, 3, FEATURES)
        out = pool(x, torch.ones(1, 3, dtype=torch.bool))
        assert torch.allclose(out, torch.ones(1, FEATURES), atol=1e-5)


class TestFiLM:
    def test_metadata_changes_the_embedding(self):
        # Findings are sequence-specific; if metadata does not move the
        # features, the conditioning is doing nothing.
        torch.manual_seed(0)
        film = net.FiLM(FEATURES).eval()
        x = torch.rand(2, 3, FEATURES)
        a = film(x, torch.zeros(2, 3, len(META_COLUMNS)))
        b = film(x, torch.ones(2, 3, len(META_COLUMNS)))
        assert not torch.allclose(a, b)

    def test_shape_preserved(self):
        film = net.FiLM(FEATURES).eval()
        x = torch.rand(2, 3, FEATURES)
        assert film(x, torch.rand(2, 3, len(META_COLUMNS))).shape == x.shape


class TestKneeNet:
    def test_output_shape(self):
        model = make_model()
        volumes, meta, mask = make_batch()
        with torch.no_grad():
            out = model(volumes, meta, mask)
        assert out.shape == (2, len(config.LABELS))

    def test_padding_does_not_change_a_study(self):
        # The load-bearing test. A study with two real series must score the
        # same whether or not the batch pads it to three.
        model = make_model()
        torch.manual_seed(3)
        volumes = torch.randint(0, 255, (1, 3, DEPTH, SIZE, SIZE), dtype=torch.uint8)
        meta = torch.rand(1, 3, len(META_COLUMNS))

        mask_two = torch.tensor([[True, True, False]])
        with torch.no_grad():
            padded = model(volumes, meta, mask_two)
            # Same two series, no padding at all.
            trimmed = model(volumes[:, :2], meta[:, :2],
                            torch.ones(1, 2, dtype=torch.bool))
        assert torch.allclose(padded, trimmed, atol=1e-5)

    def test_junk_in_padded_rows_is_ignored(self):
        model = make_model()
        volumes, meta, _ = make_batch(batch=1, series=3)
        mask = torch.tensor([[True, False, False]])
        with torch.no_grad():
            first = model(volumes, meta, mask)
            volumes[0, 1:] = 255
            meta[0, 1:] = 99.0
            second = model(volumes, meta, mask)
        assert torch.allclose(first, second, atol=1e-5)

    def test_single_series_study_works(self):
        model = make_model()
        volumes, meta, mask = make_batch(batch=1, series=1)
        with torch.no_grad():
            assert model(volumes, meta, mask).shape == (1, 12)

    def test_frozen_backbone_has_no_gradients(self):
        model = net.KneeNet(TinyBackbone(), FEATURES, freeze_backbone=True)
        assert not any(p.requires_grad for p in model.backbone.parameters())
        assert any(p.requires_grad for p in model.head.parameters())

    def test_gradients_flow_when_unfrozen(self):
        model = net.KneeNet(TinyBackbone(), FEATURES)
        volumes, meta, mask = make_batch()
        loss = model(volumes, meta, mask).sum()
        loss.backward()
        assert model.backbone.conv.weight.grad is not None


class TestWeightedBCE:
    def test_zero_weight_finding_is_ignored(self):
        logits = torch.zeros(2, 12)
        targets = torch.zeros(2, 12)
        weights = torch.ones(2, 12)
        weights[:, 0] = 0.0

        base = net.weighted_bce(logits, targets, weights)
        targets[:, 0] = 1.0  # would raise the loss if that finding counted
        assert torch.allclose(base, net.weighted_bce(logits, targets, weights))

    def test_soft_targets_are_accepted(self):
        loss = net.weighted_bce(torch.zeros(1, 12), torch.full((1, 12), 0.3),
                                torch.ones(1, 12))
        assert torch.isfinite(loss)

    def test_all_zero_weights_do_not_produce_nan(self):
        loss = net.weighted_bce(torch.zeros(1, 12), torch.zeros(1, 12),
                                torch.zeros(1, 12))
        assert torch.isfinite(loss)

    def test_confident_wrong_scores_worse_than_confident_right(self):
        weights = torch.ones(1, 12)
        right = net.weighted_bce(torch.full((1, 12), 5.0), torch.ones(1, 12), weights)
        wrong = net.weighted_bce(torch.full((1, 12), -5.0), torch.ones(1, 12), weights)
        assert wrong > right


class TestFrozenPath:
    """The frozen path must produce the same architecture as the end-to-end one.

    Both call `aggregate`, so FiLM, the series transformer, and the pooling
    cannot drift apart between the path used for fast iteration and the path
    used for a fine-tuned submission.
    """

    def test_forward_features_shape(self):
        model = make_model()
        features = torch.rand(2, 3, DEPTH, FEATURES)
        meta = torch.rand(2, 3, len(META_COLUMNS))
        mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            assert model.forward_features(features, meta, mask).shape == (2, 12)

    def test_masking_holds_on_the_frozen_path(self):
        model = make_model()
        torch.manual_seed(5)
        features = torch.rand(1, 3, DEPTH, FEATURES)
        meta = torch.rand(1, 3, len(META_COLUMNS))
        mask = torch.tensor([[True, True, False]])
        with torch.no_grad():
            padded = model.forward_features(features, meta, mask)
            trimmed = model.forward_features(
                features[:, :2], meta[:, :2], torch.ones(1, 2, dtype=torch.bool))
        assert torch.allclose(padded, trimmed, atol=1e-5)

    def test_backbone_may_be_omitted(self):
        # On the frozen path the trunk already ran offline.
        model = net.KneeNet(None, FEATURES).eval()
        features = torch.rand(1, 2, DEPTH, FEATURES)
        with torch.no_grad():
            out = model.forward_features(
                features, torch.rand(1, 2, len(META_COLUMNS)),
                torch.ones(1, 2, dtype=torch.bool))
        assert out.shape == (1, 12)

    def test_both_paths_share_the_aggregation(self):
        # Feed the end-to-end path's series embeddings through aggregate
        # directly and confirm the frozen path reaches the same result.
        model = make_model()
        torch.manual_seed(7)
        series = torch.rand(1, 2, FEATURES)
        meta = torch.rand(1, 2, len(META_COLUMNS))
        mask = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            a = model.aggregate(series, meta, mask)
            # A depth-1 feature stack pools to the same series embedding.
            b = model.forward_features(series.unsqueeze(2), meta, mask)
        assert torch.allclose(a, b, atol=1e-5)


class TestInputProjection:
    """Wide spatial embeddings are projected before the aggregation stack.

    Running the stack at 6,528 wide puts roughly 769M parameters in it, against
    8.2M with a projection -- most of it in FiLM, which emits two vectors per
    feature.
    """

    def test_projection_reduces_width(self):
        model = net.KneeNet(None, 64, input_dim=512).eval()
        features = torch.rand(2, 3, DEPTH, 512)
        meta = torch.rand(2, 3, len(META_COLUMNS))
        mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            assert model.forward_features(features, meta, mask).shape == (2, 12)

    def test_identity_when_widths_match(self):
        model = net.KneeNet(None, FEATURES, input_dim=FEATURES)
        assert isinstance(model.project, torch.nn.Identity)

    def test_omitting_input_dim_is_identity(self):
        model = net.KneeNet(None, FEATURES)
        assert isinstance(model.project, torch.nn.Identity)

    def test_projection_keeps_the_stack_small(self):
        wide = net.KneeNet(None, 512, input_dim=6528)
        naive = net.KneeNet(None, 6528)
        assert sum(p.numel() for p in wide.parameters()) < \
            sum(p.numel() for p in naive.parameters()) / 20

    def test_masking_still_holds_with_a_projection(self):
        model = net.KneeNet(None, 64, input_dim=256).eval()
        torch.manual_seed(11)
        features = torch.rand(1, 3, DEPTH, 256)
        meta = torch.rand(1, 3, len(META_COLUMNS))
        with torch.no_grad():
            padded = model.forward_features(
                features, meta, torch.tensor([[True, True, False]]))
            trimmed = model.forward_features(
                features[:, :2], meta[:, :2], torch.ones(1, 2, dtype=torch.bool))
        assert torch.allclose(padded, trimmed, atol=1e-5)


class TestLoadKneeState:
    """The submission must read either checkpoint format.

    Notebook 08 exports a bare state dict; Lightning writes the same tensors
    nested under `state_dict` behind a `model.` prefix. Requiring the export
    first is requiring a step that can be skipped, and one already was.
    """

    def _state(self):
        import torch
        backbone = torch.nn.Linear(4, 16)
        return net.KneeNet(backbone, 16).state_dict()

    def test_reads_a_bare_state_dict(self, tmp_path):
        import torch
        state = self._state()
        path = tmp_path / "convnext_fold0.pt"
        torch.save(state, path)
        assert set(net.load_knee_state(str(path))) == set(state)

    def test_reads_a_lightning_checkpoint(self, tmp_path):
        import datetime
        import torch
        state = self._state()
        path = tmp_path / "lit_fold0.ckpt"
        torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()},
                    "saved_at": datetime.datetime(2026, 8, 24)}, path)
        assert set(net.load_knee_state(str(path))) == set(state)

    def test_both_formats_load_into_the_same_network(self, tmp_path):
        import torch
        state = self._state()
        bare, lit = tmp_path / "a.pt", tmp_path / "b.ckpt"
        torch.save(state, bare)
        torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}},
                   lit)
        for path in (bare, lit):
            fresh = net.KneeNet(torch.nn.Linear(4, 16), 16)
            fresh.load_state_dict(net.load_knee_state(str(path)))

    def test_an_empty_checkpoint_raises(self, tmp_path):
        import torch
        path = tmp_path / "empty.ckpt"
        torch.save({"state_dict": {}}, path)
        with pytest.raises(ValueError, match="no weights found"):
            net.load_knee_state(str(path))
