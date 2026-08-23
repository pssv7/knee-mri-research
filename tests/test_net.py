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
