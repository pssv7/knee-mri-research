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


class TestForwardPooled:
    def test_logits_match_forward(self):
        model = make_model()
        volumes, meta, mask = make_batch()
        with torch.no_grad():
            logits, pooled = model.forward_pooled(volumes, meta, mask)
            plain = model(volumes, meta, mask)
        assert torch.allclose(logits, plain)
        assert pooled.shape == (2, FEATURES)

    def test_pooled_feeds_the_head(self):
        # The pooled embedding must be exactly what the head reads, or the
        # auxiliary loss supervises a different representation than the one
        # producing the logits.
        model = make_model()
        volumes, meta, mask = make_batch()
        with torch.no_grad():
            logits, pooled = model.forward_pooled(volumes, meta, mask)
        assert torch.allclose(model.head(pooled), logits)


class TestPairwiseRankLoss:
    def _pair(self, score_current, score_ref, target_current=1.0,
              target_ref=0.0, **kwargs):
        return net.pairwise_rank_loss(
            torch.full((1, 12), score_current), torch.full((1, 12), target_current),
            torch.full((1, 12), score_ref), torch.full((1, 12), target_ref),
            **kwargs)

    def test_correct_order_scores_lower_than_reversed(self):
        good = self._pair(2.0, -2.0)   # higher target scored higher
        bad = self._pair(-2.0, 2.0)
        assert good < bad

    def test_pairs_below_the_gap_do_not_count(self):
        # Targets 0.55 vs 0.45 are extraction noise, not an ordering.
        loss = self._pair(-5.0, 5.0, target_current=0.55, target_ref=0.45)
        assert loss == 0.0

    def test_empty_reference_is_zero_with_grad(self):
        scores = torch.zeros(2, 12, requires_grad=True)
        loss = net.pairwise_rank_loss(
            scores, torch.ones(2, 12),
            torch.zeros(0, 12), torch.zeros(0, 12))
        assert loss == 0.0
        loss.backward()  # the zero must still be attached to the graph
        assert scores.grad is not None

    def test_zero_weights_silence_a_pair(self):
        loss = self._pair(-5.0, 5.0,
                          weights=torch.zeros(1, 12),
                          ref_weights=torch.ones(1, 12))
        assert loss == 0.0

    def test_bigger_target_gap_weighs_more(self):
        # Two reference studies: a large-gap pair ranked correctly and a
        # small-gap pair ranked wrongly. Gap weighting must pull the mean
        # toward the large-gap (low-loss) pair, below the unweighted mean.
        scores = torch.zeros(1, 1)
        targets = torch.zeros(1, 1)
        ref_scores = torch.tensor([[2.0], [-2.0]])
        ref_targets = torch.tensor([[0.9], [0.4]])

        combined = net.pairwise_rank_loss(scores, targets,
                                          ref_scores, ref_targets)
        each = [net.pairwise_rank_loss(scores, targets,
                                       ref_scores[i:i + 1],
                                       ref_targets[i:i + 1])
                for i in range(2)]
        assert combined < (each[0] + each[1]) / 2


class TestStratifiedRankLoss:
    """Stage 1: rank only within an effusion stratum.

    Conditional AUC is within-stratum pair concordance, so a loss that leaks
    across strata is optimising the metric the plan is trying to stop using.
    """

    def _loss(self, strata, ref_strata, findings=("Synovitis",), caliper=0.15,
              score=-2.0, ref_score=2.0):
        # A pair ordered wrongly: target 1.0 scored below target 0.0.
        return net.stratified_rank_loss(
            torch.full((1, 12), score), torch.ones(1, 12),
            torch.full((1, 12), ref_score), torch.zeros(1, 12),
            torch.tensor(strata), torch.tensor(ref_strata),
            findings, caliper)

    def test_a_pair_outside_the_caliper_is_dropped(self):
        assert self._loss([0.1], [0.9]) == 0.0

    def test_a_pair_inside_the_caliper_counts(self):
        assert self._loss([0.1], [0.2]) > 0.0

    def test_only_the_named_findings_receive_gradient(self):
        scores = torch.zeros(1, 12, requires_grad=True)
        loss = net.stratified_rank_loss(
            scores, torch.ones(1, 12),
            torch.full((1, 12), 2.0), torch.zeros(1, 12),
            torch.tensor([0.5]), torch.tensor([0.5]), ("Synovitis",))
        loss.backward()
        touched = (scores.grad.abs().sum(dim=0) > 0).tolist()
        assert touched == [name == "Synovitis" for name in config.LABELS]

    def test_it_reduces_to_the_plain_loss_at_an_infinite_caliper(self):
        stratified = net.stratified_rank_loss(
            torch.full((1, 12), -2.0), torch.ones(1, 12),
            torch.full((1, 12), 2.0), torch.zeros(1, 12),
            torch.tensor([0.0]), torch.tensor([1.0]),
            config.LABELS, caliper=10.0)
        plain = net.pairwise_rank_loss(
            torch.full((1, 12), -2.0), torch.ones(1, 12),
            torch.full((1, 12), 2.0), torch.zeros(1, 12))
        assert stratified == pytest.approx(float(plain))


class TestTheMarginClamp:
    """The guard for the failure that actually happened.

    A satisfied pair's loss decays as its margin grows; a violated pair's grows
    linearly and without bound. When the pairs contradict each other, inflating
    the logit scale pays -- the majority's loss falls faster than the
    minority's rises -- until the head saturates and every study ties. That run
    is on record: BCE 0.60 -> 5.99, macro 0.86 -> 0.71, conditional AUC exactly
    0.5000 from epoch 3 on.
    """

    def _violated(self, margin, **kwargs):
        # target 1.0 scored BELOW target 0.0 by `margin`: a violated pair.
        return net.pairwise_rank_loss(
            torch.full((1, 1), -margin / 2), torch.ones(1, 1),
            torch.full((1, 1), margin / 2), torch.zeros(1, 1), **kwargs)

    def test_an_unclamped_violated_pair_grows_without_bound(self):
        assert self._violated(40.0) > self._violated(20.0) > self._violated(10.0)

    def test_a_clamped_one_stops_growing(self):
        at_cap = self._violated(16.0, max_margin=8.0)
        far_past = self._violated(400.0, max_margin=8.0)
        assert at_cap == pytest.approx(float(far_past))

    def test_the_gradient_vanishes_past_the_cap(self):
        scores = torch.full((1, 1), -50.0, requires_grad=True)
        net.pairwise_rank_loss(
            scores, torch.ones(1, 1), torch.full((1, 1), 50.0),
            torch.zeros(1, 1), max_margin=8.0).backward()
        assert scores.grad.abs().max() == 0.0

    def test_ordinary_margins_are_untouched(self):
        # The clamp must not change the regime every earlier run trained in.
        for margin in (0.5, 2.0, 6.0):
            assert self._violated(margin, max_margin=8.0) == pytest.approx(
                float(self._violated(margin)))

    def test_the_stratified_loss_clamps_by_default(self):
        # The reference losses default to None and stay bit-identical; the one
        # whose pairs contradict each other does not get that default.
        import inspect
        assert inspect.signature(
            net.pairwise_rank_loss).parameters["max_margin"].default is None
        assert inspect.signature(
            net.stratified_rank_loss).parameters["max_margin"].default == 8.0


class TestCouplingPenalty:
    """The penalty must aim at the truth's coupling, not at independence."""

    @staticmethod
    def _scores(correlation, n=64):
        torch.manual_seed(0)
        parent = torch.randn(n)
        noise = torch.randn(n)
        child = correlation * parent + (1 - correlation ** 2) ** 0.5 * noise
        scores = torch.zeros(n, 12)
        scores[:, config.LABELS.index("Effusion")] = parent
        scores[:, config.LABELS.index("Synovitis")] = child
        return scores

    def test_it_is_smallest_at_the_target(self):
        child = config.LABELS.index("Synovitis")
        parent = config.LABELS.index("Effusion")
        empty = torch.zeros(0, 12)
        at_target = net.coupling_penalty(self._scores(0.4), empty, child,
                                         parent, 0.4)
        decoupled = net.coupling_penalty(self._scores(0.0), empty, child,
                                         parent, 0.4)
        overcoupled = net.coupling_penalty(self._scores(0.8), empty, child,
                                           parent, 0.4)
        assert at_target < decoupled
        assert at_target < overcoupled

    def test_too_few_studies_is_a_zero_that_keeps_the_graph(self):
        scores = torch.zeros(2, 12, requires_grad=True)
        loss = net.coupling_penalty(scores, torch.zeros(0, 12), 8, 7, 0.4)
        assert loss == 0.0
        loss.backward()
        assert scores.grad is not None


class TestResidualCoupling:
    """`syn_logit = beta * stopgrad(eff_logit) + g(features)`.

    The coupling has to travel in the checkpoint. Every construction site that
    had to be told about a model detail has eventually forgotten -- that is
    what the input-normalisation buffers cost -- so a bare `KneeNet` loading a
    residual checkpoint must reproduce it without being asked.
    """

    CHILD = config.LABELS.index("Synovitis")
    PARENT = config.LABELS.index("Effusion")

    def test_off_by_default(self):
        model = make_model()
        assert model.residual_pairs is None
        volumes, meta, mask = make_batch()
        assert torch.equal(model(volumes, meta, mask),
                           model.couple(model(volumes, meta, mask)))

    def test_the_parent_logit_is_added_with_beta(self):
        model = make_model()
        logits = torch.randn(3, len(config.LABELS))
        coupled = make_model(residual_pairs=[(self.CHILD, self.PARENT)],
                             residual_beta=0.5).couple(logits)
        expected = logits[:, self.CHILD] + 0.5 * logits[:, self.PARENT]
        assert torch.allclose(coupled[:, self.CHILD], expected)
        # Nothing else moves.
        other = [i for i in range(len(config.LABELS)) if i != self.CHILD]
        assert torch.allclose(coupled[:, other], logits[:, other])

    def test_the_parent_receives_no_gradient_through_the_coupling(self):
        model = make_model(residual_pairs=[(self.CHILD, self.PARENT)],
                           residual_beta=0.9)
        logits = torch.randn(4, len(config.LABELS), requires_grad=True)
        model.couple(logits)[:, self.CHILD].sum().backward()
        assert logits.grad[:, self.PARENT].abs().max() == 0.0
        assert logits.grad[:, self.CHILD].abs().min() > 0.0

    def test_it_travels_in_the_state_dict(self):
        trained = make_model(residual_pairs=[(self.CHILD, self.PARENT)],
                             residual_beta=0.6)
        volumes, meta, mask = make_batch()
        expected = trained(volumes, meta, mask)

        # A scoring site that knows nothing about the coupling.
        scoring = make_model()
        scoring.load_state_dict(trained.state_dict())
        assert torch.allclose(scoring(volumes, meta, mask), expected)
        assert scoring.residual_beta.tolist() == pytest.approx([0.6], abs=1e-6)

    def test_an_older_checkpoint_still_loads_strictly(self):
        older = make_model().state_dict()
        assert not any(key.startswith("residual") for key in older)
        make_model().load_state_dict(older)

    def test_mismatched_beta_count_is_refused(self):
        with pytest.raises(ValueError):
            make_model(residual_pairs=[(8, 7), (9, 7)],
                       residual_beta=[0.5])
        with pytest.raises(ValueError):
            make_model(residual_pairs=[(8, 7)])


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


class TestBuildBackbone:
    """Both notebooks construct their trunk here, so the two cannot diverge.

    The interesting behaviour is the input-size argument: ViT-family weights
    need it and ConvNets reject it, and getting that wrong surfaces either as
    a crash at submission time or as interpolated position embeddings that
    were never asked for. timm is faked so the test states the contract
    without downloading weights.
    """

    def _fake_timm(self, monkeypatch, accepts_img_size):
        calls = []

        class FakeTimm:
            @staticmethod
            def create_model(name, **kwargs):
                calls.append(kwargs)
                if "img_size" in kwargs and not accepts_img_size:
                    raise TypeError("unexpected keyword argument 'img_size'")
                return nn.Identity()

        monkeypatch.setitem(sys.modules, "timm", FakeTimm)
        return calls

    def test_vit_family_receives_the_cache_size(self, monkeypatch):
        calls = self._fake_timm(monkeypatch, accepts_img_size=True)
        net.build_backbone("vit_small_patch14_reg4_dinov2.lvd142m")
        assert len(calls) == 1
        assert calls[0]["img_size"] == config.CACHE_SIZE

    def test_a_convnet_is_retried_without_the_argument(self, monkeypatch):
        calls = self._fake_timm(monkeypatch, accepts_img_size=False)
        net.build_backbone("convnext_tiny.fb_in22k_ft_in1k")
        assert len(calls) == 2
        assert "img_size" not in calls[1]

    def test_channels_match_the_2p5d_stack(self, monkeypatch):
        calls = self._fake_timm(monkeypatch, accepts_img_size=True)
        net.build_backbone("anything", pretrained=False)
        assert calls[0]["in_chans"] == net.SLICE_CONTEXT
        assert calls[0]["pretrained"] is False
        assert calls[0]["num_classes"] == 0

    def test_the_cache_size_divides_the_dinov2_patch_grid(self):
        # 224 = 16 x 14. A cache size that did not divide by 14 would make the
        # position-embedding interpolation silently lossy rather than exact.
        assert config.CACHE_SIZE % 14 == 0


class TestBackboneDefaults:
    """ConvNeXt must keep the exact behaviour its 0.861 reference was trained
    under, while a ViT gets the normalisation and trunk rate it needs."""

    def test_convnext_keeps_the_original_behaviour(self):
        d = net.backbone_defaults("convnext_tiny.fb_in22k_ft_in1k")
        assert d["input_mean"] is None and d["input_std"] is None
        assert d["backbone_lr_scale"] == 1.0

    def test_a_dinov2_trunk_asks_for_normalisation_and_a_slower_rate(self):
        d = net.backbone_defaults("vit_small_patch14_reg4_dinov2.lvd142m")
        assert d["input_mean"] == net.IMAGENET_MEAN
        assert d["backbone_lr_scale"] < 1.0

    def test_normalisation_changes_the_input_the_backbone_sees(self):
        seen = {}

        class Spy(nn.Module):
            def forward(self, x):
                seen["mean"] = float(x.mean())
                return torch.zeros(x.shape[0], 8)

        volumes = torch.full((1, 2, 4, 8, 8), 255, dtype=torch.uint8)
        plain = net.KneeNet(Spy(), 8)
        plain.encode_series(volumes.reshape(2, 4, 8, 8))
        raw = seen["mean"]

        normed = net.KneeNet(Spy(), 8, input_mean=net.IMAGENET_MEAN,
                             input_std=net.IMAGENET_STD)
        normed.encode_series(volumes.reshape(2, 4, 8, 8))
        assert abs(seen["mean"] - raw) > 0.5

    def test_a_convnext_state_dict_gains_no_keys(self):
        # ConvNeXt trains with no normalisation, so nothing new appears in its
        # checkpoint and every file this project has already shipped still
        # loads strictly against the current class.
        plain = net.KneeNet(nn.Linear(4, 8), 8)
        assert not [k for k in plain.state_dict() if k.startswith("input_")]

    def test_the_statistics_travel_with_the_weights(self):
        normed = net.KneeNet(nn.Linear(4, 8), 8, input_mean=net.IMAGENET_MEAN,
                             input_std=net.IMAGENET_STD)
        state = normed.state_dict()
        assert "input_mean" in state and "input_std" in state

    def test_a_bare_model_adopts_the_checkpoint_s_statistics(self):
        # The bug this replaced: a scoring site that forgot the statistics
        # loaded every weight strictly and silently scored without them.
        normed = net.KneeNet(nn.Linear(4, 8), 8, input_mean=net.IMAGENET_MEAN,
                             input_std=net.IMAGENET_STD)
        bare = net.KneeNet(nn.Linear(4, 8), 8)
        assert bare.input_mean is None
        bare.load_state_dict(normed.state_dict())
        assert torch.equal(bare.input_mean, normed.input_mean)
        assert torch.equal(bare.input_std, normed.input_std)

    def test_an_older_checkpoint_still_loads_strictly(self):
        # Files written before the buffers were persistent carry neither key.
        normed = net.KneeNet(nn.Linear(4, 8), 8, input_mean=net.IMAGENET_MEAN,
                             input_std=net.IMAGENET_STD)
        old = {k: v for k, v in normed.state_dict().items()
               if not k.startswith("input_")}
        fresh = net.KneeNet(nn.Linear(4, 8), 8, input_mean=net.IMAGENET_MEAN,
                            input_std=net.IMAGENET_STD)
        fresh.load_state_dict(old, strict=True)
        assert fresh.input_mean is not None
