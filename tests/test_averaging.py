"""Tests for weight averaging: EMA, SWA, and soups.

The failures worth guarding are the quiet ones. An average that silently
includes a member twice, an EMA whose effective decay changes with the
accumulation setting, a soup built from two architectures that happen to share
a key -- none of these raise, and each produces a model that merely scores
worse than it should.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import averaging  # noqa: E402


def state(value, extra=None):
    """A tiny state dict whose every tensor is `value`."""
    out = {"w": torch.full((2, 2), float(value)),
           "b": torch.tensor([float(value)])}
    if extra is not None:
        out.update(extra)
    return out


class TestAverageStates:
    def test_equal_weights_are_the_mean(self):
        result = averaging.average_states([state(1), state(3)])
        assert result["w"].tolist() == [[2.0, 2.0], [2.0, 2.0]]
        assert result["b"].tolist() == [2.0]

    def test_weights_are_honoured(self):
        result = averaging.average_states([state(0), state(4)], [3.0, 1.0])
        assert result["b"].item() == pytest.approx(1.0)

    def test_a_single_state_is_returned_unchanged(self):
        result = averaging.average_states([state(7)])
        assert result["b"].item() == pytest.approx(7.0)

    def test_the_result_keeps_each_tensor_s_dtype(self):
        half = {k: v.half() for k, v in state(1).items()}
        result = averaging.average_states([half, half])
        assert result["w"].dtype == torch.float16

    def test_integer_tensors_are_taken_not_averaged(self):
        # A BatchNorm's num_batches_tracked is int64; averaging it would
        # produce a fraction, and torch cannot store one there.
        counted = torch.tensor(4)
        result = averaging.average_states([state(1, {"n": counted}),
                                           state(3, {"n": torch.tensor(10)})])
        assert result["n"].dtype == counted.dtype
        assert result["n"].item() == 4

    def test_accumulation_does_not_lose_precision_across_many_members(self):
        # Averaged in float64 and cast back, so a long list of fp16 members
        # does not drift the way a running fp16 sum would.
        members = [{k: v.half() for k, v in state(1).items()}
                   for _ in range(1000)]
        result = averaging.average_states(members)
        assert result["b"].item() == pytest.approx(1.0)

    def test_no_states_is_an_error(self):
        with pytest.raises(ValueError, match="no states"):
            averaging.average_states([])

    def test_mismatched_weight_count_is_an_error(self):
        with pytest.raises(ValueError, match="weights for"):
            averaging.average_states([state(1), state(2)], [1.0])


class TestAssertSoupable:
    def test_different_keys_are_refused(self):
        with pytest.raises(ValueError, match="different keys"):
            averaging.assert_soupable([state(1), {"w": torch.zeros(2, 2)}])

    def test_different_shapes_are_refused(self):
        other = {"w": torch.zeros(3, 3), "b": torch.zeros(1)}
        with pytest.raises(ValueError, match="w at"):
            averaging.assert_soupable([state(1), other])

    def test_matching_states_pass(self):
        averaging.assert_soupable([state(1), state(2), state(3)])


class Tiny(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.fill_(value)
            self.linear.bias.fill_(value)

    def set(self, value):
        with torch.no_grad():
            self.linear.weight.fill_(value)
            self.linear.bias.fill_(value)


class TestWeightAverage:
    def test_ema_tracks_the_declared_decay(self):
        model = Tiny(0.0)
        average = averaging.WeightAverage(model, decay=0.5)
        model.set(1.0)
        average.update(model)
        # 0.5 * 0 + 0.5 * 1
        assert average.ema["linear.bias"][0].item() == pytest.approx(0.5)
        average.update(model)
        assert average.ema["linear.bias"][0].item() == pytest.approx(0.75)

    def test_ema_state_loads_into_the_model(self):
        model = Tiny(1.0)
        average = averaging.WeightAverage(model, decay=0.9)
        model.set(2.0)
        average.update(model)
        model.load_state_dict(average.ema_state())
        assert model.linear.bias[0].item() == pytest.approx(1.1)

    def test_swa_averages_only_the_tail(self):
        model = Tiny(0.0)
        average = averaging.WeightAverage(model, swa_from=3)
        for value in (10.0, 20.0, 1.0, 2.0, 3.0):
            model.set(value)
            average.update(model)
        # Steps 3, 4, 5 hold 1, 2, 3; the first two are outside the window.
        assert average.swa_count == 3
        assert average.swa_state()["linear.bias"][0].item() == pytest.approx(2.0)

    def test_swa_is_absent_when_no_step_reached_the_window(self):
        model = Tiny(0.0)
        average = averaging.WeightAverage(model, swa_from=100)
        model.set(1.0)
        average.update(model)
        assert average.swa_state() is None

    def test_swa_is_off_when_no_window_was_given(self):
        model = Tiny(0.0)
        average = averaging.WeightAverage(model)
        model.set(1.0)
        average.update(model)
        assert average.swa_state() is None

    def test_a_decay_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="decay"):
                averaging.WeightAverage(Tiny(), decay=bad)

    def test_the_average_is_independent_of_the_live_model(self):
        # The shadow copies rather than aliases; otherwise the "average" is
        # just the current weights wearing a different name.
        model = Tiny(0.0)
        average = averaging.WeightAverage(model, decay=0.9)
        model.set(5.0)
        assert average.ema["linear.bias"][0].item() == pytest.approx(0.0)


class TestTheResidualCouplingSurvivesAveraging:
    """A run with the residual head ships three checkpoints, not one.

    `residual_pairs` is a long tensor, so it is not averageable and travels in
    `_fixed`; `residual_beta` is float and is averaged, which is harmless only
    because it never changes during training. If either were dropped, the EMA
    and SWA files would load into a bare KneeNet as an *uncoupled* model and
    score as one -- silently, since nothing raises. That is the same shape of
    failure the input-normalisation buffers already cost this project once.
    """

    CHILD, PARENT = 8, 7

    def _model(self):
        from src.model import net
        return net.KneeNet(nn.Identity(), 8, input_dim=8,
                           residual_pairs=[(self.CHILD, self.PARENT)],
                           residual_beta=0.64)

    def _averaged(self):
        model = self._model()
        average = averaging.WeightAverage(model, decay=0.9, swa_from=1)
        for _ in range(5):
            average.update(model)
        return average

    @pytest.mark.parametrize("kind", ("ema_state", "swa_state"))
    def test_a_bare_knee_net_loads_the_average_still_coupled(self, kind):
        from src.model import net
        exported = getattr(self._averaged(), kind)()
        bare = net.KneeNet(nn.Identity(), 8, input_dim=8)
        bare.load_state_dict(exported)
        assert bare.residual_pairs.tolist() == [[self.CHILD, self.PARENT]]
        assert float(bare.residual_beta[0]) == pytest.approx(0.64, abs=1e-5)

    def test_the_integer_buffer_is_not_averaged_into_a_float(self):
        average = self._averaged()
        assert "residual_pairs" in average._fixed
        assert average._fixed["residual_pairs"].dtype == torch.long


class TestGreedySoup:
    def _score_towards(self, target):
        """Higher when the state's value is nearer `target`."""
        return lambda s: -abs(s["b"].item() - target)

    def test_it_keeps_the_members_that_help(self):
        # Best single is 4; adding 6 averages to 5 and scores better; adding
        # 100 would wreck it and must be refused.
        states = [state(4), state(6), state(100)]
        souped, kept = averaging.greedy_soup(
            states, self._score_towards(5.0), names=["a", "b", "c"])
        assert kept == ["a", "b"]
        assert souped["b"].item() == pytest.approx(5.0)

    def test_it_is_never_worse_than_the_best_member(self):
        score = self._score_towards(5.0)
        states = [state(4), state(50), state(100)]
        souped, kept = averaging.greedy_soup(states, score)
        assert score(souped) >= max(score(s) for s in states)
        assert len(kept) == 1

    def test_tolerance_admits_a_member_inside_the_noise(self):
        score = self._score_towards(5.0)
        states = [state(5), state(7)]
        _, kept = averaging.greedy_soup(states, score, names=["a", "b"],
                                        tolerance=2.0)
        assert kept == ["a", "b"]

    def test_names_must_match_the_states(self):
        with pytest.raises(ValueError, match="names for"):
            averaging.greedy_soup([state(1), state(2)], lambda s: 0.0,
                                  names=["only-one"])

    def test_default_names_are_generated(self):
        _, kept = averaging.greedy_soup([state(1)], lambda s: 0.0)
        assert kept == ["member0"]


class TestRankAverage:
    """Combining ensemble members for a rank metric.

    Macro AUC reads only the ordering of studies within a finding, so a member
    whose outputs are confident should not outvote one whose outputs are
    compressed when the two agree on the order. A probability mean lets it.
    """

    def _matrix(self, *columns):
        import numpy as np
        return np.array(columns, dtype=float).T.reshape(len(columns[0]), -1)

    def test_scale_does_not_buy_influence(self):
        # Two members that agree on the ORDER and disagree wildly on scale.
        # Under a rank average they are interchangeable, and the pair says
        # exactly what either says alone. Under a probability mean the
        # confident one would dominate.
        import numpy as np
        confident = np.array([[0.01], [0.99], [0.02], [0.98]])
        timid = np.array([[0.50], [0.53], [0.51], [0.52]])
        assert list(np.argsort(confident.ravel())) == \
            list(np.argsort(timid.ravel()))

        together = averaging.rank_average([confident, timid]).ravel()
        alone = averaging.rank_average([confident]).ravel()
        assert together == pytest.approx(alone)

    def test_a_probability_mean_would_have_been_dominated(self):
        # The contrast the rank average exists for, asserted so the claim in
        # the docstring is measured rather than stated.
        import numpy as np
        confident = np.array([[0.01], [0.99], [0.02], [0.98]])
        timid = np.array([[0.50], [0.53], [0.51], [0.52]])
        mean = np.mean([confident, timid], axis=0).ravel()
        # The mean's spread is set almost entirely by the confident member.
        assert mean.max() - mean.min() > 0.4
        ranked = averaging.rank_average([confident, timid]).ravel()
        assert ranked.max() - ranked.min() == pytest.approx(1.0)

    def test_opposite_members_cancel(self):
        import numpy as np
        up = np.array([[0.1], [0.2], [0.3], [0.4]])
        down = np.array([[0.4], [0.3], [0.2], [0.1]])
        out = averaging.rank_average([up, down]).ravel()
        assert out == pytest.approx([0.5] * 4)

    def test_a_constant_member_contributes_no_ordering(self):
        import numpy as np
        real = np.array([[0.1], [0.9], [0.2], [0.8]])
        flat = np.full((4, 1), 0.5)
        blended = averaging.rank_average([real, flat]).ravel()
        alone = averaging.rank_average([real]).ravel()
        # Ties average, so the flat member shifts every study equally and the
        # ORDER is untouched.
        assert list(np.argsort(blended)) == list(np.argsort(alone))

    def test_each_finding_is_ranked_separately(self):
        import numpy as np
        member = np.array([[0.9, 0.1], [0.1, 0.9]])
        out = averaging.rank_average([member])
        assert out[0, 0] > out[1, 0]
        assert out[1, 1] > out[0, 1]

    def test_the_output_is_bounded_and_ordered(self):
        import numpy as np
        rng = np.random.default_rng(0)
        members = [rng.random((30, 12)) for _ in range(5)]
        out = averaging.rank_average(members)
        assert out.shape == (30, 12)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_a_single_study_falls_back_to_the_mean(self):
        import numpy as np
        out = averaging.rank_average([np.array([[0.2]]), np.array([[0.4]])])
        assert out.ravel()[0] == pytest.approx(0.3)

    def test_a_single_member_is_rank_transformed_not_passed_through(self):
        # One code path for one model and for five.
        import numpy as np
        out = averaging.rank_average([np.array([[0.05], [0.9], [0.5]])])
        assert out.ravel() == pytest.approx([0.0, 1.0, 0.5])

    def test_empty_and_malformed_input_are_refused(self):
        import numpy as np
        with pytest.raises(ValueError, match="no predictions"):
            averaging.rank_average([])
        with pytest.raises(ValueError, match="members, studies, findings"):
            averaging.rank_average([np.array([1.0, 2.0])])
