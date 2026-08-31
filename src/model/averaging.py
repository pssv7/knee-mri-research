"""Weight averaging: EMA, SWA, and model soups.

**Why this module exists.** Checkpoint selection here is broken in a specific,
diagnosable way. Training labels are mined from radiology reports and agree
with image-derived ground truth about 82% of the time, and the validation fold
is mined from the *same* distribution. So the network first fits the structure
the two sources share and then memorises the 18% they do not -- Arpit et al.,
"A Closer Look at Memorization in Deep Networks" (ICML 2017), and Liu et al.,
"Early-Learning Regularization Prevents Memorization of Noisy Labels" (NeurIPS
2020). Mined validation agreement keeps *rising* through that second phase,
because agreeing with the noise is what it measures. It is therefore a biased
selector, not merely a noisy one, and "take the best epoch on validation"
selects into the memorisation phase on purpose.

Measured here: mined val rose to 0.860 at epoch 9 and fell to 0.843 by epoch
20, while gold held flat at 0.864-0.869 across eleven consecutive epochs.

The fix is to stop selecting. Averaging weights is selection-free:

- **EMA** (Tarvainen & Valpola, NeurIPS 2017) -- an exponential moving average
  maintained during training. Cheap, standard, and already the default in most
  detection and segmentation pipelines.
- **SWA** (Izmailov et al., UAI 2018, arXiv:1803.05407) -- an equal-weight
  average over the tail of the trajectory, which lands in a flatter and better
  generalising region than any single point on it.
- **Model soups** (Wortsman et al., ICML 2022, arXiv:2203.05482) -- average
  the weights of several finished fine-tunes instead of picking the best one
  on a held-out set. The paper's argument is directed at exactly the recipe
  failing here: "pick the individual model which performs best on a held-out
  validation set, discarding the remainder".

**Averaging weights is only meaningful for models in the same basin.** Two
runs whose aggregation was randomly initialised from different seeds are not,
and averaging them produces a model that is worse than either -- silently, as
a plausible-looking set of numbers. `assert_soupable` checks what it can
(shapes and key sets); it cannot check the basin, so the training notebook
seeds every fold identically, and the docstrings below say when a soup is
well-posed.

**Nothing here may select on the gold 58.** `greedy_soup` takes a caller's
score function precisely so the choice of scoring set stays visible at the
call site: it must be mined validation or an out-of-fold dump. Selecting a
soup on gold would turn the project's one honest measuring stick into a
model-selection set, which has already cost this project an inflated figure
once.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

State = Dict[str, torch.Tensor]


def _is_averageable(tensor: torch.Tensor) -> bool:
    """Floating-point tensors average; integer counters and flags do not.

    `num_batches_tracked` on a BatchNorm is the canonical example: averaging
    it produces a fractional count, and torch stores it as int64 anyway. The
    aggregation stack is all LayerNorm so this does not arise today, but a
    convnet backbone with BatchNorm would hit it immediately.
    """
    return tensor.is_floating_point()


def assert_soupable(states: Sequence[State]) -> None:
    """Refuse states that cannot be averaged elementwise.

    Catches the two mechanical failures -- a different architecture, and a
    checkpoint saved at a different width -- before they become a tensor of
    the wrong shape several cells later.
    """
    if not states:
        raise ValueError("no states to average")
    reference = states[0]
    for i, state in enumerate(states[1:], start=1):
        if set(state) != set(reference):
            missing = sorted(set(reference) - set(state))[:3]
            extra = sorted(set(state) - set(reference))[:3]
            raise ValueError(
                f"state {i} has different keys from state 0 "
                f"(missing {missing}, extra {extra}) -- these are not the "
                "same architecture")
        for key, tensor in state.items():
            if tensor.shape != reference[key].shape:
                raise ValueError(
                    f"state {i} has {key} at {tuple(tensor.shape)} against "
                    f"{tuple(reference[key].shape)} in state 0")


def average_states(states: Sequence[State],
                   weights: Optional[Sequence[float]] = None) -> State:
    """Elementwise weighted average of state dicts.

    `weights` defaults to equal, which is SWA and the uniform soup. Averaging
    happens in float64 and is cast back to each tensor's own dtype, so a run
    of fp16 checkpoints does not accumulate rounding across a dozen members.

    Non-floating tensors are taken from the first state rather than averaged.
    """
    assert_soupable(states)
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError(f"{len(weights)} weights for {len(states)} states")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to something positive")

    averaged: State = {}
    for key, reference in states[0].items():
        if not _is_averageable(reference):
            averaged[key] = reference.clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for state, weight in zip(states, weights):
            accumulator += state[key].to(torch.float64) * float(weight)
        averaged[key] = (accumulator / total).to(reference.dtype)
    return averaged


class WeightAverage:
    """EMA and SWA of one model's weights, maintained during training.

    Both are accumulated from the same call so a run pays one copy of the
    weights per update and produces both artefacts. `decay` is the EMA's, and
    0.999 is the usual choice; `swa_from` is the step at which the equal-weight
    tail average starts, so a caller sets it to 75% of the planned steps to
    average over the last quarter.

    **The averages live wherever the model does, in float32.** Accumulating on
    the host instead would copy the whole trunk across the PCIe bus on every
    optimiser step -- ~110 MB for ConvNeXt-T plus the aggregation, twice per
    step once the SWA window opens, about 11,000 times over a twenty-epoch
    run. That is a wall-clock tax on the run for no benefit; two shadows on
    the device cost ~220 MB of a 16 GB T4, which the batch has to spare.
    Float32 regardless of training precision: under `16-mixed` the parameters
    are fp32 anyway (autocast does not change them), but an fp16 accumulation
    over thousands of steps would round away exactly the small updates an EMA
    exists to keep. `export` moves the result to the host.

    `device` overrides the placement -- pass "cpu" if a run is ever tight
    enough on device memory that 220 MB matters.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 swa_from: Optional[int] = None, device=None):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.swa_from = swa_from
        self.steps = 0
        self.swa_count = 0
        state = model.state_dict()
        self.ema: State = {
            k: (v.detach().clone().to(torch.float32) if device is None
                else v.detach().to(device, torch.float32).clone())
            for k, v in state.items() if _is_averageable(v)}
        self.swa: State = {k: torch.zeros_like(v) for k, v in self.ema.items()}
        # Everything not averaged -- integer counters. The normalisation
        # buffers are float, so they average harmlessly to themselves.
        self._fixed: State = {k: v.detach().cpu().clone()
                              for k, v in state.items()
                              if not _is_averageable(v)}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Fold one optimiser step's weights into both averages."""
        self.steps += 1
        state = model.state_dict()
        in_window = (self.swa_from is not None and self.steps >= self.swa_from)
        if in_window:
            self.swa_count += 1

        for key, shadow in self.ema.items():
            # Fetched once and reused by both averages: reading it per average
            # doubled the copy for every step of the SWA window.
            current = state[key].detach().to(shadow.device, torch.float32,
                                             non_blocking=True)
            shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            if in_window:
                accumulator = self.swa[key]
                # Running mean, so the tail can be any length without a second
                # pass or a list of checkpoints held in memory.
                accumulator.add_((current - accumulator) / self.swa_count)

    def ema_state(self) -> State:
        """The EMA weights, as a state dict a `KneeNet` loads.

        Moved to the host on the way out: a checkpoint holding CUDA tensors
        reloads onto a device that may not exist in the process reading it.
        """
        return {**self._fixed, **{k: v.detach().cpu().clone()
                                  for k, v in self.ema.items()}}

    def swa_state(self) -> Optional[State]:
        """The tail average, or None if no step fell inside the window."""
        if self.swa_count == 0:
            return None
        return {**self._fixed, **{k: v.detach().cpu().clone()
                                  for k, v in self.swa.items()}}


def greedy_soup(states: Sequence[State], score: Callable[[State], float],
                names: Optional[Sequence[str]] = None,
                tolerance: float = 0.0) -> Tuple[State, List[str]]:
    """Wortsman et al.'s greedy soup: add a member only if it helps.

    Members are ranked by their individual score, then considered in order;
    each is kept if the average *including* it scores at least as well as the
    average without it. The result is never worse than the best single member
    on the scoring set, which is what makes greedy preferable to a uniform
    soup when the members are of uneven quality.

    `score` maps a state dict to a number that is better when larger --
    macro AUC here. **It must not read the gold 58.** Score on the mined
    validation fold or on an out-of-fold dump; the whole reason this takes a
    callable rather than a dataset is to keep that choice at the call site
    where a reviewer can see it.

    `tolerance` lets a member in when it is within that much of the running
    best, for the case where the scoring set is too small to resolve the
    difference and averaging more members is the safer default. At 0.0 this is
    the paper's algorithm exactly.

    Returns the souped state and the names of the members that went into it,
    in the order they were added.
    """
    assert_soupable(states)
    if names is None:
        names = [f"member{i}" for i in range(len(states))]
    if len(names) != len(states):
        raise ValueError(f"{len(names)} names for {len(states)} states")

    ranked = sorted(zip(names, states), key=lambda pair: score(pair[1]),
                    reverse=True)

    kept_names: List[str] = [ranked[0][0]]
    kept_states: List[State] = [ranked[0][1]]
    best = score(kept_states[0])

    for name, state in ranked[1:]:
        candidate = average_states(kept_states + [state])
        value = score(candidate)
        if value >= best - tolerance:
            kept_names.append(name)
            kept_states.append(state)
            best = max(best, value)

    return average_states(kept_states), kept_names


def rank_average(predictions: Sequence["np.ndarray"]) -> "np.ndarray":
    """Average ensemble members by within-member rank, not by probability.

    The metric is macro ROC-AUC, which reads only the ordering of studies
    within a finding. Averaging probabilities lets a member with a compressed
    or shifted output distribution contribute less than one with a confident
    one, even when the two rank studies identically -- the mean is sensitive to
    calibration and AUC is not. Converting each member to ranks first removes
    that: every member gets one vote per finding, whatever its scale.

    Each member is (N studies, L findings); ranking is per finding, per member,
    with ties averaged, and the result is scaled to [0, 1] so it can be written
    into a submission unchanged. Because AUC is invariant to any increasing
    transform, that scaling costs nothing and keeps the output looking like the
    probabilities the rest of the pipeline passes around.

    A single member is returned rank-transformed rather than untouched, so that
    a one-model and a five-model submission are produced by the same code path.
    """
    import numpy as np

    if not predictions:
        raise ValueError("no predictions to average")
    stacked = np.stack([np.asarray(p, dtype=float) for p in predictions])
    if stacked.ndim != 3:
        raise ValueError(
            f"expected (members, studies, findings), got {stacked.shape}")

    n_studies = stacked.shape[1]
    if n_studies < 2:
        # Ranks are meaningless for a single study, and the scaling below
        # would divide by zero.
        return stacked.mean(axis=0)

    ranked = np.empty_like(stacked)
    for member in range(stacked.shape[0]):
        for finding in range(stacked.shape[2]):
            column = stacked[member, :, finding]
            order = np.argsort(column, kind="mergesort")
            positions = np.empty(n_studies, dtype=float)
            positions[order] = np.arange(n_studies, dtype=float)
            # Average the positions of tied values, so a member that predicts
            # a constant contributes no ordering rather than an arbitrary one.
            unique, inverse, counts = np.unique(column, return_inverse=True,
                                                return_counts=True)
            sums = np.zeros(len(unique))
            np.add.at(sums, inverse, positions)
            ranked[member, :, finding] = (sums / counts)[inverse]

    return ranked.mean(axis=0) / (n_studies - 1)
