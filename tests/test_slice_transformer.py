"""Behavior and provenance checks for the protocol slice-sequence experiment."""

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import net
from src.model.experiments import get_experiment
from src.model.lightning import KneeLitModule
from tools.prepare_kaggle_training_pair import ROOT, training_source
from tools.kaggle_training_pair_body import validate_contracts


def test_shared_initialization_identity_start_and_checkpoint_roundtrip():
    states = []
    for enabled in (False, True):
        torch.manual_seed(14)
        module = KneeLitModule(torch.nn.Linear(8, 8), 8, slice_transformer=enabled)
        states.append(module.state_dict())
    shared = {k for k in states[1] if not k.startswith("model.slice_encoder.")}
    assert shared == set(states[0])
    assert all(torch.equal(states[0][k], states[1][k]) for k in shared)

    candidate = net.KneeNet(None, 8, slice_transformer=True).eval()
    x = torch.randn(2, 16, 8)
    assert torch.equal(candidate.slice_encoder(x), x)
    defaults = net.aggregation_defaults(candidate.state_dict())
    assert defaults["slice_transformer"]
    clone = net.KneeNet(None, 8, **defaults).eval()
    clone.load_state_dict(candidate.state_dict(), strict=True)
    assert torch.equal(candidate.slice_encoder(x), clone.slice_encoder(x))
    assert not net.aggregation_defaults(states[0])["slice_transformer"]


def test_slice_context_learns_and_positions_break_permutation_equivariance():
    torch.manual_seed(2)
    encoder = net.SliceSequenceEncoder(8, width=16, layers=2, dropout=0).eval()
    x = torch.randn(2, 5, 8)
    opt = torch.optim.SGD(encoder.parameters(), lr=0.1)
    encoder(x).square().mean().backward()
    assert encoder.output.weight.grad.abs().sum() > 0
    opt.step()
    opt.zero_grad()
    encoder(x).square().mean().backward()
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    original = encoder(x)
    changed = x.clone()
    changed[:, -1] += torch.arange(8).float()
    assert not torch.allclose(original[:, 0], encoder(changed)[:, 0])
    assert not torch.allclose(original, encoder(x.flip(1)).flip(1))
    for depth in (1, 16, 24):
        assert torch.isfinite(encoder(torch.randn(1, depth, 8))).all()


@pytest.mark.parametrize("fold", [1, 3])
def test_protocol_control_and_candidate_have_same_input_policy(fold):
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    for name in ("baseline", "slice_transformer"):
        source = training_source(notebook, name, "hash", fold=fold, match_slice=True)
        ast.parse(source)
        assert "RUN = replace(RUN, series_selection='protocol')" in source
        assert "slice_transformer=RUN.slice_transformer" in source
        assert "MATCH_SLICE = True" in source
        assert "('model.slice_encoder.',)" in source


def test_slice_contract_accepts_only_shared_protocol_recipe():
    control = replace(get_experiment("baseline"), series_selection="protocol")
    contracts = [{"experiment": exp.to_dict(), "seed": 0,
                  "initial_weights_sha256": exp.name,
                  "shared_initial_weights_sha256": "same"}
                 for exp in (control, get_experiment("slice_transformer"))]
    validate_contracts(contracts, "slice_transformer")
    contracts[0]["experiment"]["series_selection"] = "uniform"
    with pytest.raises(ValueError, match="protocol selection"):
        validate_contracts(contracts, "slice_transformer")
    contracts[0]["experiment"]["series_selection"] = "protocol"
    contracts[1]["shared_initial_weights_sha256"] = "drift"
    with pytest.raises(ValueError, match="initial weights"):
        validate_contracts(contracts, "slice_transformer")


def test_disabled_encoder_preserves_legacy_state_and_masked_series():
    class Trunk(torch.nn.Module):
        def forward(self, x):
            return x.mean((2, 3)).repeat(1, 8)[:, :8]
    model = net.KneeNet(Trunk(), 8, slice_transformer=True).eval()
    volumes = torch.rand(1, 2, 4, 8, 8)
    meta = torch.zeros(1, 2, len(net.META_COLUMNS))
    mask = torch.tensor([[True, False]])
    with torch.no_grad():
        a = model(volumes, meta, mask)
        b = model(volumes[:, :1], meta[:, :1], mask[:, :1])
    assert torch.allclose(a, b, atol=1e-6)


def test_no_timeout_setting_reaches_both_subprocesses(tmp_path):
    from tools.prepare_kaggle_training_pair import prepare
    # Stage using the real source and labels, but isolate generated artifacts.
    for path in ("notebooks", "data", "tools", "outputs/kaggle_sources"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    prepare(root=tmp_path, candidate="slice_transformer", no_timeout=True)
    p = tmp_path / "outputs/kaggle_slice_transformer_fold1"
    tree = ast.parse((p / "train_pair.py").read_text())
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("SMOKE_TIMEOUT", "RUN_TIMEOUT")}
    assert values == {"SMOKE_TIMEOUT": None, "RUN_TIMEOUT": None}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "subprocess.run"]
    assert len(calls) == 2
    for call in calls:
        value = next(k.value for k in call.keywords if k.arg == "timeout")
        assert values[value.id] is None
    assert json.loads((p / "staging_receipt.json").read_text())["per_run_timeout_seconds"] is None
