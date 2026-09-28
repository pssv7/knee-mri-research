import ast
import json

import pytest

from tools.kaggle_training_pair_body import screening_gate
from tools.prepare_kaggle_training_pair import ROOT, prepare


@pytest.mark.parametrize("gain,drop,passed", [(.004999, 0, False), (.005, -.020, True),
                                             (.006, -.020001, False), (.006, 0, True),
                                             (float("nan"), 0, False)])
def test_spatial_screen_boundaries(gain, drop, passed):
    gate = screening_gate(gain, [drop] * 12, .005)
    assert gate["passed"] is passed
    assert gate["required_macro_delta"] == .005


@pytest.mark.parametrize("candidate,threshold", [("spatial", .005), ("protocol", .003),
                                               ("classwise", .003), ("convnextv2", .003),
                                               ("slice_transformer", .003)])
def test_staging_and_executable_share_gate(tmp_path, candidate, threshold):
    for path in ("notebooks", "data", "tools", "outputs/kaggle_sources"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    prepare(root=tmp_path, candidate=candidate)
    folder = tmp_path / ("outputs/kaggle_training_pair" if candidate == "protocol"
                         else f"outputs/kaggle_{candidate}_fold1")
    receipt = json.loads((folder / "staging_receipt.json").read_text())
    tree = ast.parse((folder / "train_pair.py").read_text())
    configured = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "REQUIRED_MACRO_DELTA")
    assert configured == receipt["required_macro_delta"] == threshold
    assert screening_gate(.003, [0] * 12, configured)["passed"] is (candidate != "spatial")
