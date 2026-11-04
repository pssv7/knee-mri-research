import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.validation.proxy import ID, LABELS, sha
from tools import label_repair_pair as pair
from tools.train_nested_proxy_control import run
from test_validation_fallback import frozen_design


@pytest.fixture
def staged(frozen_design, tmp_path):
    original, inputs, spec, roles = frozen_design
    design = tmp_path / "pair"; design.mkdir()
    shutil.copyfile(original.parent / "exclusions.csv", design / "exclusions.csv")
    (design / "label_receipt.json").write_text('{}')
    change = {ID: "s0", "finding": "ACL", "old_target": .1, "new_target": .8}
    (design / "changes.jsonl").write_text(json.dumps(change) + '\n')
    spec["runtime_code_sha256"]["tools/train_nested_proxy_control.py"] = sha(pair.ROOT / "tools/train_nested_proxy_control.py")
    hashes = {}
    for arm in ("control", "candidate"):
        folder = design / arm; folder.mkdir()
        shutil.copyfile(original / "roles.csv", folder / "roles.csv")
        table = pd.read_parquet(original / "proxy_labels.parquet")
        if arm == "candidate": table.loc[0, "ACL"] = .8
        table.to_parquet(folder / "proxy_labels.parquet", index=False)
        arm_spec = {**spec, "label_arm": arm, "sampling_policy": "legacy",
                    "proxy_labels_sha256": sha(folder / "proxy_labels.parquet")}
        (folder / "spec.json").write_text(json.dumps(arm_spec))
        hashes[arm] = sha(folder / "spec.json")
    manifest = {"arm_spec_sha256": hashes, "changes_sha256": sha(design / "changes.jsonl"),
                "label_receipt_sha256": sha(design / "label_receipt.json"),
                "comparison_tool_sha256": sha(Path(pair.__file__)), "scope": "synthetic_test",
                "adoption_gate": {"macro_auc_delta_at_least": .005, "per_finding_point_delta_at_least": -.02,
                                  "paired_component_bootstrap_lower_bound_above": 0}}
    (design / "pair.json").write_text(json.dumps(manifest))
    return design, inputs, roles


def test_both_label_arms_pass_existing_training_preflight(staged):
    design, inputs, _ = staged
    _, _, _, _, check = pair.load_pair(design)
    assert check["changed_slots"] == 1 and check["held_out_labels_unchanged"]
    for arm in ("control", "candidate"):
        assert run(design / arm, inputs, None)["training_launched"] is False


@pytest.mark.parametrize('kind', ['heldout', 'weight', 'unlogged', 'ledger'])
def test_label_only_guards(staged, kind):
    design, _, roles = staged
    a = pd.read_parquet(design / "control/proxy_labels.parquet")
    b = pd.read_parquet(design / "candidate/proxy_labels.parquet")
    changes = [json.loads((design / "changes.jsonl").read_text())]
    if kind == 'heldout': b.loc[8, "ACL"] = .9
    if kind == 'weight': b.loc[0, "ACL_weight"] = 1
    if kind == 'unlogged': b.loc[1, "PCL"] = .1
    if kind == 'ledger': changes[0]["new_target"] = .9
    with pytest.raises(ValueError): pair.validate_changes(a, b, roles, changes)


def test_comparison_uses_each_training_hash_and_common_heldout_labels(staged, tmp_path):
    design, inputs, roles = staged
    dirs = []
    for arm in ("control", "candidate"):
        d = tmp_path / (arm + "_run"); d.mkdir(); dirs.append(d)
        spec = json.loads((design / arm / "spec.json").read_text())
        (d / "selected.pt").write_bytes(b'synthetic checkpoint fixture')
        pred = pd.DataFrame({ID: roles.loc[roles.role == 'outer', ID], **{f: [.2, .8, .2, .8] for f in LABELS}})
        pred.to_parquet(d / "outer_predictions.parquet", index=False)
        contract = {**spec, "training_recipe": {"same": True}, "initial_weights_sha256": "same",
                    "runtime_versions": {"test": "synthetic"}, "input_config_sha256": sha(inputs),
                    "checkpoint_sha256": sha(d / "selected.pt"),
                    "outer_predictions_sha256": sha(d / "outer_predictions.parquet"),
                    "training_script_sha256": spec["runtime_code_sha256"]["tools/train_nested_proxy_control.py"]}
        for role in ("train", "selection", "outer"):
            contract[role + "_uids"] = roles.loc[roles.role == role, ID].tolist()
        (d / "run_contract.json").write_text(json.dumps(contract))
    result = pair.evaluate(design, *dirs, tmp_path / "result.json", draws=100)
    assert result["macro_delta"] == 0
    assert not result["label_ablation_screen_passed"] and not result["image_gain_demonstrated"]
    path = dirs[1] / "run_contract.json"
    bad = json.loads(path.read_text()); bad['initial_weights_sha256'] = 'different'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='initial_weights'):
        pair.evaluate(design, *dirs, tmp_path / "rejected.json", draws=100)
