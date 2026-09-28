"""Static staging and guard checks; never contact Kaggle."""

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_kaggle_submission import ROOT, prepare


def test_submission_keeps_fixed_members_and_drops_cache_inputs(tmp_path):
    for name in ["notebooks/10_submit_ensemble.ipynb",
                 "outputs/kaggle_benchmark_results/benchmark_report.json",
                 "outputs/kaggle_benchmark_receipt/kernel-metadata.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("benchmark_report.json"):
            path.write_text(json.dumps({"raw_complete": True, "oof_complete": True, "errors": []}))
        elif name.endswith("kernel-metadata.json"):
            path.write_text(json.dumps({"docker_image": "image", "dataset_sources": ["ensemble", "labels"],
                                       "competition_sources": ["competition"]}))
        else:
            shutil.copy2(ROOT / name, path)
    upload = tmp_path / "outputs/kaggle_ensemble_upload"
    upload.mkdir()
    entries = []
    for slices in [16, 24]:
        for fold in range(5):
            filename = f"slices_{slices}_fold{fold}.pt"
            entries.append({"checkpoint": filename, "weight": .1})
            (upload / (filename + ".json")).write_text(json.dumps({"experiment": {"slices": slices},
                "fold": fold, "sha256": f"fixture-{slices}-{fold}"}))
    (upload / "ensemble.json").write_text(json.dumps({"members": entries}))
    prepare(tmp_path)
    out = tmp_path / "outputs/kaggle_submission"
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["is_private"] and not meta["enable_internet"]
    assert meta["kernel_sources"] == ["pssv27/knee-side-classifier"]
    notebook = json.loads((out / "submission.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    assert "EXPECTED_CHECKPOINTS" in code[0]
    assert "submission_report.json" in code[-1]
    assert all("!pip" not in source and "competition_submit(" not in source for source in code)
    entries[0]["weight"] = .2
    (upload / "ensemble.json").write_text(json.dumps({"members": entries}))
    with pytest.raises(ValueError, match="equal"):
        prepare(tmp_path)
