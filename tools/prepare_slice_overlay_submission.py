"""Stage fixed 18-model overlay inference."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import artifacts
from tools.prepare_protocol_benchmark import INPUT_SEARCH

OVERLAY = ROOT / "outputs/kaggle_slice_overlay_upload"
FULL = ROOT / "outputs/ensemble_16_24_protocol_slice_overlay"
OUT = ROOT / "outputs/kaggle_slice_overlay_submission"


def main():
    manifest = (OVERLAY / "overlay.json").read_bytes()
    entries = json.loads(manifest)["members"]
    assert len(entries) == 18 and all(e["weight"] == 1 / 18 for e in entries)
    expected = {}
    for entry in json.loads((FULL / "ensemble.json").read_text())["members"]:
        meta = artifacts.load_metadata(FULL / entry["checkpoint"])
        key = f"{meta['experiment']['name']}:{meta['fold']}"
        assert key not in expected
        expected[key] = meta["sha256"]
    assert len(expected) == 18
    notebook = json.loads((ROOT / "notebooks/10_submit_ensemble.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(sources) == 5
    # Replace the embedded source BEFORE bootstrap imports any model modules.
    current_net = (ROOT / "src/model/net.py").read_text()
    bootstrap = ast.parse(sources[0])
    assignments = [n for n in bootstrap.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets)]
    assert len(assignments) == 1
    files = ast.literal_eval(assignments[0].value)
    files['src/model/net.py'] = current_net
    assignments[0].value = ast.parse(repr(files), mode='eval').body
    sources[0] = ast.unparse(ast.fix_missing_locations(bootstrap))
    sources[0] += "\nassert artifacts.net is net\nassert net.aggregation_defaults({'slice_encoder.project.weight': None})['slice_transformer'] is True\n"
    tree = ast.parse("\n".join(line for line in sources[1].splitlines() if not line.startswith("!")))
    positions = [i for i, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef) and node.name == "find_inputs"]
    assert len(positions) == 1
    tree.body[positions[0]] = ast.parse("def find_inputs(pattern):\n    return [str(p) for p in attached_files(pattern)]").body[0]
    sources[1] = ast.unparse(tree) + "\nassert device == 'cuda'\ntorch.set_num_threads(2)\n"
    header = "import hashlib, json, os, pathlib, time\nSUBMISSION_STARTED = time.monotonic()\n" + INPUT_SEARCH
    header += "\nmanifests = attached_files('overlay.json')\nassert len(manifests) == 1\n"
    header += f"assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == {hashlib.sha256(manifest).hexdigest()!r}\n"
    header += f"os.environ['KNEE_ENSEMBLE_JSON'] = str(manifests[0])\nEXPECTED_CHECKPOINTS = {expected!r}\n"
    sources[2] += "\nassert len(explicit_members) == len(tuned_models) == 18\nassert BLEND == 1.0 and not FLIP_TTA and not SOUP and not MEMBER_RANK_AVERAGE\nassert np.array_equal(np.asarray(member_weights), np.full(18, 1 / 18))\nactual_checkpoints = {f\"{m['spec']['name']}:{m['metadata']['fold']}\": m['metadata']['sha256'] for m in explicit_members}\nassert actual_checkpoints == EXPECTED_CHECKPOINTS\n"
    validate = "\nassert not FAST_DECODE\nassert submission[config.ID_COL].tolist() == test_csv[config.ID_COL].tolist()\nassert submission.columns.tolist() == config.SUBMISSION_COLUMNS\nassert np.isfinite(submission[config.LABELS].to_numpy()).all()\nassert ((submission[config.LABELS] >= 0) & (submission[config.LABELS] <= 1)).all().all()\nreport = {'rows': len(submission), 'members': len(explicit_members), 'fallbacks': failures, 'checkpoints': actual_checkpoints, 'weights': list(map(float, member_weights)), 'elapsed_seconds': time.monotonic() - SUBMISSION_STARTED, 'gpu': torch.cuda.get_device_name(), 'columns': submission.columns.tolist(), 'submission_sha256': hashlib.sha256(pathlib.Path('/kaggle/working/submission.csv').read_bytes()).hexdigest()}\npathlib.Path('/kaggle/working/submission_report.json').write_text(json.dumps(report, indent=2) + '\\n')\nprint(json.dumps(report, indent=2))\n"
    code = [header, *sources, validate]
    for source in code:
        ast.parse(source)
    notebook["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": ["# Fixed 18-member protocol + slice overlay\n"]}]
    notebook["cells"] += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)} for source in code]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = json.loads((ROOT / "outputs/kaggle_protocol_submission/kernel-metadata.json").read_text())
    metadata.update(id="pssv27/knee-16-24-protocol-slice-overlay", title="Knee 16 24 Protocol Slice Overlay", code_file="submission.ipynb", dataset_sources=["pssv27/knee-ensemble-protocol-private", "pssv27/knee-slice-transformer-overlay-private", "pssv27/knee-labels"])
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Staged {OUT}; no remote run or submission")


if __name__ == "__main__":
    main()
