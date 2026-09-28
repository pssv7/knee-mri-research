"""Colab fold-2 uses existing inputs and preserves both embedded arm paths."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.colab_slice_runtime import copy_verified, relocated_pair, stage_drive_inputs, mount_inputs, choose_metadata
from tools.prepare_colab_slice_pair import build


def test_drive_copy_verifies_full_content_and_atomically_replaces(tmp_path):
    source, target = tmp_path / "drive.bin", tmp_path / "local/input.bin"
    source.write_bytes(bytes(range(256)) * 40000)
    digest = copy_verified(source, target)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.read_bytes() == target.read_bytes()
    assert not target.with_name(target.name + ".partial").exists()


def test_completed_copy_is_reused_but_same_size_corruption_is_repaired(tmp_path):
    source, target = tmp_path / "drive", tmp_path / "local"
    source.write_bytes(b"correct")
    target.write_bytes(b"correct")
    previous = target.stat().st_mtime_ns
    assert copy_verified(source, target) == hashlib.sha256(b"correct").hexdigest()
    assert target.stat().st_mtime_ns == previous
    target.write_bytes(b"damaged")
    copy_verified(source, target)
    assert target.read_bytes() == b"correct"


def test_metadata_duplicates_are_resolved_by_content_not_path_order(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert choose_metadata([b, a], "train.csv") == a
    b.write_bytes(b"different")
    assert choose_metadata([a, b], "train.csv") is None
    assert choose_metadata([a, b], "train_labels.parquet",
                           hashlib.sha256(b"different").hexdigest()) == b
    assert choose_metadata([a, b], "train_labels.parquet", "unmatched") is None
    assert choose_metadata([], "train.csv") is None


def test_relocation_rewrites_outer_and_both_nested_training_paths():
    source = '''TRAINING_SCRIPTS = {"baseline": 'OUTPUT = "/kaggle/working/baseline_fold2"',
    "slice_transformer": 'CONTROL = "/kaggle/working/baseline_fold2/run_contract.json"'}
SMOKE_SOURCE = 'OUTPUT = "/kaggle/working/ddp_smoke"'
OUTPUT = "/kaggle/working"
INPUT = "/kaggle/input"
'''
    moved = relocated_pair(source, "/drive/attempt")
    values = {}
    exec(moved, values)
    assert values["OUTPUT"] == "/drive/attempt"
    assert values["INPUT"] == "/content/knee_slice_mounts"
    assert "/drive/attempt/baseline_fold2/run_contract.json" in values["TRAINING_SCRIPTS"]["slice_transformer"]
    assert "/kaggle/working" not in moved


def test_incomplete_drive_cache_stops_before_any_download(tmp_path):
    with pytest.raises(RuntimeError, match="Expected 3"):
        stage_drive_inputs(tmp_path, tmp_path / "local", "labels-hash")


def test_csv_mount_does_not_hide_cache_from_training_finder(tmp_path, monkeypatch):
    import glob
    import os
    from tools.prepare_kaggle_training_pair import training_source
    local, inputs = tmp_path / "local", tmp_path / "input"
    for name in ("cache", "metadata"):
        (local / name).mkdir(parents=True)
    names = [f"cache_train_series_{i:02}.npy" for i in range(3)]
    names += [f"manifest_train_series_{i:02}.parquet" for i in range(3)]
    names += ["train_labels.parquet", "side_predictions.parquet"]
    for name in names:
        (local / "cache" / name).touch()
    for name in ("train.csv", "train_series.csv"):
        (local / "metadata" / name).touch()
    mount_inputs(local, inputs)
    # Remote training starts in an empty attempt directory, not this repo's
    # cwd containing a second copy of data/meta/train_labels.parquet.
    monkeypatch.chdir(tmp_path)
    notebook = json.loads((ROOT / "notebooks/08_finetune.ipynb").read_text())
    from tools.colab_slice_runtime import relocate_inputs
    training = training_source(notebook, "slice_transformer", "hash", fold=2)
    tree = ast.parse(relocate_inputs(training, inputs))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("find_inputs", "find_competition_file")]
    source = ast.unparse(ast.Module(body=funcs, type_ignores=[]))
    scope = {"glob": glob, "os": os}
    exec(source, scope)
    assert len(scope["find_inputs"]("cache_train_series_*.npy")) == 3
    assert len(scope["find_inputs"]("train_labels.parquet")) == 1
    assert Path(scope["find_competition_file"]("train.csv")).name == "train.csv"


def test_outer_smoke_and_training_input_paths_all_relocated():
    training = 'INPUT = "/kaggle/input"\nPATTERN = "/kaggle/input/*/train.csv"'
    source = (f'TRAINING_SCRIPTS = {dict(baseline=training, slice_transformer=training)!r}\n'
              f'SMOKE_SOURCE = {training!r}\nINPUT = "/kaggle/input"')
    values = {}
    exec(relocated_pair(source, "/drive/attempt"), values)
    assert values["INPUT"] == "/content/knee_slice_mounts"
    for script in [*values["TRAINING_SCRIPTS"].values(), values["SMOKE_SOURCE"]]:
        assert "/kaggle/input" not in script
        assert "/content/knee_slice_mounts/*/train.csv" in script


def test_colab_notebook_contains_exact_fold2_recipe_and_no_parent_limits(tmp_path):
    for path in ("data", "tools", "outputs/kaggle_sources"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / path, target_is_directory=True)
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "08_finetune.ipynb").write_bytes((ROOT / "notebooks/08_finetune.ipynb").read_bytes())
    notebook = json.loads(build(tmp_path).read_text())
    sources = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    setup = next(source for source in sources if source.startswith("%pip"))
    assert '"pydicom>=3.0"' in setup.splitlines()[0]
    assert '"pydicom"' in setup and "importlib.import_module(package)" in setup
    for source in sources:
        ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
    embedded = {}
    exec(next(source for source in sources if source.startswith("PAIR_SOURCE =")), embedded)
    tree = ast.parse(embedded["PAIR_SOURCE"])
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert values["FOLD"] == 2 and values["VALIDATION_STUDIES"] == 870
    assert values["SMOKE_TIMEOUT"] is None and values["RUN_TIMEOUT"] is None
    assert set(values["TRAINING_SCRIPTS"]) == {"baseline", "slice_transformer"}
    for source in values["TRAINING_SCRIPTS"].values():
        assert "FOLD = 2" in source
        assert "RUN = replace(RUN, series_selection='protocol')" in source
    # Self-contained runtime, without requiring the repo on Colab.
    runtime = next(source for source in sources if "def relocated_pair" in source)
    namespace = {}
    exec(runtime, namespace)
    assert callable(namespace["run_pair_on_drive"])
    assert "from tools." not in runtime
