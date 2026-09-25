"""The Kaggle probe kernel assembles exactly the Colab notebook's bundle."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from tools.prepare_colab_perturbation import ROOT, payload
from tools.prepare_kaggle_perturbation import BODY, EMBED, SOURCES, main, script

RESULTS = ROOT / 'outputs/kaggle_distinct_training_pair_v1_results'


def kernel_scope(root, working, monkeypatch):
    """The kernel's constants and definitions, without pip or the run itself."""
    monkeypatch.setenv('PROBE_INPUT', str(root))
    monkeypatch.setenv('PROBE_WORKING', str(working))
    source, _ = script()
    header, body = source.split('\nimport subprocess, sys\n', 1)
    definitions = body.split('\nimport base64, hashlib', 1)[1].split('\ntry:\n', 1)[0]
    scope = {}
    exec(header + '\nimport base64, hashlib' + definitions, scope)
    return scope


def test_kernel_fits_kaggle_and_carries_the_colab_manifest(tmp_path):
    out = main(destination=tmp_path / 'kernel')
    source = (out / 'probe_kernel.py').read_text()
    assert len(source.encode()) < 1_000_000  # Kaggle rejects larger kernel source
    scope = {}
    exec(source.split('\nimport subprocess, sys\n', 1)[0], scope)
    archive, _, _ = payload(include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        for name in EMBED:
            assert base64.b64decode(scope['EMBEDDED'][name]) == bundle.read(name)
    assert scope['MANIFEST'] == manifest and 'metadata/train_series.csv' not in manifest
    assert scope['BUNDLE_SHA256'] == script()[1]
    metadata = json.loads((out / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] and metadata['enable_gpu'] and metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == SOURCES
    # The checkpoint is chosen by the control contract's hash, never by path.
    assert "sha(p) == reference['checkpoint_sha256']" in source
    assert "phase('smoke', checkpoints[0], inputs_path, limit=5)" in source


def test_assembly_from_the_real_control_output_reproduces_the_bundle(tmp_path, monkeypatch):
    # Lay the downloaded control output out as Kaggle mounts it (small files only).
    root = tmp_path / 'input'
    mount = root / 'notebooks/pssv27/knee-distinct-sampling-pair-v1'
    for path in RESULTS.rglob('*'):
        if path.is_file() and path.stat().st_size < 5 * 10**7:
            target = mount / path.relative_to(RESULTS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    scope = kernel_scope(root, tmp_path / 'working', monkeypatch)
    scope['assemble']()
    code = Path(scope['code'])
    archive, _, _ = payload(include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in json.loads(bundle.read('bundle_manifest.json')):
            assert (code / name).read_bytes() == bundle.read(name), name
    # A tampered attached file is refused rather than used.
    shutil.rmtree(code)
    (mount / 'control/run_contract.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='reference/run_contract.json'):
        scope['assemble']()


def test_input_search_skips_the_dicom_tree_and_checks_small_file_hashes(tmp_path, monkeypatch):
    root = tmp_path / 'input'
    competition = root / 'competitions/rsna-knee-abnormality-detection'
    (competition / 'train_images/deep').mkdir(parents=True)
    (competition / 'train_series.csv').write_bytes(b'series')
    (competition / 'train_images/deep/selected.pt').write_bytes(b'decoy')
    control = root / 'notebooks/pssv27/knee-distinct-sampling-pair-v1/control'
    candidate = control.parent / 'candidate'
    control.mkdir(parents=True)
    candidate.mkdir()
    (control / 'selected.pt').write_bytes(b'tiny')
    (candidate / 'selected.pt').write_bytes(b'other')
    scope = kernel_scope(root, tmp_path / 'working', monkeypatch)
    assert scope['attached']('selected.pt') == sorted([control / 'selected.pt', candidate / 'selected.pt'])
    entry = {'path': '/elsewhere/train_series.csv', 'bytes': 6,
             'sha256': hashlib.sha256(b'series').hexdigest()}
    assert scope['locate'](entry, True) == str(competition / 'train_series.csv')
    with pytest.raises(FileNotFoundError):
        scope['locate'](dict(entry, sha256='0' * 64), True)
