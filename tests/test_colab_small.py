"""Exercise the actual embedded Small notebook, staging and failure boundaries."""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.validation.proxy import ID, LABELS, sha
from tools.prepare_colab_small import ROOT, build, payload, small_source
from tools.colab_small_runtime import copy_pinned, stage_inputs, verify_completed, completion_files, compare_candidate


def entry(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


@pytest.mark.parametrize('custom', [False, True])
def test_staging_cell_restores_missing_config_and_preserves_custom_paths(custom):
    import sys
    notebook = json.loads((ROOT / 'notebooks/22_convnext_small_224_fold0_colab.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if c['cell_type'] == 'code' and 'INPUTS = stage_inputs(' in ''.join(c['source']))
    calls = []
    scope = {'CODE_ROOT': Path('/content/code'), 'pathlib': __import__('pathlib'), 'sys': sys,
             'stage_inputs': lambda *args: calls.append(args) or Path('/content/inputs.json'),
             'run_process': lambda *args: None}
    if custom:
        scope.update(DRIVE_CACHE='/custom/cache', SIDE_FILE='/custom/side',
                     DRIVE_OUT='/custom/output', LOCAL_INPUTS='/custom/local')
    exec(source, scope)
    assert calls[0][0] == ('/custom/cache' if custom else '/content/drive/MyDrive/kagglehub')
    assert calls[0][1] == ('/custom/local' if custom else '/content/knee_small_inputs')
    assert calls[0][4] == ('/custom/side' if custom else '')
    assert scope['DRIVE_OUT'] == ('/custom/output' if custom else
                                  '/content/drive/MyDrive/knee_runs/convnext_small_224_outer0_v1')
    with pytest.raises(RuntimeError, match='Run section 2'):
        exec(source, {})


def test_copy_rejects_wrong_same_size_bytes_and_repairs_local_copy(tmp_path):
    good, bad, local = [tmp_path / n for n in ['good', 'bad', 'local']]
    good.write_bytes(b'correct')
    bad.write_bytes(b'corrupt')
    local.write_bytes(b'corrupt')
    copy_pinned([bad, good], local, entry(good))
    assert local.read_bytes() == b'correct'
    stamp = local.stat().st_mtime_ns
    copy_pinned([], local, entry(good))
    assert local.stat().st_mtime_ns == stamp
    with pytest.raises(RuntimeError, match='No hash-matching'):
        copy_pinned([bad], tmp_path / 'missing', entry(good))


def test_staging_uses_only_pinned_files_and_reuses_local_copies(tmp_path):
    drive, local, code = [tmp_path / n for n in ['drive', 'local', 'code']]
    drive.mkdir()
    (code / 'metadata').mkdir(parents=True)
    entries = []
    for name in ['train_series.csv', 'side_predictions.parquet', 'array.npy', 'manifest.parquet']:
        path = drive / name
        path.write_bytes(name.encode())
        entries.append(entry(path))
    expected = dict(series_meta=entries[0], side_predictions=entries[1],
                    shards=[dict(array=entries[2], manifest=entries[3])])
    first = stage_inputs(drive, local, code, expected)
    bound = json.loads(first.read_text())
    assert all(Path(e['path']).parent == local for e in bound['shards'][0].values())
    for path in drive.iterdir():
        path.unlink()
    assert stage_inputs(drive, local, code, expected) == first


def test_missing_shards_fail_before_credential_or_network_access(tmp_path):
    drive = tmp_path / 'drive'
    drive.mkdir()
    expected = {k: {'path': k, 'bytes': 20, 'sha256': 'missing'} for k in ['series_meta', 'side_predictions']}
    expected['shards'] = [{'array': {'path': 'array.npy', 'bytes': 20, 'sha256': 'missing'}}]
    with pytest.raises(RuntimeError, match='Missing original'):
        stage_inputs(drive, tmp_path / 'local', tmp_path / 'code', expected)


def test_notebook_is_self_contained_and_generated_deterministically(tmp_path):
    path = build(destination=tmp_path / 'small.ipynb')
    import nbformat
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')))
    assert path.read_bytes() == (ROOT / 'notebooks/22_convnext_small_224_fold0_colab.ipynb').read_bytes()
    archive, _ = payload()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        for name, digest in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
            if name.endswith('.py'):
                ast.parse(bundle.read(name).decode())
        spec = json.loads(bundle.read('design/candidate/spec.json'))
        assert spec['backbone'] == 'convnext_small.fb_in22k_ft_in1k'
        assert (spec['cache_size'], spec['slices'], spec['max_series'], spec['epochs']) == (224, 16, 5, 12)
        assert spec['sampling_policy'] == 'legacy' and spec['gold_monitoring'] is False
        roles = pd.read_csv(io.BytesIO(bundle.read('design/candidate/roles.csv')))
        assert roles.role.value_counts().to_dict() == {'train': 2208, 'outer': 691, 'selection': 553}
        for name in ['roles.csv', 'proxy_labels.parquet']:
            assert bundle.read('design/candidate/' + name) == (ROOT / 'outputs/distinct_sampling_pair_v1/control' / name).read_bytes()
        assert not any(name.endswith(('.pt', '.ckpt', '.npy')) for name in bundle.namelist())


def test_generated_small_preflight_accepts_small_and_rejects_tiny(tmp_path):
    # Run the actual embedded preflight over real miniature files, without a GPU.
    design = tmp_path / 'candidate'
    design.mkdir()
    ids = [f's{i}' for i in range(6)]
    roles = pd.DataFrame({ID: ids, 'component': ids, 'role': ['train']*2+['selection']*2+['outer']*2})
    roles.to_csv(design / 'roles.csv', index=False)
    labels = pd.DataFrame({ID: ids})
    for label in LABELS:
        labels[label] = [.1, .9]*3
        labels[label + '_weight'] = .7
    labels.to_parquet(design / 'proxy_labels.parquet')
    pd.DataFrame({ID: ['protected'], 'component': ['protected']}).to_csv(tmp_path / 'exclusions.csv', index=False)
    spec = json.loads((ROOT / 'outputs/distinct_sampling_pair_v1/control/spec.json').read_text())
    spec.update(backbone='convnext_small.fb_in22k_ft_in1k', roles_sha256=sha(design / 'roles.csv'),
                proxy_labels_sha256=sha(design / 'proxy_labels.parquet'),
                protected_exclusions_sha256=sha(tmp_path / 'exclusions.csv'),
                runtime_code_sha256={'src/validation/proxy.py': sha(ROOT / 'src/validation/proxy.py')})
    spec.pop('raw_counts_sha256')
    (design / 'spec.json').write_text(json.dumps(spec))
    series = pd.DataFrame({ID: ids, 'SeriesInstanceUID': [f'ser{i}' for i in range(6)],
                           'Fluid_Sensitive': 1, 'Fat_Suppression': 1, 'Anatomical_Plane': 'Sagittal'})
    series.to_csv(tmp_path / 'series.csv', index=False)
    pd.DataFrame({'SeriesInstanceUID': series.SeriesInstanceUID, 'side_predicted': 'L'}).to_parquet(tmp_path / 'sides.parquet')
    np.save(tmp_path / 'cache.npy', np.ones((6, 24, 224, 224), dtype=np.uint8))
    manifest = series[[ID, 'SeriesInstanceUID']].copy()
    manifest['position'] = np.arange(6)
    manifest.to_parquet(tmp_path / 'manifest.parquet')
    inputs = {'series_meta': entry(tmp_path / 'series.csv'), 'side_predictions': entry(tmp_path / 'sides.parquet'),
              'shards': [{'array': entry(tmp_path / 'cache.npy'), 'manifest': entry(tmp_path / 'manifest.parquet')}]}
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps(inputs))
    scope = {'__file__': str(ROOT / 'tools/train_small.py'), '__name__': 'small_test'}
    exec(small_source((ROOT / 'tools/train_nested_proxy_control.py').read_text()), scope)
    result = scope['run'](design, path, None)
    assert result['status'] == 'preflight_passed' and result['training_launched'] is False
    spec['backbone'] = 'convnext_tiny.fb_in22k_ft_in1k'
    (design / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='recipe differs'):
        scope['run'](design, path, None)


def test_completed_run_requires_all_artifact_hashes(tmp_path):
    files = {}
    for name in completion_files():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = sha(path)
    (tmp_path / 'complete.json').write_text(json.dumps({'bundle_sha256': 'frozen', 'files': files}))
    assert verify_completed(tmp_path, 'frozen')
    (tmp_path / 'candidate/selected.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_completed(tmp_path, 'frozen')


def test_comparison_labels_historical_control_and_rejects_wrong_recipe(tmp_path):
    archive, _ = payload()
    code, output = tmp_path / 'code', tmp_path / 'out'
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        bundle.extractall(code)
    candidate = output / 'candidate'
    candidate.mkdir(parents=True)
    spec = json.loads((code / 'design/candidate/spec.json').read_text())
    control = json.loads((code / 'reference/run_contract.json').read_text())
    contract = dict(control)
    contract.update(spec)
    contract['training_recipe'] = dict(control['training_recipe'], backbone=spec['backbone'])
    contract['arm'] = 'candidate'
    # Dummy checkpoint bytes suffice here: exercise provenance and comparison,
    # while the notebook's real smoke checks model serialization separately.
    (candidate / 'selected.pt').write_bytes(b'candidate weights')
    contract['checkpoint_sha256'] = sha(candidate / 'selected.pt')
    (candidate / 'outer_predictions.parquet').write_bytes((code / 'reference/outer_predictions.parquet').read_bytes())
    (candidate / 'run_contract.json').write_text(json.dumps(contract))
    (output / 'smoke').mkdir()
    (output / 'smoke/smoke_receipt.json').write_text(json.dumps({
        'status': 'passed', 'initial_weights_sha256': contract['initial_weights_sha256'],
        'runtime_versions': contract['runtime_versions']}))
    metrics = candidate / 'training/version_0/metrics.csv'
    metrics.parent.mkdir(parents=True)
    pd.DataFrame({'epoch': range(12)}).to_csv(metrics, index=False)
    report, decision = compare_candidate(code, code / 'reference/inputs.json', output)
    assert report['macro_delta'] == 0
    assert report['matched_hardware_runtime'] is False
    assert not decision['promotion_allowed'] and not decision['automatic_followup_training']
    contract['training_recipe']['slices'] = 24
    (candidate / 'run_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='beyond backbone'):
        compare_candidate(code, code / 'reference/inputs.json', output)


def test_smoke_executes_two_steps_and_roundtrip_on_cpu_surrogate(tmp_path, monkeypatch):
    # Exercise smoke orchestration with a tiny CNN and CPU. This does NOT
    # certify CUDA memory/precision; the actual notebook performs that gate.
    import contextlib
    import torch
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset
    from src.model import net, lightning as knee
    from tools.prepare_colab_small import SMOKE
    class Trunk(torch.nn.Sequential):
        num_features = 8
        def __init__(self):
            super().__init__(torch.nn.Conv2d(3, 8, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    class Studies(Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return {ID: f's{i}', 'volumes': np.full((5, 16, 8, 8), 70+i, dtype=np.uint8),
                    'meta': np.zeros((5, 7), dtype=np.float32),
                    'targets': np.full(12, i, dtype=np.float32), 'weights': np.ones(12, dtype=np.float32)}
    trainer_class = pl.Trainer
    def cpu_trainer(**kwargs):
        kwargs.update(accelerator='cpu', precision='32-true', enable_progress_bar=False)
        return trainer_class(**kwargs)
    monkeypatch.setattr(pl, 'Trainer', cpu_trainer)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    original_to = torch.nn.Module.to
    monkeypatch.setattr(torch.nn.Module, 'to', lambda self, *args, **kw:
                        original_to(self, *('cpu' if a == 'cuda' else a for a in args), **kw))
    monkeypatch.setattr(torch, 'autocast', lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU surrogate')
    monkeypatch.setattr(net, 'build_backbone', lambda *args, **kwargs: Trunk())
    import time
    from src.validation.proxy import write_json_new
    namespace = {'time': time, 'write_json_new': write_json_new}
    exec(SMOKE, namespace)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        module = knee.KneeLitModule(Trunk(), 8, total_steps=3312, warmup_steps=994)
        result = namespace['smoke_model'](module, DataLoader(Studies()), tmp_path, 'init-hash')
    finally:
        torch.set_num_threads(old_threads)
    assert result['status'] == 'passed'
    assert result['optimizer_steps'] == 2 and result['microbatches'] == 16
    assert result['checkpoint_roundtrip_max_abs'] == 0
    assert not (tmp_path / 'transient_roundtrip.pt').exists()
