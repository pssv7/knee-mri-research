"""Exercise the perturbation notebook bundle, reference checks and runner loop."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.validation.proxy import ID, LABELS, sha
from tools.prepare_colab_perturbation import NOTEBOOK, PHASES, ROOT, build, payload
from tools import perturbation_probe

CONTROL = ROOT / 'outputs/kaggle_distinct_training_pair_v1_results/control'
FROZEN = ROOT / 'outputs/distinct_sampling_pair_v1'


def entry(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def test_notebook_is_self_contained_and_generated_deterministically(tmp_path):
    path = build(destination=tmp_path / 'probe.ipynb')
    import nbformat
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')))
    assert path.read_bytes() == (ROOT / NOTEBOOK).read_bytes()
    archive, _, checkpoint_hash = payload()
    assert checkpoint_hash == json.loads((CONTROL / 'run_contract.json').read_text())['checkpoint_sha256']
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        for name, digest in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
            if name.endswith('.py'):
                ast.parse(bundle.read(name).decode())
        # The frozen control design travels byte-identical, not re-serialised.
        for name in ['spec.json', 'roles.csv', 'proxy_labels.parquet']:
            assert bundle.read('design/control/' + name) == (FROZEN / 'control' / name).read_bytes()
        for name in ['run_contract.json', 'outer_predictions.parquet']:
            assert bundle.read('reference/' + name) == (CONTROL / name).read_bytes()
        assert bundle.read('src/model/perturb.py') == (ROOT / 'src/model/perturb.py').read_bytes()
        assert not any(name.endswith(('.pt', '.ckpt', '.npy')) for name in bundle.namelist())


def test_setup_cell_checks_imports_in_a_child_process():
    notebook = json.loads((ROOT / NOTEBOOK).read_text())
    setup = next(''.join(c['source']) for c in notebook['cells']
                 if c['cell_type'] == 'code' and '%pip install' in ''.join(c['source']))
    assert 'subprocess.run([sys.executable, "-c"' in setup and 'lightning.pytorch' in setup
    assert 'capture_output=True' in setup and 'check.returncode != 0' in setup


def test_reference_check_accepts_the_real_control_and_rejects_other_weights(tmp_path):
    spec = json.loads((FROZEN / 'control/spec.json').read_text())
    roles = pd.read_csv(FROZEN / 'control/roles.csv')
    recipe = json.loads((CONTROL / 'run_contract.json').read_text())['training_recipe']
    contract, outer = perturbation_probe.check_reference(spec, roles, recipe, CONTROL / 'selected.pt', CONTROL)
    assert contract['arm'] == 'control' and len(outer) == 691
    wrong = tmp_path / 'selected.pt'
    wrong.write_bytes(b'other weights')
    with pytest.raises(ValueError, match='Checkpoint differs'):
        perturbation_probe.check_reference(spec, roles, recipe, wrong, CONTROL)
    with pytest.raises(ValueError, match='recipe differs'):
        perturbation_probe.check_reference(spec, roles, dict(recipe, slices=24), CONTROL / 'selected.pt', CONTROL)


@pytest.mark.parametrize('state', ['complete', 'partial', 'fresh'])
def test_phase_reuses_complete_output_and_stops_on_partial(tmp_path, state):
    import pathlib
    import sys
    out = tmp_path / 'probe'
    calls = []
    def fake_process(command, cwd, log):
        calls.append(command)
        (out / 'smoke').mkdir(parents=True)
        (out / 'smoke/run_contract.json').write_text(json.dumps({'reproduction': {'passed': True}}))
    if state == 'complete':
        (out / 'smoke').mkdir(parents=True)
        (out / 'smoke/run_contract.json').write_text(json.dumps({'reproduction': {'passed': True}}))
    elif state == 'partial':
        (out / 'smoke').mkdir(parents=True)
    scope = {'json': json, 'pathlib': pathlib, 'sys': sys, 'CODE_ROOT': tmp_path, 'INPUTS': 'inputs.json',
             'CHECKPOINT': 'selected.pt', 'DRIVE_OUT': str(out), 'run_process': fake_process}
    exec(PHASES, scope)
    if state == 'partial':
        with pytest.raises(RuntimeError, match='partial output'):
            scope['run_phase']('smoke', limit=5)
        return
    assert scope['run_phase']('smoke', limit=5)['reproduction']['passed']
    assert len(calls) == (0 if state == 'complete' else 1)
    if calls:
        assert calls[0][-5:-3] == ['--limit', '5'] and '--execute' in calls[0]


class Trunk:
    """Factory for a tiny CPU trunk standing in for ConvNeXt."""
    def __new__(cls):
        import torch
        class _Trunk(torch.nn.Sequential):
            num_features = 8
        return _Trunk(torch.nn.Conv2d(3, 8, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())


def miniature(tmp_path):
    """Real miniature design/inputs files that pass the frozen preflight."""
    design = tmp_path / 'control'
    design.mkdir()
    ids = [f's{i}' for i in range(6)]
    roles = pd.DataFrame({ID: ids, 'component': ids, 'role': ['train']*2 + ['selection']*2 + ['outer']*2})
    roles.to_csv(design / 'roles.csv', index=False)
    labels = pd.DataFrame({ID: ids})
    for label in LABELS:
        labels[label] = [.1, .9]*3
        labels[label + '_weight'] = .7
    labels.to_parquet(design / 'proxy_labels.parquet')
    pd.DataFrame({ID: ['protected'], 'component': ['protected']}).to_csv(tmp_path / 'exclusions.csv', index=False)
    spec = json.loads((FROZEN / 'control/spec.json').read_text())
    spec.update(roles_sha256=sha(design / 'roles.csv'), proxy_labels_sha256=sha(design / 'proxy_labels.parquet'),
                protected_exclusions_sha256=sha(tmp_path / 'exclusions.csv'),
                runtime_code_sha256={'src/validation/proxy.py': sha(ROOT / 'src/validation/proxy.py')})
    spec.pop('raw_counts_sha256')
    (design / 'spec.json').write_text(json.dumps(spec))
    series = pd.DataFrame({ID: ids, 'SeriesInstanceUID': [f'ser{i}' for i in range(6)],
                           'Fluid_Sensitive': 1, 'Fat_Suppression': 1,
                           'Anatomical_Plane': ['Sagittal', 'Coronal'] * 3})
    series.to_csv(tmp_path / 'series.csv', index=False)
    pd.DataFrame({'SeriesInstanceUID': series.SeriesInstanceUID, 'side_predicted': ['L', 'R'] * 3}
                 ).to_parquet(tmp_path / 'sides.parquet')
    rng = np.random.default_rng(0)
    np.save(tmp_path / 'cache.npy', rng.integers(0, 256, (6, 24, 224, 224), dtype=np.uint8))
    manifest = series[[ID, 'SeriesInstanceUID']].copy()
    manifest['position'] = np.arange(6)
    manifest.to_parquet(tmp_path / 'manifest.parquet')
    inputs = {'series_meta': entry(tmp_path / 'series.csv'), 'side_predictions': entry(tmp_path / 'sides.parquet'),
              'shards': [{'array': entry(tmp_path / 'cache.npy'), 'manifest': entry(tmp_path / 'manifest.parquet')}]}
    (tmp_path / 'inputs.json').write_text(json.dumps(inputs))
    return design, tmp_path / 'inputs.json', spec, roles


def surrogate_gpu(monkeypatch):
    """Run the CUDA code path on CPU. Certifies orchestration, not precision."""
    import torch
    from src.model import net
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'CPU surrogate')
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    original_to = torch.nn.Module.to
    monkeypatch.setattr(torch.nn.Module, 'to', lambda self, *args, **kw:
                        original_to(self, *('cpu' if a == 'cuda' else a for a in args), **kw))
    monkeypatch.setattr(torch, 'autocast', lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(net, 'build_backbone', lambda *args, **kwargs: Trunk())


def reference_run(tmp_path, design, inputs, spec, roles):
    """Save surrogate weights and their own identity predictions as the reference."""
    import torch
    from src.model import dataset, lightning as knee, net
    from tools.train_nested_proxy_control import preflight
    torch.manual_seed(0)
    model = net.KneeNet(Trunk(), 8).eval()
    checkpoint = tmp_path / 'selected.pt'
    torch.save(model.state_dict(), checkpoint)
    _, _, labels, manifest, series, sides, paths, recipe = preflight(design, inputs)
    index = dataset.StudyIndex(manifest, series, labels, sides, max_series=5, series_selection='protocol')
    outer = sorted(roles.loc[roles.role == 'outer', ID])
    data = knee.StudyVolumeDataset(index, outer, paths, 16, augment=False, resample=False)
    rows = []
    with torch.inference_mode():
        for i, uid in enumerate(outer):
            batch = dataset.collate_studies([data[i]])
            p = model(batch['volumes'], batch['meta'], batch['mask']).sigmoid().numpy()[0]
            rows.append({ID: uid, **dict(zip(LABELS, map(float, p)))})
    reference = tmp_path / 'reference'
    reference.mkdir()
    pd.DataFrame(rows).to_parquet(reference / 'outer_predictions.parquet', index=False)
    contract = {'checkpoint_sha256': sha(checkpoint), 'arm': 'control',
                'outer_predictions_sha256': sha(reference / 'outer_predictions.parquet'),
                'training_recipe': recipe, 'outer_uids': outer, 'runtime_versions': {'gpu': 'CPU'},
                **{k: spec[k] for k in ['roles_sha256', 'proxy_labels_sha256', 'seed', 'sampling_policy']}}
    (reference / 'run_contract.json').write_text(json.dumps(contract))
    return checkpoint, reference


def test_runner_scores_every_variant_and_reproduces_on_cpu_surrogate(tmp_path, monkeypatch):
    design, inputs, spec, roles = miniature(tmp_path)
    checkpoint, reference = reference_run(tmp_path, design, inputs, spec, roles)
    preflight = perturbation_probe.run(design, inputs, checkpoint, reference, None)
    assert preflight['status'] == 'preflight_passed' and preflight['probe_launched'] is False
    surrogate_gpu(monkeypatch)
    out = tmp_path / 'probe'
    result = perturbation_probe.run(design, inputs, checkpoint, reference, out, execute=True)
    assert result['status'] == 'probe_complete' and result['reproduction']['passed']
    assert result['reproduction']['max_abs_dp'] < 1e-6
    predictions = pd.read_parquet(out / 'perturbation_predictions.parquet')
    assert len(predictions) == 2 * 15 and predictions[LABELS].notna().all().all()
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary['rows']) == 14 * 12
    contract = json.loads((out / 'run_contract.json').read_text())
    assert contract['labels_used'] is False and contract['training_launched'] is False
    assert contract['predictions_sha256'] == sha(out / 'perturbation_predictions.parquet')
    with pytest.raises(FileExistsError):
        perturbation_probe.run(design, inputs, checkpoint, reference, out, execute=True)


def test_runner_stops_when_identity_does_not_reproduce(tmp_path, monkeypatch):
    design, inputs, spec, roles = miniature(tmp_path)
    checkpoint, reference = reference_run(tmp_path, design, inputs, spec, roles)
    saved = pd.read_parquet(reference / 'outer_predictions.parquet')
    saved[LABELS] = (saved[LABELS] + 0.05).clip(0, 1)
    saved.to_parquet(reference / 'outer_predictions.parquet', index=False)
    contract = json.loads((reference / 'run_contract.json').read_text())
    contract['outer_predictions_sha256'] = sha(reference / 'outer_predictions.parquet')
    (reference / 'run_contract.json').write_text(json.dumps(contract))
    surrogate_gpu(monkeypatch)
    out = tmp_path / 'probe'
    with pytest.raises(RuntimeError, match='does not reproduce'):
        perturbation_probe.run(design, inputs, checkpoint, reference, out, execute=True, limit=1)
    assert not (out / 'summary.json').exists()
