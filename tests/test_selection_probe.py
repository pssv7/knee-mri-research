"""Series-selection policies and the inference-only selection runner."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import selection_policies as policies
from src.model.dataset import select_series
from src.validation.proxy import ID, LABELS, sha
from test_colab_perturbation import FROZEN, ROOT, entry, reference_run, surrogate_gpu

PROTOCOLS = [('Sagittal', 1, 1), ('Sagittal', 0, 0), ('Coronal', 1, 1), ('Coronal', 0, 0),
             ('Axial', 1, 1), ('Axial', 0, 0), ('Sagittal', 1, 0)]


def rows_from(tuples, thickness=None):
    frame = pd.DataFrame(tuples, columns=['Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'])
    frame['SeriesInstanceUID'] = [f'ser{i:02d}' for i in range(len(frame))]
    if thickness is not None:
        frame['slice_thickness'] = thickness
        frame['n_instances'] = 20
        frame['pixel_spacing'] = 0.4
    return frame


@pytest.mark.parametrize('seed', range(40))
def test_greedy_with_spread_is_exactly_the_frozen_rule(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 12))
    tuples = [PROTOCOLS[i] for i in rng.integers(0, len(PROTOCOLS), count)]
    rows = rows_from(tuples)
    for cap in (3, 5, 7):
        assert np.array_equal(policies.greedy(rows, cap, policies.spread), select_series(rows, cap, 'protocol'))
    assert np.array_equal(policies.choose('protocol5', rows), select_series(rows, 5, 'protocol'))


def test_sagittal_fluid_policy_always_keeps_one_when_present():
    # Seven distinct protocols with the only fluid-sensitive sagittal series last
    # in UID order: the frozen rule can drop it, the policy must not.
    tuples = [('Sagittal', 0, 0), ('Coronal', 1, 1), ('Coronal', 0, 0), ('Axial', 1, 1),
              ('Axial', 0, 0), ('Coronal', 1, 0), ('Sagittal', 1, 1)]
    rows = rows_from(tuples)
    frozen = policies.choose('protocol5', rows)
    assert 6 not in frozen
    kept = policies.choose('sagfluid5', rows)
    assert 6 in kept and len(kept) == 5
    no_fluid = rows_from([t for t in tuples if t != ('Sagittal', 1, 1)])
    assert np.array_equal(policies.choose('sagfluid5', no_fluid), policies.choose('protocol5', no_fluid))
    # Where the frozen rule already keeps one, the policy changes nothing.
    first = rows_from([('Sagittal', 1, 1)] + tuples[:-1])
    assert np.array_equal(policies.choose('sagfluid5', first), policies.choose('protocol5', first))


def test_quality_policy_keeps_the_thinner_duplicate():
    tuples = [('Sagittal', 1, 1), ('Coronal', 1, 1), ('Axial', 1, 1), ('Sagittal', 0, 0),
              ('Coronal', 0, 0), ('Sagittal', 1, 1), ('Coronal', 0, 0)]
    rows = rows_from(tuples, thickness=[4.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.5])
    kept = policies.choose('quality5', rows)
    assert 5 in kept and 0 not in kept  # the 2 mm sagittal PD-FS beats the 4 mm one
    assert len(kept) == 5
    missing = rows.assign(slice_thickness=np.nan)
    assert len(policies.choose('quality5', missing)) == 5


def test_caps_and_all():
    rows = rows_from(PROTOCOLS * 2)
    assert len(policies.choose('protocol7', rows)) == 7
    assert np.array_equal(policies.choose('all', rows), np.arange(14))
    small = rows_from(PROTOCOLS[:4])
    for policy in policies.POLICIES:
        assert np.array_equal(policies.choose(policy, small), np.arange(4))
    with pytest.raises(KeyError):
        policies.choose('random', rows)


def test_series_quality_summarises_headers():
    index = pd.DataFrame({'SeriesInstanceUID': ['a', 'a', 'b'], 'SOPInstanceUID': ['1', '2', '3'],
                          'SliceThickness': [3.0, 3.0, 1.5],
                          'PixelSpacing': [np.array([0.4, 0.4]), np.array([0.4, 0.4]), None]})
    table = policies.series_quality(index).set_index('SeriesInstanceUID')
    assert table.loc['a', 'n_instances'] == 2 and table.loc['a', 'pixel_spacing'] == pytest.approx(0.4)
    assert table.loc['b', 'slice_thickness'] == 1.5 and np.isnan(table.loc['b', 'pixel_spacing'])


def multi_series_miniature(tmp_path):
    """Frozen-preflight-valid miniature where one outer study has seven series."""
    design = tmp_path / 'control'
    design.mkdir()
    ids = [f's{i}' for i in range(6)]
    roles = pd.DataFrame({ID: ids, 'component': ids, 'role': ['train'] * 2 + ['selection'] * 2 + ['outer'] * 2})
    roles.to_csv(design / 'roles.csv', index=False)
    labels = pd.DataFrame({ID: ids})
    for label in LABELS:
        labels[label] = [.1, .9] * 3
        labels[label + '_weight'] = .7
    labels.to_parquet(design / 'proxy_labels.parquet')
    pd.DataFrame({ID: ['protected'], 'component': ['protected']}).to_csv(tmp_path / 'exclusions.csv', index=False)
    spec = json.loads((FROZEN / 'control/spec.json').read_text())
    spec.update(roles_sha256=sha(design / 'roles.csv'), proxy_labels_sha256=sha(design / 'proxy_labels.parquet'),
                protected_exclusions_sha256=sha(tmp_path / 'exclusions.csv'),
                runtime_code_sha256={'src/validation/proxy.py': sha(ROOT / 'src/validation/proxy.py')})
    spec.pop('raw_counts_sha256')
    (design / 'spec.json').write_text(json.dumps(spec))
    rows = []
    for study in ids:
        tuples = [('Sagittal', 0, 0), ('Coronal', 1, 1), ('Coronal', 0, 0), ('Axial', 1, 1),
                  ('Axial', 0, 0), ('Coronal', 1, 0), ('Sagittal', 1, 1)] if study == 's5' else [('Sagittal', 1, 1)]
        for k, (plane, fluid, fat) in enumerate(tuples):
            rows.append({ID: study, 'SeriesInstanceUID': f'{study}_ser{k}', 'Fluid_Sensitive': fluid,
                         'Fat_Suppression': fat, 'Anatomical_Plane': plane})
    series = pd.DataFrame(rows)
    series.to_csv(tmp_path / 'series.csv', index=False)
    pd.DataFrame({'SeriesInstanceUID': series.SeriesInstanceUID,
                  'side_predicted': ['L', 'R'] * (len(series) // 2) + ['R'] * (len(series) % 2)}
                 ).to_parquet(tmp_path / 'sides.parquet')
    rng = np.random.default_rng(0)
    np.save(tmp_path / 'cache.npy', rng.integers(0, 256, (len(series), 24, 224, 224), dtype=np.uint8))
    manifest = series[[ID, 'SeriesInstanceUID']].copy()
    manifest['position'] = np.arange(len(series))
    manifest.to_parquet(tmp_path / 'manifest.parquet')
    inputs = {'series_meta': entry(tmp_path / 'series.csv'), 'side_predictions': entry(tmp_path / 'sides.parquet'),
              'shards': [{'array': entry(tmp_path / 'cache.npy'), 'manifest': entry(tmp_path / 'manifest.parquet')}]}
    (tmp_path / 'inputs.json').write_text(json.dumps(inputs))
    quality = series[['SeriesInstanceUID']].assign(slice_thickness=3.0, n_instances=20, pixel_spacing=0.4)
    quality.to_parquet(tmp_path / 'quality.parquet', index=False)
    return design, tmp_path / 'inputs.json', spec, roles


def test_runner_reproduces_reference_and_rescores_only_changed_studies(tmp_path, monkeypatch):
    from tools import selection_probe
    design, inputs, spec, roles = multi_series_miniature(tmp_path)
    checkpoint, reference = reference_run(tmp_path, design, inputs, spec, roles)
    quality = tmp_path / 'quality.parquet'
    pre = selection_probe.run(design, inputs, checkpoint, reference, None, quality_path=quality)
    assert pre['status'] == 'preflight_passed'
    surrogate_gpu(monkeypatch)
    out = tmp_path / 'probe'
    result = selection_probe.run(design, inputs, checkpoint, reference, out, execute=True, quality_path=quality)
    assert result['reproduction']['passed'] and result['reproduction']['max_abs_dp'] < 1e-6
    predictions = pd.read_parquet(out / 'selection_predictions.parquet')
    assert len(predictions) == 2 * len(policies.POLICIES)
    changed = predictions[predictions.changed]
    assert set(changed[ID]) == {'s5'} and {'sagfluid5', 'protocol7', 'all'} <= set(changed.policy)
    selections = pd.read_parquet(out / 'selections.parquet').set_index(['policy', ID])
    assert selections.loc[('all', 's5'), 'n_kept'] == 7 and selections.loc[('protocol5', 's5'), 'n_kept'] == 5
    assert 's5_ser6' in json.loads(selections.loc[('sagfluid5', 's5'), 'kept'])
    assert 's5_ser6' not in json.loads(selections.loc[('protocol5', 's5'), 'kept'])
    contract = json.loads((out / 'run_contract.json').read_text())
    assert contract['labels_used'] is False and contract['quality_table_sha256'] == sha(quality)


def test_kernel_reuses_the_perturbation_body_and_fits_kaggle(tmp_path):
    from tools.prepare_kaggle_selection import main
    out = main(destination=tmp_path / 'kernel')
    source = (out / 'selection_kernel.py').read_text()
    assert len(source.encode()) < 1_000_000
    assert "code / 'tools/selection_probe.py'" in source and 'perturbation_probe.py\')' not in source
    scope = {}
    exec(source.split('\nimport subprocess, sys\n', 1)[0], scope)
    assert {'src/model/selection_policies.py', 'tools/selection_probe.py',
            'design/series_quality.parquet', 'tools/perturbation_probe.py'} <= set(scope['EMBEDDED'])
    assert set(scope['EMBEDDED']) <= set(scope['MANIFEST'])
