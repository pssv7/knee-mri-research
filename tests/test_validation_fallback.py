"""Leakage, metric, and real-file preflight regressions for the new protocol."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config
from src.labels import fuse, pf_mention_candidate
from src.validation import proxy
from tools.evaluate_nested_proxy import evaluate
from tools.train_nested_proxy_control import preflight, run

ID, L = config.ID_COL, config.LABELS


def labels(n=200):
    rng = np.random.default_rng(12)
    d = pd.DataFrame({ID: [f's{i:04}' for i in range(n)], 'component': [f'g{i//2:04}' for i in range(n)]})
    for l in L:
        d[l] = rng.random(n)
        d[l + '_weight'] = .7
    return d


def test_links_are_transitive_and_protected_components_are_excluded():
    m = pd.DataFrame({ID: ['gold', 'a', 'b', 'c'], 'normalized_report_hash': ['x', 'x', 'y', 'z']})
    near = pd.DataFrame({'study_a': ['a'], 'study_b': ['b']})
    c = proxy.linked_components(m, near)
    assert c.set_index(ID).loc[['gold', 'a', 'b'], 'component'].nunique() == 1
    assert proxy.exclude_components(c, ['gold'])[ID].tolist() == ['c']


def test_unknown_patient_link_fails():
    m = pd.DataFrame({ID: ['a'], 'normalized_report_hash': ['x']})
    with pytest.raises(ValueError, match='unknown study'):
        proxy.linked_components(m, pd.DataFrame(), pd.DataFrame({ID: ['z'], 'patient_group': ['p']}))


def test_nested_roles_disjoint_component_safe_and_order_invariant():
    f = labels()
    f['outer_fold'] = proxy.group_folds(f)
    shuffled = f.sample(frac=1, random_state=99)
    pd.testing.assert_series_equal(proxy.group_folds(shuffled).sort_index(), f.outer_fold, check_names=False)
    seen = []
    for fold in range(5):
        roles = proxy.nested_roles(f, fold)
        assert set(roles.role) == {'train', 'selection', 'outer'}
        assert roles.groupby('component').role.nunique().max() == 1
        seen.extend(roles.loc[roles.role == 'outer', ID])
    assert sorted(seen) == sorted(f[ID])


def test_component_split_is_rejected():
    roles = pd.DataFrame({ID: ['a', 'b', 'c'], 'component': ['same', 'same', 'other'],
                          'role': ['train', 'selection', 'outer']})
    with pytest.raises(ValueError, match='crosses split'):
        proxy.validate_roles(roles)


def test_training_transform_cannot_read_held_out_label_distribution():
    raw = labels(20).drop(columns='component')
    reference = raw.copy()
    train = raw[ID][:12].tolist()
    a = proxy.training_only_sharpen(raw, reference, train)
    changed_raw, changed_ref = raw.copy(), reference.copy()
    changed_raw.loc[12:, L] = 1
    changed_ref.loc[12:, L] = 0
    b = proxy.training_only_sharpen(changed_raw, changed_ref, train)
    pd.testing.assert_frame_equal(a.iloc[:12], b.iloc[:12])
    expected = fuse.sharpen_targets(raw.iloc[:12], reference.iloc[:12])
    np.testing.assert_allclose(a.iloc[:12][L], expected[L], atol=1e-14)


def test_metric_matches_independent_implementation_with_ties():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=(200, 12))
    scores = rng.integers(0, 4, size=(200, 12))/3
    per, macro = proxy.strict_aucs(y, scores)
    np.testing.assert_allclose(per, roc_auc_score(y, scores, average=None))
    assert macro == pytest.approx(roc_auc_score(y, scores))


def test_macro_is_unestimable_not_eleven_finding_average():
    y = np.tile([0, 1], (12, 20)).T.astype(float)
    y[:, 4] = 1
    per, macro = proxy.strict_aucs(y, np.full(y.shape, .5))
    assert macro is None and np.isnan(per[4])
    y[:, 4] = np.nan
    assert proxy.strict_aucs(y, np.full(y.shape, .5))[1] is None
    y[0, 4] = np.inf
    with pytest.raises(ValueError, match='Invalid'):
        proxy.strict_aucs(y, np.full(y.shape, .5))


def test_paired_bootstrap_preserves_clusters_and_cannot_promote_proxy_gain():
    y = np.tile([0, 1], (12, 50)).T.astype(float)
    result = proxy.paired_proxy_evaluation(y, 1-y, y, [f'c{i//2}' for i in range(100)], draws=100)
    assert result['macro_delta'] == 1
    assert result['paired_sampling_95ci'] == [1, 1]
    assert result['valid_bootstrap_draws'] == 100
    assert result['image_gain_demonstrated'] is False
    assert result['promotion'] == 'not_supported_by_proxy_labels'


def test_blending_and_selected_fold_contracts_are_rejected():
    roles = pd.DataFrame({ID: ['a', 'b', 'c'], 'component': ['a', 'b', 'c'],
                          'role': ['train', 'selection', 'outer']})
    pred = pd.DataFrame({ID: ['c'], **{l: [.5] for l in L}})
    c = {'train_uids': ['a'], 'selection_uids': ['b'], 'outer_uids': ['c'],
         'model_count': 1, 'checkpoint_selection_role': 'selection', 'initialization': 'original_pretrained'}
    proxy.validate_prediction_provenance(roles, c, pred)
    for change in [{'model_count': 2}, {'selection_uids': ['c']},
                   {'checkpoint_selection_role': 'outer'}, {'initialization': 'historical_knee_checkpoint'}]:
        with pytest.raises(ValueError):
            proxy.validate_prediction_provenance(roles, {**c, **change}, pred)


@pytest.mark.parametrize('text', [
    'Grade III chondromalacia of the patella.',
    'Retropatellare Chondromalazie Grad II.',
    'Hondromalacija patele II. stupnja.',
    'Possible chondromalacia of the patella.',
    'No chondromalacia of the patella.',
    'Patella and medial femoral condyle cartilage defects.',
    'The patella is normal.',
])
def test_narrow_pf_rule_abstains(text):
    assert pf_mention_candidate.propose(text, .01) is None


def test_pf_proposal_is_review_only_and_does_not_rewrite_targets():
    p = pf_mention_candidate.propose('Patellofemoral osteoarthritis with osteophytes.', .01)
    assert p['action'] == 'review_only'
    assert pf_mention_candidate.propose('Patellofemoral osteoarthritis.', .42) is None


@pytest.fixture
def frozen_design(tmp_path):
    design = tmp_path / 'outer0'
    design.mkdir()
    ids = [f's{i}' for i in range(12)]
    roles = pd.DataFrame({ID: ids, 'component': ids, 'role': ['train']*4+['selection']*4+['outer']*4})
    roles.to_csv(design / 'roles.csv', index=False)
    table = pd.DataFrame({ID: ids})
    for l in L:
        table[l] = [.1, .9]*6
        table[l + '_weight'] = .7
    table.to_parquet(design / 'proxy_labels.parquet', index=False)
    pd.DataFrame({ID: ['protected'], 'component': ['protected'], 'reason': ['gold']}).to_csv(tmp_path / 'exclusions.csv', index=False)
    recipe = {'backbone': 'convnext_tiny.fb_in22k_ft_in1k', 'cache_size': 224,
              'slices': 16, 'max_series': 5, 'series_selection': 'protocol',
              'epochs': 12, 'effective_batch': 8, 'learning_rate': .0001,
              'schedule': 'onecycle', 'warmup_epochs': 3.6}
    root = Path(__file__).resolve().parents[1]
    spec = {**recipe, 'seed': 0, 'roles_sha256': proxy.sha(design / 'roles.csv'),
            'proxy_labels_sha256': proxy.sha(design / 'proxy_labels.parquet'),
            'protected_exclusions_sha256': proxy.sha(tmp_path / 'exclusions.csv'),
            'initialization': 'original_pretrained', 'model_count': 1,
            'checkpoint_selection_role': 'selection', 'gold_monitoring': False,
            'endpoint': 'mined_report_label_macro_roc_auc',
            'runtime_code_sha256': {'src/validation/proxy.py': proxy.sha(root / 'src/validation/proxy.py')}}
    (design / 'spec.json').write_text(json.dumps(spec))
    series = pd.DataFrame({ID: ids, 'SeriesInstanceUID': [f'ser{i}' for i in range(12)],
                           'Fluid_Sensitive': 1, 'Fat_Suppression': 1, 'Anatomical_Plane': 'Sagittal'})
    series.to_csv(tmp_path / 'series.csv', index=False)
    pd.DataFrame({'SeriesInstanceUID': series.SeriesInstanceUID, 'side_predicted': 'L'}).to_parquet(tmp_path / 'sides.parquet')
    np.save(tmp_path / 'cache.npy', np.ones((12, 24, 224, 224), dtype=np.uint8))
    manifest = series[[ID, 'SeriesInstanceUID']].copy()
    manifest['position'] = np.arange(12)
    manifest.to_parquet(tmp_path / 'cache_manifest.parquet')
    def entry(name):
        return {'path': name, 'sha256': proxy.sha(tmp_path / name)}
    inputs = {'series_meta': entry('series.csv'), 'side_predictions': entry('sides.parquet'),
              'shards': [{'array': entry('cache.npy'), 'manifest': entry('cache_manifest.parquet')}]}
    (tmp_path / 'inputs.json').write_text(json.dumps(inputs))
    return design, tmp_path / 'inputs.json', spec, roles


def test_real_file_preflight_never_launches_training(frozen_design):
    design, inputs, _, _ = frozen_design
    result = run(design, inputs, None)
    assert result['status'] == 'preflight_passed'
    assert result['training_launched'] is False
    assert result['counts'] == {'train': 4, 'selection': 4, 'outer': 4}


def test_preflight_rejects_modified_labels(frozen_design):
    design, inputs, _, _ = frozen_design
    f = pd.read_parquet(design / 'proxy_labels.parquet')
    f.loc[0, 'ACL'] = .999
    f.to_parquet(design / 'proxy_labels.parquet')
    with pytest.raises(ValueError, match='hash mismatch'):
        preflight(design, inputs)


def test_preflight_rejects_missing_series_even_with_updated_input_hash(frozen_design):
    design, inputs, _, _ = frozen_design
    path = inputs.parent / 'cache_manifest.parquet'
    pd.read_parquet(path).iloc[:-1].to_parquet(path)
    config = json.loads(inputs.read_text())
    config['shards'][0]['manifest']['sha256'] = proxy.sha(path)
    inputs.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='coverage'):
        preflight(design, inputs)


def test_preflight_rejects_patient_link_to_protected_set(frozen_design):
    design, inputs, spec, _ = frozen_design
    path = design.parent / 'exclusions.csv'
    pd.DataFrame({ID: ['protected'], 'component': ['s0']}).to_csv(path, index=False)
    spec['protected_exclusions_sha256'] = proxy.sha(path)
    (design / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='leaked'):
        preflight(design, inputs)


@pytest.mark.parametrize('bad_count', [False, True])
def test_distinct_sampler_requires_matching_source_counts(frozen_design, bad_count):
    design, inputs, spec, roles = frozen_design
    manifest_path = inputs.parent / 'cache_manifest.parquet'
    manifest = pd.read_parquet(manifest_path)
    manifest['n_slices_raw'] = 24
    manifest['mirrored'] = False
    manifest.to_parquet(manifest_path)
    source = manifest[[ID, 'SeriesInstanceUID']].copy()
    source['raw_slices'] = 23 if bad_count else 24
    source.to_parquet(design.parent / 'raw_counts.parquet')
    spec.update(sampling_policy='distinct_raw_17_23_v1',
                raw_counts_sha256=proxy.sha(design.parent / 'raw_counts.parquet'))
    (design / 'spec.json').write_text(json.dumps(spec))
    cfg = json.loads(inputs.read_text())
    cfg['shards'][0]['manifest']['sha256'] = proxy.sha(manifest_path)
    inputs.write_text(json.dumps(cfg))
    if bad_count:
        with pytest.raises(ValueError, match='raw counts disagree'):
            preflight(design, inputs)
    else:
        assert run(design, inputs, None)['training_launched'] is False


def test_end_to_end_proxy_comparison_and_no_overwrite(frozen_design, tmp_path):
    design, inputs, spec, roles = frozen_design
    for arm in ['control', 'candidate']:
        d = tmp_path / arm
        d.mkdir()
        (d / 'selected.pt').write_bytes(b'synthetic checkpoint test fixture')
        outer = roles[roles.role == 'outer']
        pred = pd.DataFrame({ID: outer[ID].tolist(), **{l: [.2, .8, .2, .8] for l in L}})
        pred.to_parquet(d / 'outer_predictions.parquet')
        c = {**spec, 'training_recipe': {'same': True}, 'initial_weights_sha256': 'same_init',
             'runtime_versions': {'test': 'synthetic'}, 'input_config_sha256': proxy.sha(inputs),
             'checkpoint_sha256': proxy.sha(d / 'selected.pt'),
             'outer_predictions_sha256': proxy.sha(d / 'outer_predictions.parquet')}
        for role, name in [('train', 'train_uids'), ('selection', 'selection_uids'), ('outer', 'outer_uids')]:
            c[name] = roles.loc[roles.role == role, ID].tolist()
        (d / 'run_contract.json').write_text(json.dumps(c))
    out = tmp_path / 'evaluation.json'
    r = evaluate(design, tmp_path/'control', tmp_path/'candidate', out, draws=100)
    assert r['macro_delta'] == 0 and r['image_gain_demonstrated'] is False
    with pytest.raises(FileExistsError):
        evaluate(design, tmp_path/'control', tmp_path/'candidate', out, draws=100)
    c = json.loads((tmp_path/'candidate/run_contract.json').read_text())
    c['selection_uids'] = c['outer_uids']
    (tmp_path/'candidate/run_contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='selection_uids'):
        evaluate(design, tmp_path/'control', tmp_path/'candidate', tmp_path/'rejected.json', draws=100)
