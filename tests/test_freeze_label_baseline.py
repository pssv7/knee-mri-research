import json

import numpy as np
import pandas as pd
import pytest

from src import config
from tools import freeze_label_baseline as freeze
from tools.build_label_review import sha

REPORTS = {'a': 'Synthetic report alpha: medial meniscus intact.',
           'b': 'Synthetic report beta: small effusion.',
           'c': 'Synthetic report gamma: no fracture.',
           'g': 'Synthetic gold report.'}


def write(root, rel, frame):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False) if rel.endswith('.parquet') else frame.to_csv(path, index=False)
    return path


def text(root, rel, content):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def build_root(root):
    ids = ['a', 'b', 'c']
    labels = pd.DataFrame({config.ID_COL: ids})
    for name in config.LABELS:
        labels[name] = [0.9, 0.1, 0.5]
        labels[name + '_weight'] = [1.0, 0.7, 0.4]
    write(root, freeze.LABELS_PATH, labels)
    reports = pd.DataFrame({config.ID_COL: list(REPORTS), 'Report': list(REPORTS.values())})
    for name in config.LABELS:
        reports[name] = [np.nan, np.nan, np.nan, 1.0]
    write(root, freeze.REPORTS_PATH, reports)
    write(root, freeze.FOLDS_PATH, pd.DataFrame({config.ID_COL: ids, 'fold': [0, 1, 2]}))
    write(root, freeze.GOLD_PATH, pd.DataFrame({config.ID_COL: ['g'], 'ACL': [1]}))
    for rel in freeze.DUMPS:
        write(root, rel, pd.DataFrame({config.ID_COL: ids, 'ACL': [1.0, 0.0, 0.5]}))
    oof = write(root, 'outputs/protocol/oof_fold0.parquet', pd.DataFrame({config.ID_COL: ['a'], 'ACL': [0.3]}))
    summary = text(root, 'data/meta/compare_protocol_incumbent_all_folds/summary.json', json.dumps(
        {'source_receipts': [{'family': 'protocol', 'fold': 0, 'checkpoint_sha256': 'c' * 64,
                              'oof': 'outputs/protocol/oof_fold0.parquet', 'oof_sha256': sha(oof)}]}))
    write(root, freeze.INCUMBENT_OOF, pd.DataFrame({config.ID_COL: ids, 'ACL': [0.2, 0.4, 0.6]}))
    write(root, freeze.RESOLUTION_OOF, pd.DataFrame({config.ID_COL: ids, 'ACL': [0.1, 0.5, 0.9]}))
    text(root, freeze.RESOLUTION_RECEIPT, json.dumps({'members': [{'fold': 0, 'checkpoint_sha256': 'd' * 64}]}))
    for rel in freeze.SOURCE_DOCS:
        text(root, rel, f'# {rel}\n')
    for rel in freeze.CODE:
        text(root, rel, f'# {rel}\n')
    prior = {freeze.LABELS_PATH: sha(root / freeze.LABELS_PATH),
             freeze.REPORTS_PATH: sha(root / freeze.REPORTS_PATH),
             freeze.INCUMBENT_OOF: sha(root / freeze.INCUMBENT_OOF),
             str(summary.relative_to(root)): sha(summary)}
    text(root, freeze.PRIOR_RECEIPT, json.dumps({'input_sha256': prior}))

    def resolver(r):
        return [oof], [{'path': str(oof.relative_to(r)), 'sha256': sha(oof)}], None, summary
    return resolver


@pytest.fixture
def frozen(tmp_path):
    resolver = build_root(tmp_path)
    path = freeze.create(tmp_path, 'v1', resolver)
    return tmp_path, resolver, path


def test_create_records_hashes_not_report_text(frozen):
    root, _, path = frozen
    raw = path.read_text()
    record = json.loads(raw)
    assert all(report not in raw for report in REPORTS.values())
    assert record['inputs']['labels']['rows'] == 3 and record['inputs']['labels']['label_order_ok']
    assert record['inputs']['gold'] == {**record['inputs']['gold'], 'studies': 1, 'in_training_labels': 0}
    assert all(record['cross_check']['matches'].values())
    hashes = pd.read_parquet(root / record['inputs']['reports']['report_hashes_file']['path'])
    assert list(hashes.columns) == [config.ID_COL, 'report_sha256'] and len(hashes) == 4


def test_verify_passes_on_unchanged_inputs(frozen):
    root, resolver, _ = frozen
    assert freeze.verify(root, 'v1', resolver) == ([], [])


def test_frozen_version_is_never_overwritten(frozen):
    root, resolver, _ = frozen
    with pytest.raises(FileExistsError):
        freeze.create(root, 'v1', resolver)


def test_same_inputs_freeze_identically(frozen):
    root, resolver, path = frozen
    again = json.loads(freeze.create(root, 'v2', resolver).read_text())
    first = json.loads(path.read_text())
    for record in (first, again):
        record['inputs']['reports'].pop('report_hashes_file')
    assert first == again


@pytest.mark.parametrize('rel, change, expected', [
    (freeze.LABELS_PATH, lambda f: f.assign(ACL=[0.0, 0.1, 0.5]), 'labels.sha256'),
    (freeze.FOLDS_PATH, lambda f: f.assign(fold=[1, 0, 2]), 'folds.sha256'),
    (freeze.DUMPS[0], lambda f: f.assign(ACL=[0.0, 0.0, 0.5]), f'extractor_dumps.{freeze.DUMPS[0]}.sha256'),
])
def test_verify_rejects_changed_tables(frozen, rel, change, expected):
    root, resolver, _ = frozen
    write(root, rel, change(pd.read_parquet(root / rel)))
    changed, _ = freeze.verify(root, 'v1', resolver)
    assert expected in changed


def test_verify_rejects_one_changed_report(frozen):
    root, resolver, _ = frozen
    reports = pd.read_csv(root / freeze.REPORTS_PATH)
    reports.loc[1, 'Report'] = 'Synthetic report beta: large effusion.'
    write(root, freeze.REPORTS_PATH, reports)
    changed, _ = freeze.verify(root, 'v1', resolver)
    assert 'reports.report_hashes_sha256' in changed
    assert 'report hashes (1 studies)' in changed
    assert all('effusion' not in item for item in changed)


def test_verify_rejects_changed_source_document(frozen):
    root, resolver, _ = frozen
    text(root, freeze.SOURCE_DOCS[0], '# edited\n')
    changed, _ = freeze.verify(root, 'v1', resolver)
    assert f'source_documents.{freeze.SOURCE_DOCS[0]}.sha256' in changed


def test_code_drift_is_reported_without_failing(frozen):
    root, resolver, _ = frozen
    text(root, 'tools/freeze_label_baseline.py', '# edited\n')
    changed, drift = freeze.verify(root, 'v1', resolver)
    assert changed == [] and drift == ['tools/freeze_label_baseline.py']


def test_create_refuses_inputs_that_differ_from_the_prior_receipt(tmp_path):
    resolver = build_root(tmp_path)
    labels = pd.read_parquet(tmp_path / freeze.LABELS_PATH)
    write(tmp_path, freeze.LABELS_PATH, labels.assign(ACL=[0.0, 0.1, 0.5]))
    with pytest.raises(ValueError, match='inputs differ'):
        freeze.create(tmp_path, 'v1', resolver)


def test_create_refuses_gold_inside_training_labels(tmp_path):
    resolver = build_root(tmp_path)
    labels = pd.read_parquet(tmp_path / freeze.LABELS_PATH)
    labels.loc[2, config.ID_COL] = 'g'
    write(tmp_path, freeze.LABELS_PATH, labels)
    prior = json.loads((tmp_path / freeze.PRIOR_RECEIPT).read_text())
    prior['input_sha256'][freeze.LABELS_PATH] = sha(tmp_path / freeze.LABELS_PATH)
    text(tmp_path, freeze.PRIOR_RECEIPT, json.dumps(prior))
    with pytest.raises(ValueError, match='gold'):
        freeze.create(tmp_path, 'v1', resolver)
