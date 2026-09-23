import copy
import hashlib
import json
from pathlib import Path
import ast
import pathlib

import pandas as pd
import pytest

from src.labels.reviewed import apply_reviewed
from tools.prepare_reviewed_labels import stage_source


def inputs():
    labels = pd.DataFrame({'StudyInstanceUID': ['development', 'val1', 'val3'],
                           'ACL': [.9, .7, .2], 'ACL_weight': [1., .7, .4]})
    folds = pd.DataFrame({'StudyInstanceUID': ['development', 'val1', 'val3'], 'fold': [0,1,3]})
    reports = pd.DataFrame({'Report': ['ACL intact.']}, index=['development'])
    corrections = [{'StudyInstanceUID': 'development', 'finding': 'ACL',
        'old_target': .9, 'new_target': 0., 'evidence': 'ACL intact.',
        'report_sha256': hashlib.sha256(b'ACL intact.').hexdigest()}]
    return labels, folds, corrections, reports


@pytest.mark.parametrize('fold', [1,3])
def test_training_only_with_immutable_inputs(fold):
    labels, folds, corrections, reports = inputs()
    before = labels.copy(deep=True)
    result = apply_reviewed(labels, folds, corrections, fold, reports)
    pd.testing.assert_frame_equal(labels, before)
    assert result.loc[0, 'ACL'] == 0
    pd.testing.assert_frame_equal(result.iloc[1:], before.iloc[1:])
    pd.testing.assert_series_equal(result.ACL_weight, before.ACL_weight)


@pytest.mark.parametrize('mutation', ['heldout', 'old_target', 'quote', 'duplicate'])
def test_rejects_unreviewed_or_leaking_changes(mutation):
    labels, folds, corrections, reports = inputs()
    if mutation == 'heldout':
        corrections[0]['StudyInstanceUID'] = 'val3'
    elif mutation == 'old_target':
        corrections[0]['old_target'] = .8
    elif mutation == 'quote':
        corrections[0]['evidence'] = 'ACL torn.'
    else:
        corrections.append(copy.deepcopy(corrections[0]))
    with pytest.raises(ValueError):
        apply_reviewed(labels, folds, corrections, 1, reports)


def test_actual_candidate_has_exactly_reviewed_changes():
    root = Path(__file__).resolve().parents[1]
    labels = pd.read_parquet(root/'data/meta/train_labels.parquet')
    candidate = pd.read_parquet(root/'outputs/reviewed_labels_v1/train_labels_reviewed_v1.parquet')
    corrections = json.loads((root/'outputs/reviewed_labels_v1/corrections.json').read_text())
    folded = pd.read_parquet(root/'outputs/ensemble_16_24/reconstructed_folds.parquet')
    reports = pd.read_csv(root/'data/meta/train.csv').set_index('StudyInstanceUID')
    expected = apply_reviewed(labels, folded, corrections, 1, reports)
    pd.testing.assert_frame_equal(candidate, expected)
    assert (labels != candidate).to_numpy().sum() == len(corrections) == 11
    assert labels.StudyInstanceUID.equals(candidate.StudyInstanceUID)


def test_staged_review_hook_executes_before_the_notebooks_later_imports(tmp_path, monkeypatch):
    """Exercise the exact injected preflight, not merely compilation."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    corrections = json.loads((root / 'outputs/reviewed_labels_v1/corrections.json').read_text())
    label_hash = hashlib.sha256((root / 'data/meta/train_labels.parquet').read_bytes()).hexdigest()
    labels = pd.read_parquet(root / 'data/meta/train_labels.parquet')
    split = pd.read_parquet(root / 'outputs/ensemble_16_24/reconstructed_folds.parquet')
    folded = labels.merge(split, on='StudyInstanceUID', validate='one_to_one')
    monkeypatch.chdir(tmp_path)

    for arm in ('control', 'candidate'):
        source = stage_source(notebook, label_hash, corrections, arm, fold=1)
        tree = ast.parse(source)
        start = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Import)
                     and [alias.name for alias in node.names] == ['hashlib']
                     and i + 1 < len(tree.body)
                     and isinstance(tree.body[i + 1], ast.Import)
                     and [alias.name for alias in tree.body[i + 1].names] == ['json'])
        end = next(i for i, node in enumerate(tree.body[start:], start)
                   if any(isinstance(item, ast.Constant) and item.value == 'label_review_contract.json'
                          for item in ast.walk(node)))
        namespace = {'folded': folded.copy(deep=True), 'pd': pd, 'pathlib': pathlib}
        exec(compile(ast.fix_missing_locations(ast.Module(tree.body[start:end + 1], [])),
                     f'{arm}_preflight.py', 'exec'), namespace)
        contract = json.loads((tmp_path / 'label_review_contract.json').read_text())
        assert contract['arm'] == arm
        assert contract['validation_labels_changed'] is False
        assert contract['folds_changed'] is False
