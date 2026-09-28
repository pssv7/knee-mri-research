"""384-only notebook source and safe completion orchestration checks (no GPU)."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools import resolution_completion as completion

ROOT = Path(__file__).resolve().parents[1]


def test_generated_remaining_notebook():
    notebook = json.loads((ROOT / 'notebooks/21_resolution384_remaining_folds_colab.ipynb').read_text())
    sources = None
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        if text.startswith('%pip'):
            text = '\n'.join(text.splitlines()[1:])
        compile(text, '<cell>', 'exec')
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TRAINING_SOURCES_BY_FOLD' for t in node.targets):
                sources = ast.literal_eval(node.value)
    assert set(sources) == {0, 2, 3, 4}
    for fold, phases in sources.items():
        assert set(phases) == {'smoke', 'full'}
        for phase, source in phases.items():
            compile(source, '<train>', 'exec')
            assert f'FOLD = {fold}' in source
            assert 'config.CACHE_SIZE = 384' in source
            assert 'set_grad_checkpointing(True)' in source
            assert 'REFERENCE_CONTRACT' in source
            assert "monitor='val_macro'" in source
            assert 'SEED = 0' in source
            for old in ('frozen (retired)', 'OFF-REFERENCE', 'DIVERGENCE REPRODUCES',
                        'comparisons that matter', 'Ship the averaged weights'):
                assert old not in source
            if phase == 'full':
                assert 'gold_predictions_fold' in source and 'oof_fold' in source
                assert 'Gold-58 diagnostic AUC' in source
                assert 'max_epochs=EPOCHS' in source
            else:
                assert 'limit_train_batches=8' in source


def fixture_inputs(tmp_path, monkeypatch):
    reference = tmp_path / 'old' / 'attempt' / 'full' / 'resolution384_fold1'
    reference.mkdir(parents=True)
    contract = {'fold': 1, 'cache_size': 384, 'seed': 0, 'steps': 5}
    (reference / 'run_contract.json').write_text(json.dumps(contract))
    (reference.parent / 'resolution_run.json').write_text(json.dumps({'state': 'complete', 'smoke': False}))
    (reference / 'convnext_fold1.pt').write_bytes(b'checkpoint')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (reference / 'convnext_fold1.pt.json').write_text(json.dumps({'sha256': digest(reference / 'convnext_fold1.pt')}))
    inputs = {'metadata_sha256': {'labels': 'abc'}, 'side_predictions_sha256': 'side', 'shards': [{'shard': 0}]}
    (reference.parent.parent / 'input_receipt.json').write_text(json.dumps(inputs))
    local = tmp_path / 'local'
    local.mkdir()
    (local / 'input_receipt.json').write_text(json.dumps(inputs))
    monkeypatch.setattr(completion, 'digest', digest, raising=False)
    return reference, local, contract


@pytest.mark.parametrize('folds', [[1], [], [0, 0], [5]])
def test_rejects_fold1_and_invalid_lists(folds, tmp_path):
    with pytest.raises(ValueError):
        completion.run_remaining_folds({}, tmp_path, tmp_path, folds, tmp_path)


def test_injects_reference_and_never_trains_fold1(tmp_path, monkeypatch):
    reference, local, _ = fixture_inputs(tmp_path, monkeypatch)
    calls = []
    def run(sources, root, inputs, fold, size):
        calls.append(fold)
        assert size == 384
        assert all(s.startswith('REFERENCE_CONTRACT = ') for s in sources.values())
        return str(root)
    monkeypatch.setattr(completion, 'run_training', run, raising=False)
    sources = {f: {'smoke': 'pass', 'full': 'pass'} for f in (0, 2, 3, 4)}
    completion.run_remaining_folds(sources, tmp_path / 'out', local, [0, 2, 3, 4], reference)
    assert calls == [0, 2, 3, 4]


def test_skips_only_hash_verified_completion(tmp_path, monkeypatch):
    reference, local, contract = fixture_inputs(tmp_path, monkeypatch)
    attempt = tmp_path / 'out' / 'fold0' / 'attempt_test'
    output = attempt / 'full' / 'resolution384_fold0'
    output.mkdir(parents=True)
    contract['fold'] = 0
    (output / 'run_contract.json').write_text(json.dumps(contract))
    required = ['convnext_fold0.pt', 'convnext_fold0.pt.json', 'oof_fold0.parquet',
                'gold_predictions_fold0.parquet', 'folds.parquet']
    for name in required:
        (output / name).write_bytes(b'fixture')
    progress = {'state': 'complete', 'sha256': {name: completion.digest(output / name)
                for name in required + ['run_contract.json']}}
    (attempt / 'progress.json').write_text(json.dumps(progress))
    monkeypatch.setattr(completion, 'run_training', lambda *a, **k: pytest.fail('must skip'), raising=False)
    assert completion.run_remaining_folds({}, tmp_path / 'out', local, [0], reference) == {0: str(output)}
    (output / 'run_contract.json').write_text(json.dumps(contract) + '\n')
    with pytest.raises(AssertionError, match='artifact changed'):
        completion.run_remaining_folds({}, tmp_path / 'out', local, [0], reference)
