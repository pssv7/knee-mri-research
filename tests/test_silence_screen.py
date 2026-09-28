"""Silence masking, its explicit-label endpoint, generated runner and kernel."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import silence_mask
from test_colab_perturbation import ROOT, miniature
from test_mirror_screen import cpu_training


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f's{i:02d}' for i in range(n)]
    roles = pd.DataFrame({config.ID_COL: ids, 'component': ids,
                          'role': ['train'] * (n // 2) + ['outer'] * (n - n // 2)})
    labels = pd.DataFrame({config.ID_COL: ids})
    for f in config.LABELS:
        labels[f] = rng.choice([0.02, 0.9], n)
        labels[f + '_weight'] = rng.choice([0.25, 0.4, 0.7, 1.0], n)
    return labels, roles


def test_mask_touches_only_silent_cells_of_masked_findings_in_training_rows():
    labels, roles = frame()
    masked, counts = silence_mask.mask_silence(labels, roles)
    train = labels[config.ID_COL].isin(roles.loc[roles.role == 'train', config.ID_COL])
    for f in config.LABELS:
        before, after = labels[f + '_weight'], masked[f + '_weight']
        assert (masked[f] == labels[f]).all()  # targets never change
        if f in silence_mask.MASKED:
            zeroed = train & np.isclose(before, 0.25)
            assert (after[zeroed] == 0).all() and (after[~zeroed] == before[~zeroed]).all()
            assert counts[f] == int(zeroed.sum()) > 0
        else:
            assert (after == before).all()
    assert (labels.filter(like='_weight') > 0).all().all()  # input frame not modified


def test_explicit_truth_keeps_only_confident_cells():
    labels, roles = frame()
    ids = roles[config.ID_COL]
    truth = silence_mask.explicit_truth(labels, ids)
    w = labels.set_index(config.ID_COL).loc[ids, [f + '_weight' for f in config.LABELS]].to_numpy()
    assert (np.isnan(truth.to_numpy()) == (w < 0.7)).all()


def test_explicit_evaluation_rewards_ranking_explicit_labels_better():
    labels, roles = frame(200, 1)
    outer = roles[roles.role == 'outer'].sort_values(config.ID_COL)
    truth = (labels.set_index(config.ID_COL).loc[outer[config.ID_COL], config.LABELS] >= .5).to_numpy(float)
    rng = np.random.default_rng(2)
    control = truth * 0.2 + rng.uniform(0, 1, truth.shape)
    candidate = control.copy()
    masked = [config.LABELS.index(f) for f in silence_mask.MASKED]
    candidate[:, masked] = truth[:, masked] * 0.8 + rng.uniform(0, 1, (len(truth), len(masked)))
    report = silence_mask.explicit_evaluation(labels, roles, control, candidate, draws=300)
    d = report['decision']
    assert d['explicit_mean_delta'] > 0.05 and d['paired_lower_bound_positive'] and d['point_screen_passed']
    assert all(v == 0 for v in report['unmasked_full_delta_guard'].values())
    assert not d['promotion_allowed']
    worse = silence_mask.explicit_evaluation(labels, roles, candidate, control, draws=300)['decision']
    assert worse['explicit_mean_delta'] < 0 and not worse['point_screen_passed']


def test_runner_adapter_fails_closed_and_candidate_trains_on_cpu(tmp_path, monkeypatch):
    from tools.prepare_kaggle_silence_screen import LABEL_WEIGHTING, silence_source
    from tools.evaluate_nested_proxy import evaluate
    frozen = (ROOT / 'tools/train_nested_proxy_control.py').read_text()
    with pytest.raises(ValueError, match='needs review'):
        silence_source(frozen.replace('recipe = data\n', 'recipe = tuple(data)\n'))
    design, inputs, spec, roles = miniature(tmp_path)
    cpu_training(monkeypatch)
    import torch
    torch.set_num_threads(1)
    for arm, source in [('control', frozen), ('candidate', silence_source(frozen))]:
        scope = {'__file__': str(ROOT / 'tools/train_nested_proxy_control.py'), '__name__': f'arm_{arm}'}
        exec(source.split("if __name__ == '__main__':", 1)[0], scope)
        scope['run'](design, inputs, tmp_path / 'runs' / arm, execute=True)
    candidate = json.loads((tmp_path / 'runs/candidate/run_contract.json').read_text())
    assert candidate['arm'] == 'candidate' and candidate['label_weighting'] == LABEL_WEIGHTING
    assert set(candidate['silence_masked']) == set(silence_mask.MASKED)
    report = evaluate(design, tmp_path / 'runs/control', tmp_path / 'runs/candidate', tmp_path / 'e.json', draws=200)
    assert report['macro_delta'] is not None


def test_kernel_fits_kaggle_and_uses_the_declared_endpoint(tmp_path):
    from tools.prepare_kaggle_silence_screen import main
    out = main(destination=tmp_path / 'kernel')
    source = (out / 'silence_screen.py').read_text()
    assert len(source.encode()) < 1_000_000
    compile(source, 'silence_screen.py', 'exec')
    assert "('candidate', 'tools/train_silence.py')" in source and 'train_mirror.py' not in source
    assert "decision = explicit['decision']" in source and 'training_launched=True' in source
    assert source.count('training_launched=True') == 1
    scope = {}
    exec(source.split('\nimport subprocess, sys\n', 1)[0], scope)
    assert {'src/model/silence_mask.py', 'tools/train_silence.py'} <= set(scope['EMBEDDED'])
