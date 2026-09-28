"""Seed-override runner and the seed-repeat kernel."""
import json

import pandas as pd
import pytest

from src.validation.proxy import validate_prediction_provenance
from test_colab_perturbation import ROOT, miniature
from test_mirror_screen import cpu_training


def test_seed_is_the_only_change_and_it_changes_the_initialisation(tmp_path, monkeypatch):
    from tools.prepare_kaggle_seed_repeat import seed_source
    frozen = (ROOT / 'tools/train_nested_proxy_control.py').read_text()
    with pytest.raises(ValueError, match='needs review'):
        seed_source(frozen.replace("spec['seed'], workers=True", "0, workers=True"), 1)
    design, inputs, spec, roles = miniature(tmp_path)
    cpu_training(monkeypatch)
    import torch
    torch.set_num_threads(1)
    contracts = {}
    for name, source in [('seed0', frozen), ('seed1', seed_source(frozen, 1))]:
        scope = {'__file__': str(ROOT / 'tools/train_nested_proxy_control.py'), '__name__': name}
        exec(source.split("if __name__ == '__main__':", 1)[0], scope)
        scope['run'](design, inputs, tmp_path / 'runs' / name, execute=True)
        contracts[name] = json.loads((tmp_path / 'runs' / name / 'run_contract.json').read_text())
    seed1 = contracts['seed1']
    assert seed1['seed'] == 1 and seed1['seed_repeat_of'] == 0 and seed1['arm'] == 'seed1'
    assert contracts['seed0']['seed'] == 0
    assert seed1['initial_weights_sha256'] != contracts['seed0']['initial_weights_sha256']
    assert seed1['training_recipe'] == contracts['seed0']['training_recipe']
    predictions = pd.read_parquet(tmp_path / 'runs/seed1/outer_predictions.parquet')
    validate_prediction_provenance(roles, seed1, predictions)


def test_kernel_fits_and_embeds_both_seed_runners(tmp_path):
    from tools.prepare_kaggle_seed_repeat import SEEDS, main
    out = main(destination=tmp_path / 'kernel')
    source = (out / 'seed_repeat.py').read_text()
    assert len(source.encode()) < 1_000_000
    compile(source, 'seed_repeat.py', 'exec')
    scope = {}
    exec(source.split('\nimport subprocess, sys\n', 1)[0], scope)
    assert {f'tools/train_seed{s}.py' for s in SEEDS} <= set(scope['EMBEDDED'])
    assert 'train_mirror.py' not in source and 'evaluate_nested_proxy' not in source.split('def assemble():', 1)[1]
    assert source.count('training_launched=True') == 1
