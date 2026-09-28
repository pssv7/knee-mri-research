import ast
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from tools.compare_incumbent_addition import fixed_mix, validated_family
from tools.compare_protocol_incumbent import fixed_predictions
from tools.prepare_campaign_completion import prepare
from tools.prepare_kaggle_training_pair import ROOT


def frame(value):
    return pd.DataFrame({'fold': [1, 3], **{label: [value, 1 - value] for label in config.LABELS}},
                        index=pd.Index(['a', 'b'], name=config.ID_COL))


def test_historical_equal_thirds_and_mandatory_half_are_distinct():
    families = [frame(.1), frame(.3), frame(.9)]
    incumbent, protocol = fixed_predictions(families)
    pd.testing.assert_frame_equal(protocol, fixed_mix([incumbent, families[2]], [2, 1]))
    half = fixed_mix([incumbent, families[2]], [1, 1])
    assert half.iloc[0][config.LABELS[0]] == pytest.approx(.55)
    assert protocol.iloc[0][config.LABELS[0]] == pytest.approx(1.3 / 3)
    combined = fixed_mix([incumbent, families[1], families[2]], [2, 1, 1])
    assert combined.iloc[0][config.LABELS[0]] == pytest.approx(.4)


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'fold', 'nan', 'range'])
def test_recipe_rejects_invalid_or_unmatched_predictions(bad):
    a, b = frame(.1), frame(.9)
    if bad == 'missing': b = b.iloc[:1]
    if bad == 'duplicate': b = pd.concat([b, b.iloc[:1]])
    if bad == 'fold': b['fold'] = 0
    if bad == 'nan': b.loc['a', config.LABELS[0]] = np.nan
    if bad == 'range': b.loc['a', config.LABELS[0]] = 1.1
    with pytest.raises(ValueError): fixed_mix([a, b], [1, 1])


@pytest.mark.parametrize('family', ['spatial', 'resolution384'])
@pytest.mark.parametrize('fold,seed', [(0, 0), (2, 0), (4, 0), (1, 1)])
def test_completion_preserves_split_and_separates_training_seeds(tmp_path, family, fold, seed):
    for path in ('notebooks', 'data'):
        (tmp_path / path).symlink_to(ROOT / path, target_is_directory=True)
    out = prepare(family, fold, seed, tmp_path)
    receipt = json.loads((out / 'staging_receipt.json').read_text())
    ast.parse((out / 'run_candidate.py').read_text())
    assert receipt['seed'] == seed and receipt['split_seed'] == 0
    assert f'_seed{seed}' in receipt['identity']
    source = (out / 'full.py').read_text()
    tree = ast.parse(source)
    assert f'EXPECTED_SEED = {seed}' in source and f'SEED = {seed}' in source
    assert f'FOLD = {fold}' in source
    split = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'folded')
    assert ast.unparse(split) == 'splits.make_folds(labels, groups=groups)'
    assert f'{family}_fold{fold}_seed{seed}/full' in source


def test_provenance_rejects_checkpoint_tampering_and_train_overlap(tmp_path):
    from src.model.artifacts import write_metadata
    folded = pd.DataFrame({config.ID_COL: ['train', 'val'], 'fold': [0, 1]})
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'fixture')
    write_metadata(checkpoint, {}, 1, ['train'], ['val'])
    family = {'name': 'fixture', 'runs': [{'fold': 1, 'checkpoint': 'model.pt', 'oof': 'oof.parquet',
                                        'legacy_reconstructed': True}]}
    sidecar = tmp_path / 'model.pt.json'
    meta = json.loads(sidecar.read_text())
    meta['train_uids'] = ['val']
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='membership'):
        validated_family(family, tmp_path, folded, [1], False)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        validated_family(family, tmp_path, folded, [1], False)


def test_matched_colab_has_both_resolutions_and_only_one_screening_fold(tmp_path):
    from tools.prepare_colab_resolution_pair import build
    (tmp_path / 'notebooks').mkdir()
    (tmp_path / 'notebooks/08_finetune.ipynb').symlink_to(ROOT / 'notebooks/08_finetune.ipynb')
    for path in ('data', 'tools', 'src'):
        (tmp_path / path).symlink_to(ROOT / path, target_is_directory=True)
    notebook = json.loads(build(tmp_path).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse('\n'.join(line for line in cell['source'].splitlines() if not line.startswith('%')))
    embedded = next(c['source'] for c in notebook['cells'] if 'SOURCES = ' in c['source'])
    assignments = {n.targets[0].id: n.value for n in ast.parse(embedded).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    sources = ast.literal_eval(assignments['SOURCES'])
    assert set(sources) == {224, 384}
    assert ast.literal_eval(assignments['FOLD']) == 1
    for size, phases in sources.items():
        assert set(phases) == {'smoke', 'full'}
        assert f'config.CACHE_SIZE = {size}' in phases['full']
        assert 'EXPECTED_SLICES = 16' in phases['full']
