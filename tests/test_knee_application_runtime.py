import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import config
from src.application.runtime import register_model, run_batch
from src.model import artifacts, net


class Trunk(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))

    def forward(self, value):
        return self.layers(value)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CACHE_SIZE', 8)
    monkeypatch.setattr(config, 'CACHE_SLICES', 4)
    monkeypatch.setattr(net, 'build_backbone', lambda *a, **kw: Trunk())
    torch.manual_seed(47)
    model = net.KneeNet(Trunk(), 8).eval()
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    spec = dict(name='baseline', backbone='convnext_test', slices=3,
                max_series=2, series_selection='protocol')
    artifacts.write_metadata(checkpoint, spec, 0, ['train'], ['study'])
    (tmp_path / 'ensemble.json').write_text(json.dumps({'members': [
        {'checkpoint': 'model.pt', 'weight': 1}]}))
    rows = pd.DataFrame({'StudyInstanceUID': ['study', 'study'],
                         'SeriesInstanceUID': ['b', 'a'], 'cache_path': ['cache', 'cache'],
                         'position': [0, 1], 'laterality_resolved': ['L', 'L'],
                         'Fluid_Sensitive': [1, 0], 'Fat_Suppression': [1, 0],
                         'Anatomical_Plane': ['Sagittal', 'Coronal']})
    cols = ['SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']
    rows.drop(columns=cols[1:]).to_csv(tmp_path / 'manifest.csv', index=False)
    rows[cols].to_csv(tmp_path / 'series.csv', index=False)
    volumes = np.random.default_rng(31).integers(0, 256, (2, 4, 8, 8), dtype=np.uint8)
    np.save(tmp_path / 'cache.npy', volumes)
    settings = dict(schema_version=1, ensemble='ensemble.json', manifest='manifest.csv',
                    series_metadata='series.csv', sides=None, shards={'cache': 'cache.npy'})
    path = tmp_path / 'application.json'
    path.write_text(json.dumps(settings))
    return path, rows.assign(side_predicted='L'), volumes


def test_batch_matches_existing_prediction_and_records_artifacts(bundle, tmp_path):
    path, rows, volumes = bundle
    members = artifacts.load_ensemble(tmp_path / 'ensemble.json')
    expected = artifacts.predict_members(members, volumes, rows)[0]
    receipt = run_batch(path, ['study'], tmp_path / 'run')
    assert receipt['status'] == 'completed'
    prediction = pd.read_csv(tmp_path / 'run/predictions.csv')
    assert list(prediction.columns) == config.SUBMISSION_COLUMNS
    np.testing.assert_allclose(prediction[config.LABELS].iloc[0], expected, atol=1e-7)
    saved = json.loads((tmp_path / 'run/run.json').read_text())
    assert saved == receipt
    assert len(saved['input_sha256']) == 3
    assert 'predictions.csv' in saved['output_sha256']
    with pytest.raises(FileExistsError):
        run_batch(path, ['study'], tmp_path / 'run')


def test_missing_study_isolated_without_partial_prediction_csv(bundle, tmp_path):
    receipt = run_batch(bundle[0], ['missing', 'study'], tmp_path / 'run')
    assert receipt['failed'] == 1 and receipt['completed'] == 1
    assert not (tmp_path / 'run/predictions.csv').exists()
    responses = [json.loads(line) for line in (tmp_path / 'run/responses.jsonl').read_text().splitlines()]
    assert responses[0]['stage'] == 'prepare'
    assert responses[1]['status'] == 'completed'


def test_model_version_changes_with_recipe_and_rejects_tampering(bundle, tmp_path):
    manifest = tmp_path / 'ensemble.json'
    old = register_model(manifest)
    sidecar = tmp_path / 'model.pt.json'
    metadata = json.loads(sidecar.read_text())
    metadata['experiment']['slices'] = 2
    sidecar.write_text(json.dumps(metadata))
    new = register_model(manifest)
    assert old['model_version'] != new['model_version']
    assert old['preprocessing_id'] != new['preprocessing_id']
    with (tmp_path / 'model.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        run_batch(bundle[0], ['study'], tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_initialization_failure_recorded(bundle, tmp_path):
    (tmp_path / 'cache.npy').unlink()
    with pytest.raises(FileNotFoundError):
        run_batch(bundle[0], ['study'], tmp_path / 'run')
    receipt = json.loads((tmp_path / 'run/run.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['error_type'] == 'FileNotFoundError'


def test_negative_cache_position_rejected(bundle, tmp_path):
    manifest = pd.read_csv(tmp_path / 'manifest.csv')
    manifest.loc[0, 'position'] = -1
    manifest.to_csv(tmp_path / 'manifest.csv', index=False)
    receipt = run_batch(bundle[0], ['study'], tmp_path / 'run')
    assert receipt['failed'] == 1


def test_duplicate_studies_rejected_before_run(bundle, tmp_path):
    with pytest.raises(ValueError, match='duplicate study'):
        run_batch(bundle[0], ['study', 'study'], tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
