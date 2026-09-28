"""Resolution cache safety, raw provenance and deterministic study partitions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data import cache, resolution


def series_frame():
    return pd.DataFrame({'StudyInstanceUID': ['a', 'a', 'a', 'b', 'b', 'c', 'd'],
                         'SeriesInstanceUID': list('ABCDEFG')})


def test_balanced_partition_is_study_grouped_and_order_invariant():
    frame = series_frame()
    a = resolution.shard_plan(frame, shards=2)
    pd.testing.assert_frame_equal(a, resolution.shard_plan(frame.sample(frac=1), shards=2))
    assert a.groupby('StudyInstanceUID').shard.nunique().eq(1).all()
    assert sorted(a.groupby('shard').size()) == [3, 4]
    with pytest.raises(ValueError, match='budget'):
        resolution.shard_plan(frame, shards=2, budget_bytes=1)
    with pytest.raises(ValueError, match='duplicate'):
        resolution.shard_plan(pd.concat([frame, frame]))


def test_geometry_audit_requires_exact_coverage():
    frame = series_frame()
    index = frame.assign(Rows=800, Columns=800, path=list('abcdefg'))
    report = resolution.audit_index(index, frame)
    assert report['series_both_dimensions_at_least_384'] == 7
    with pytest.raises(ValueError, match='exactly'):
        resolution.audit_index(index.iloc[:-1], frame)
    with pytest.raises(ValueError, match='dimensions'):
        resolution.audit_index(index.assign(Rows=np.nan), frame)


def test_explicit_resolution_uses_raw_data_without_changing_defaults(monkeypatch):
    raw = np.arange(24 * 80 * 80, dtype=np.float32).reshape(24, 80, 80)
    monkeypatch.setattr(cache.dicom_io, 'load_series_volume', lambda paths: raw)
    headers = [{'path': 'fixture', 'Rows': 80, 'Columns': 80, 'InstanceNumber': i}
               for i in range(24)]
    old = cache.build_series_volume(headers)['volume']
    new = cache.build_series_volume(headers, size=384)['volume']
    np.testing.assert_array_equal(new, cache.normalize_intensity(cache.resize_volume(raw, 384)))
    assert old.shape == (24, 224, 224) and new.shape == (24, 384, 384)
    assert config.CACHE_SIZE == 224
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='size'):
            cache.build_series_volume(headers, size=bad)


def test_streaming_shard_roundtrip_and_no_overwrite(tmp_path, monkeypatch):
    frame = series_frame().iloc[:3]
    index = frame.assign(Rows=800, Columns=800, path=['a.dcm', 'b.dcm', 'c.dcm'])
    monkeypatch.setattr(resolution, 'relocate_headers', lambda rows, root: rows)
    def fake_build(headers, size):
        return {'volume': np.full((24, size, size), 123, np.uint8),
                'StudyInstanceUID': headers[0]['StudyInstanceUID'],
                'SeriesInstanceUID': headers[0]['SeriesInstanceUID'], 'laterality': 'R'}
    monkeypatch.setattr(cache, 'build_series_volume', fake_build)
    receipt = resolution.build_shard(index, frame, tmp_path, tmp_path, 0, size=8, shards=1)
    assert receipt['state'] == 'complete' and receipt['completed'] == 3
    assert np.load(tmp_path / 'cache_train_series_00.npy').shape == (3, 24, 8, 8)
    manifest = pd.read_parquet(tmp_path / 'manifest_train_series_00.parquet')
    assert manifest.position.tolist() == [0, 1, 2]
    assert manifest.laterality_resolved.eq('R').all()
    with pytest.raises(FileExistsError):
        resolution.build_shard(index, frame, tmp_path, tmp_path, 0, size=8, shards=1)


def test_failed_shard_is_not_marked_complete(tmp_path, monkeypatch):
    frame = series_frame().iloc[:3]
    index = frame.assign(Rows=800, Columns=800, path=['a', 'b', 'c'])
    with pytest.raises(FileNotFoundError):
        resolution.build_shard(index, frame, tmp_path, tmp_path, 0, size=8, shards=1)
    import json
    assert json.loads((tmp_path / 'resolution_receipt_00.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('size', [224, 384])
def test_staged_training_isolates_geometry_and_preserves_embedded_modules(size):
    import ast
    import json
    from tools.prepare_resolution_training import resolution_source, ROOT
    notebook = json.loads((ROOT / 'notebooks/08_finetune.ipynb').read_text())
    source = resolution_source(notebook, 'fixture_hash', size, smoke=True)
    tree = ast.parse(source)
    assert f'config.CACHE_SIZE = {size}' in source
    assert f"name='resolution{size}'" in source
    assert f'volume.shape[1:] == (24, {size}, {size})' in source
    assert f'len(CACHE_PATHS) == {6 if size == 384 else 3}' in source
    assert source.index(f'config.CACHE_SIZE = {size}') < source.index('index = knee_dataset.StudyIndex')
    assert source.index('module.model.backbone.set_grad_checkpointing(True)') < source.index("_contract =")
    assert 'series_selection=\'protocol\'' in source
    assert 'max_epochs=1' in source and 'limit_train_batches=8' in source
    embedded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    for path, content in embedded.items():
        assert content == (ROOT / path).read_text()
    if size == 384:
        assert source.index('validate_cache_inputs(cache_paths') < source.index('manifest = pd.concat')


def test_resolution_comparison_rejects_unmatched_training():
    from tools.compare_resolution import validate_contracts
    from src.model.experiments import get_experiment
    spec = get_experiment('protocol').to_dict()
    contracts = [dict(cache_size=s, cache_slices=24, gradient_checkpointing=True,
                      epochs=12, effective_batch=8, initial_weights_sha256='same',
                      labels_sha256='labels', experiment={**spec, 'name': f'resolution{s}'})
                 for s in (224, 384)]
    validate_contracts(contracts)
    contracts[1]['initial_weights_sha256'] = 'different'
    with pytest.raises(ValueError, match='initial weights'):
        validate_contracts(contracts)


def test_input_receipt_rejects_manifest_corruption(tmp_path, monkeypatch):
    frame = series_frame().iloc[:3]
    index = frame.assign(Rows=800, Columns=800, path=['a', 'b', 'c'])
    monkeypatch.setattr(resolution, 'relocate_headers', lambda rows, root: rows)
    def fake_build(headers, size):
        return {'volume': np.full((24, size, size), 123, np.uint8),
                'StudyInstanceUID': headers[0]['StudyInstanceUID'],
                'SeriesInstanceUID': headers[0]['SeriesInstanceUID'], 'laterality': 'R'}
    monkeypatch.setattr(cache, 'build_series_volume', fake_build)
    resolution.build_shard(index, frame, tmp_path, tmp_path, 0, size=8, shards=1)
    paths = [tmp_path / 'cache_train_series_00.npy']
    resolution.validate_cache_inputs(paths, expected_size=8, expected_shards=1)
    (tmp_path / 'manifest_train_series_00.parquet').write_bytes(b'bad manifest')
    with pytest.raises(ValueError, match='hash mismatch'):
        resolution.validate_cache_inputs(paths, expected_size=8, expected_shards=1)
