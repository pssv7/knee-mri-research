"""Exercise resumable downloads, Drive staging, and actual Colab source relocation."""

import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.colab_resolution_runtime import download_stream, copy_from_drive, unpack_metadata
from tools.prepare_colab_resolution import build


class Response:
    def __init__(self, content, status=200, headers=None, interrupt=False):
        self.content, self.status_code = content, status
        self.headers = headers or {'Content-Length': str(len(content))}
        self.interrupt = interrupt
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size):
        yield self.content
        if self.interrupt:
            raise requests.ConnectionError('simulated disconnection')


def test_interrupted_download_resumes_range_and_is_hashed(tmp_path, monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs['headers'])
        if len(calls) == 1:
            return Response(b'abc', headers={'Content-Length': '6'}, interrupt=True)
        return Response(b'def', status=206, headers={'Content-Range': 'bytes 3-5/6'})
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr('tools.colab_resolution_runtime.time.sleep', lambda _: None)
    target = tmp_path / 'cache.npy'
    download_stream(lambda: 'https://fixture.invalid', target, hashlib.sha256(b'abcdef').hexdigest())
    assert target.read_bytes() == b'abcdef'
    assert calls[1]['Range'] == 'bytes=3-'
    assert not target.with_name('cache.npy.partial').exists()
    download_stream(lambda: pytest.fail('must reuse verified file'), target, hashlib.sha256(b'abcdef').hexdigest())


def test_range_ignored_restarts_instead_of_appending(tmp_path, monkeypatch):
    target = tmp_path / 'cache.npy'
    target.with_name('cache.npy.partial').write_bytes(b'abc')
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response(b'abcdef'))
    download_stream(lambda: 'fixture', target, hashlib.sha256(b'abcdef').hexdigest())
    assert target.read_bytes() == b'abcdef'


def test_wrong_range_cannot_corrupt_partial(tmp_path, monkeypatch):
    target = tmp_path / 'cache.npy'
    partial = target.with_name('cache.npy.partial')
    partial.write_bytes(b'abc')
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response(b'def', 206, {'Content-Range': 'bytes 2-4/6'}))
    with pytest.raises(RuntimeError, match='incorrect byte range'):
        download_stream(lambda: 'fixture', target, 'hash')
    assert partial.read_bytes() == b'abc' and not target.exists()


def test_private_output_auth_failure_is_actionable(tmp_path, monkeypatch):
    response = Response(b'', status=403)
    def denied(*args, **kwargs):
        error = requests.HTTPError('forbidden')
        error.response = response
        raise error
    with pytest.raises(RuntimeError, match='denied.*HTTP 403'):
        download_stream(lambda: 'fixture', tmp_path / 'cache.npy', request_get=denied)


def test_hub_file_replaces_interrupted_drive_file(tmp_path, monkeypatch):
    import types
    from tools.colab_resolution_runtime import _hub_file
    good = b'complete shard'
    stale = tmp_path / 'cache_train_series_00.npy'
    stale.write_bytes(b'partial')
    calls = []
    def notebook_output_download(handle, path, output_dir, force_download=False):
        calls.append(force_download)
        target = Path(output_dir) / path
        if target.exists() and not force_download:
            raise FileExistsError(f'File already exists at output_dir: {target}')
        target.write_bytes(good)
        return str(target)
    monkeypatch.setitem(sys.modules, 'kagglehub', types.SimpleNamespace(
        notebook_output_download=notebook_output_download))
    result = _hub_file('owner/kernel/versions/1', stale.name, tmp_path,
                       hashlib.sha256(good).hexdigest())
    assert calls == [True] and result.read_bytes() == good
    _hub_file('owner/kernel/versions/1', stale.name, tmp_path, hashlib.sha256(good).hexdigest())
    assert calls == [True], 'a verified Drive file must be reused without downloading'


def test_drive_copy_reuses_valid_local_and_rejects_damaged_source(tmp_path):
    source, destination = tmp_path / 'drive.bin', tmp_path / 'local/input.bin'
    source.write_bytes(b'correct')
    checksum = hashlib.sha256(b'correct').hexdigest()
    copy_from_drive(source, destination, checksum)
    timestamp = destination.stat().st_mtime_ns
    copy_from_drive(source, destination, checksum)
    assert destination.stat().st_mtime_ns == timestamp
    destination.write_bytes(b'corrupt')
    copy_from_drive(source, destination, checksum)
    assert destination.read_bytes() == b'correct'
    source.write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='checksum'):
        copy_from_drive(source, tmp_path / 'other.bin', checksum)
    assert not (tmp_path / 'other.bin').exists()


def test_metadata_extracts_named_member_without_path_traversal(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path / 'download.zip', 'w') as archive:
        archive.writestr('../../train.csv', 'safe')
        archive.writestr('../unrelated.txt', 'ignore')
    assert unpack_metadata(tmp_path, 'train.csv').read_text() == 'safe'
    assert not (tmp_path.parent / 'unrelated.txt').exists()


def test_local_staging_copies_receipted_drive_files_without_network(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from tools.colab_resolution_runtime import copy_inputs_to_colab, digest
    from src.data import resolution
    drive, local = tmp_path / 'drive', tmp_path / 'local'
    record = {'state': 'verified', 'shards': [], 'metadata_sha256': {}}
    for shard in range(6):
        directory = drive / 'cache' / f'shard{shard}'
        directory.mkdir(parents=True)
        array = directory / f'cache_train_series_{shard:02d}.npy'
        np.save(array, np.ones((1, 24, 4, 4), dtype=np.uint8))
        manifest = directory / f'manifest_train_series_{shard:02d}.parquet'
        pd.DataFrame({'SeriesInstanceUID': [str(shard)]}).to_parquet(manifest)
        receipt = {'shard': shard, 'volume_sha256': digest(array), 'manifest_sha256': digest(manifest)}
        record['shards'].append(receipt)
        (directory / f'resolution_receipt_{shard:02d}.json').write_text(json.dumps(receipt))
    for folder, name in [('metadata', 'train.csv'), ('metadata', 'train_series.csv'), ('auxiliary', 'train_labels.parquet'), ('auxiliary', 'side_predictions.parquet')]:
        path = drive / folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        if name.startswith('side_'):
            record['side_predictions_sha256'] = digest(path)
        else:
            record['metadata_sha256'][name] = digest(path)
    (drive / 'input_receipt.json').write_text(json.dumps(record))
    checked = []
    monkeypatch.setattr(resolution, 'validate_cache_inputs', lambda paths: checked.extend(paths))
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: pytest.fail('copy step must not contact Kaggle'))
    monkeypatch.setattr('tools.colab_resolution_runtime.shutil.disk_usage',
                        lambda _: type('Disk', (), {'free': 20 * 2**30})())
    copy_inputs_to_colab(drive, local)
    assert len(checked) == 6
    assert json.loads((local / 'input_receipt.json').read_text()) == record
    assert (local / 'auxiliary/side_predictions.parquet').read_bytes() == b'side_predictions.parquet'


def test_adoption_requires_original_receipts_and_never_fabricates_provenance(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from tools.colab_resolution_runtime import adopt_drive_inputs, digest
    from src.data import resolution
    root = tmp_path / 'drive'
    for shard in range(6):
        folder = root / f'raw_output_{shard}'
        folder.mkdir(parents=True)
        np.save(folder / f'cache_train_series_{shard:02d}.npy', np.ones((1, 24, 384, 384), np.uint8))
        pd.DataFrame({'StudyInstanceUID': [str(shard)], 'SeriesInstanceUID': [str(shard)]}).to_parquet(folder / f'manifest_train_series_{shard:02d}.parquet')
    for folder, name in [('metadata', 'train.csv'), ('metadata', 'train_series.csv'), ('auxiliary', 'train_labels.parquet'), ('auxiliary', 'side_predictions.parquet')]:
        path = root / folder / name; path.parent.mkdir(exist_ok=True); path.write_bytes(name.encode())
    hashes = {name: digest(root / folder / name) for folder, name in [('metadata', 'train.csv'), ('metadata', 'train_series.csv'), ('auxiliary', 'train_labels.parquet')]}
    monkeypatch.setattr(resolution, 'validate_cache_inputs', lambda paths: None)
    with pytest.raises(RuntimeError, match='original resolution_receipt'):
        adopt_drive_inputs(root, hashes)
    assert not (root / 'input_receipt.json').exists()
    assert not (root / 'cache').exists()


def test_adoption_verifies_original_hashes_and_official_coverage_without_links(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from tools.colab_resolution_runtime import adopt_drive_inputs, digest
    from src.data import resolution
    root = tmp_path / 'drive'
    official = pd.DataFrame({'StudyInstanceUID': [f's{i}' for i in range(24371)],
                             'SeriesInstanceUID': [f'r{i}' for i in range(24371)]})
    for shard in range(6):
        directory = root / f'original_{shard}'
        directory.mkdir(parents=True)
        array = directory / f'cache_train_series_{shard:02d}.npy'
        # Geometry validation has separate real memmap tests; this exercises
        # receipt/hash/coverage orchestration without constructing 80 GiB.
        np.save(array, np.ones((1, 24, 8, 8), np.uint8))
        manifest = directory / f'manifest_train_series_{shard:02d}.parquet'
        official.iloc[shard::6].to_parquet(manifest)
        receipt = {'shard': shard, 'plan_sha256': 'original-plan',
                   'volume_sha256': digest(array), 'manifest_sha256': digest(manifest)}
        (directory / f'resolution_receipt_{shard:02d}.json').write_text(json.dumps(receipt))
    official.to_csv(root / 'train_series.csv', index=False)
    official.to_parquet(root / 'side_predictions.parquet')
    (root / 'train.csv').write_bytes(b'train')
    (root / 'train_labels.parquet').write_bytes(b'labels')
    hashes = {name: digest(root / name) for name in ('train.csv', 'train_series.csv', 'train_labels.parquet')}
    monkeypatch.setattr(resolution, 'validate_cache_inputs', lambda paths: None)
    record = adopt_drive_inputs(root, hashes)
    assert record['state'] == 'verified' and len(record['files']) == 22
    assert not (root / 'cache').exists(), 'Drive adoption must not create links or copies'
    (root / 'original_0/cache_train_series_00.npy').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        adopt_drive_inputs(root, hashes)


def test_notebook_is_self_contained_and_drive_first(tmp_path, monkeypatch):
    import nbformat
    from tools.colab_protocol_runtime import relocate
    from tools.colab_slice_runtime import relocate_inputs
    path = build()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [c.source for c in notebook.cells if c.cell_type == 'code']
    for cell in code:
        ast.parse('\n'.join(line for line in cell.splitlines() if not line.startswith('%')))
    joined = '\n'.join(code)
    assert joined.index('DRIVE_RECEIPT = adopt_drive_inputs') < joined.index('LOCAL_RECEIPT = copy_inputs_to_colab') < joined.index('RUN_DIRS = run_resolution_folds')
    runtime = next(c for c in code if c.startswith('"""Resumable'))
    assert 'from tools.' not in runtime
    namespace = {}
    exec(runtime, namespace)
    values = {}
    exec(next(c for c in code if c.startswith('TRAINING_SOURCES_BY_FOLD =')), values)
    assert set(values['TRAINING_SOURCES_BY_FOLD']) == {1, 3}
    for fold_sources in values['TRAINING_SOURCES_BY_FOLD'].values():
        for arm, source in fold_sources.items():
            moved = relocate_inputs(relocate(source, tmp_path / arm), tmp_path / 'inputs')
            tree = ast.parse(moved)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in node.targets):
                    for name, content in ast.literal_eval(node.value).items():
                        assert content == (ROOT / name).read_text()
            assert f"config.CACHE_SIZE = 384" in moved
            assert 'set_grad_checkpointing(True)' in moved
            if arm == 'full':
                assert 'max_epochs=EPOCHS' in moved
            funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('find_inputs', 'find_competition_file')]
            scope = {'os': __import__('os'), 'glob': __import__('glob')}
            monkeypatch.chdir(tmp_path)
            for folder, filename in [('cache/shard0', 'cache_train_series_00.npy'), ('metadata', 'train.csv'), ('auxiliary', 'train_labels.parquet')]:
                dest = tmp_path / 'inputs' / folder / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.touch()
            exec(compile(ast.Module(body=funcs, type_ignores=[]), '<finders>', 'exec'), scope)
            assert len(scope['find_inputs']('cache_train_series_*.npy')) == 1
            assert len(scope['find_inputs']('train_labels.parquet')) == 1
            assert Path(scope['find_competition_file']('train.csv')).is_file()
