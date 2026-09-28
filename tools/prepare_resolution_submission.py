"""Import completed 384 folds and stage a standalone five-fold submission locally."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.model import artifacts
from tools.prepare_protocol_benchmark import INPUT_SEARCH

ARCHIVES = {
    0: 'fold0-20260922T163225Z-1-001.zip',
    1: '384-20260917T051143Z-1-001.zip',
    2: 'fold2-20260923T021427Z-1-001.zip',
    3: 'fold3-20260923T021430Z-1-001.zip',
    4: 'fold4-20260923T021433Z-1-001.zip',
}
DATASET = 'pssv27/knee-resolution384-five-fold-private'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def stage_notebook(package, output):
    notebook = json.loads((ROOT / 'notebooks/10_submit_ensemble.ipynb').read_text())
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(sources) == 5
    # Refresh embedded modules without modifying the incumbent notebook.
    tree = ast.parse(sources[0])
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in node.targets):
            files = ast.literal_eval(node.value)
            node.value = ast.parse(repr({p: (ROOT / p).read_text() for p in files}), mode='eval').body
    sources[0] = ast.unparse(ast.fix_missing_locations(tree)) + '\nconfig.CACHE_SIZE = 384\n'
    setup = ast.parse('\n'.join(line for line in sources[1].splitlines() if not line.startswith('!')))
    for i, node in enumerate(setup.body):
        if isinstance(node, ast.FunctionDef) and node.name == 'find_inputs':
            setup.body[i] = ast.parse('def find_inputs(pattern):\n    return [str(p) for p in attached_files(pattern)]').body[0]
    sources[1] = ast.unparse(setup) + '\nassert device == "cuda", "GPU required"\ntorch.set_num_threads(2)\n'
    manifest = json.loads((package / 'ensemble.json').read_text())
    expected = {str(i): json.loads((package / f'convnext_fold{i}.pt.json').read_text())['sha256'] for i in range(5)}
    header = 'import hashlib, json, os, pathlib, time\nSUBMISSION_STARTED = time.monotonic()\n' + INPUT_SEARCH
    header += f'''
manifests = attached_files("ensemble.json")
assert len(manifests) == 1, "Attach only the 384 five-fold model dataset"
assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == {artifacts.sha256(package / 'ensemble.json')!r}
os.environ['KNEE_ENSEMBLE_JSON'] = str(manifests[0])
EXPECTED_CHECKPOINTS = {expected!r}
'''
    sources[2] += '''
assert len(explicit_members) == len(tuned_models) == 5
assert config.CACHE_SIZE == 384 and config.CACHE_SLICES == 24
assert BLEND == 1.0 and not FLIP_TTA and not SOUP and not MEMBER_RANK_AVERAGE
assert np.allclose(member_weights, [.2] * 5)
actual_checkpoints = {str(m['metadata']['fold']): m['metadata']['sha256'] for m in explicit_members}
assert actual_checkpoints == EXPECTED_CHECKPOINTS
'''
    # SideNet must see raw -> 224, not raw -> 384 -> 224 (different interpolation).
    sources[3] = replace_once(sources[3], 'test_tag_sides = {}', 'test_tag_sides = {}\nside_volumes_224 = {}')
    sources[3] = replace_once(sources[3], '    directory = test_root / study_uid', '    side_volumes_224.clear()\n    directory = test_root / study_uid')
    sources[3] = replace_once(sources[3], '        result = cache.build_series_volume(headers)', '        result = cache.build_series_volume(headers, size=384)\n        side_volumes_224[series_dir.name] = cache.build_series_volume(headers, size=224)["volume"]')
    sources[3] = replace_once(sources[3], 'side_classifier.prepare_batch(v)[0] for v in volumes', 'side_classifier.prepare_batch(side_volumes_224[uid])[0] for uid in uids')
    validation = '''
assert submission[config.ID_COL].tolist() == test_csv[config.ID_COL].tolist()
assert submission.columns.tolist() == config.SUBMISSION_COLUMNS
values = submission[config.LABELS].to_numpy()
assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
assert failures == 0, f"Refusing submission with {failures} fallback studies"
report = {'rows': len(submission), 'members': 5, 'cache_size': 384, 'side_cache_size': 224,
          'fallbacks': failures, 'checkpoints': actual_checkpoints, 'weights': list(map(float, member_weights)),
          'elapsed_seconds': time.monotonic() - SUBMISSION_STARTED,
          'gpu': torch.cuda.get_device_name(),
          'submission_sha256': hashlib.sha256(pathlib.Path('/kaggle/working/submission.csv').read_bytes()).hexdigest()}
pathlib.Path('/kaggle/working/submission_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''
    code = [header, *sources, validation]
    for source in code:
        ast.parse(source)
    notebook['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# 384-pixel five-fold submission\nEqual selected checkpoints, 0.2 per fold. Standalone 384 ensemble. GPU runtime validation pending.\n']}]
    notebook['cells'] += [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(keepends=True)} for s in code]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'submission.ipynb', notebook)
    metadata = json.loads((ROOT / 'outputs/kaggle_protocol_submission/kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-resolution384-five-fold', title='Knee Resolution384 Five Fold', dataset_sources=[DATASET, 'pssv27/knee-labels'])
    write_json(output / 'kernel-metadata.json', metadata)


def main():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    package = ROOT / 'outputs/ensemble_resolution384'
    imported = ROOT / 'outputs/resolution384_downloads'
    package.mkdir(parents=True, exist_ok=True)
    contracts, inputs, records, predictions = [], [], [], []
    config.CACHE_SIZE = 384
    for fold, filename in ARCHIVES.items():
        archive = Path.home() / 'Downloads' / filename
        with zipfile.ZipFile(archive) as z:
            progress_name = next(n for n in z.namelist() if n.endswith('/progress.json'))
            progress = json.loads(z.read(progress_name))
            assert progress['state'] == 'complete' and progress['fold'] == fold
            attempt = Path(progress_name).parent
            prefix = str(attempt / 'full' / f'resolution384_fold{fold}') + '/'
            receipt = json.loads(z.read(str(attempt / 'full/resolution_run.json')))
            assert receipt['state'] == 'complete' and not receipt['smoke'] and receipt['size'] == 384
            destination = imported / f'fold{fold}'
            destination.mkdir(parents=True, exist_ok=True)
            for entry in z.infolist():
                name = entry.filename
                relative = Path(name).relative_to(attempt)
                # Keep selected weights and small audit material; large alternative exports stay in ZIP.
                selected = name == prefix + f'convnext_fold{fold}.pt'
                small = entry.file_size < 5_000_000 and '__pycache__' not in name
                if not (selected or small):
                    continue
                assert '..' not in relative.parts
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with z.open(entry) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
            run = destination / 'full' / f'resolution384_fold{fold}'
            for name, checksum in progress['sha256'].items():
                path = run / name
                if path.exists():
                    assert artifacts.sha256(path) == checksum, str(path)
                else:
                    assert name.endswith('.pt'), name
            checkpoint = run / f'convnext_fold{fold}.pt'
            meta = artifacts.load_metadata(checkpoint)
            assert meta['fold'] == fold
            contract = json.loads((run / 'run_contract.json').read_text())
            contracts.append({k: v for k, v in contract.items() if k not in ('fold', 'steps', 'warmup_steps')})
            input_receipt = json.loads((destination / 'input_receipt.json').read_text())
            inputs.append({k: input_receipt[k] for k in ('metadata_sha256', 'side_predictions_sha256', 'shards')})
            frame = pd.read_parquet(run / f'oof_fold{fold}.parquet')
            assert not frame[config.ID_COL].duplicated().any()
            assert set(frame[config.ID_COL]) == set(meta['val_uids'])
            assert not set(meta['train_uids']) & set(meta['val_uids'])
            splits = pd.read_parquet(run / 'folds.parquet')
            assert set(splits.loc[splits.fold == fold, config.ID_COL]) == set(meta['val_uids'])
            assert set(splits.loc[splits.fold != fold, config.ID_COL]) == set(meta['train_uids'])
            assert len(splits) == 4349
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            del state
            for suffix in ('', '.json'):
                target = package / (checkpoint.name + suffix)
                source = Path(str(checkpoint) + suffix)
                if not target.exists():
                    target.hardlink_to(source)
                assert artifacts.sha256(target) == artifacts.sha256(source)
            predictions.append(frame.assign(fold=fold))
            records.append({'fold': fold, 'archive': str(archive), 'attempt': str(attempt), 'checkpoint_sha256': meta['sha256'], 'oof_rows': len(frame), 'elapsed_training_seconds': receipt['elapsed_seconds']})
            print(f'Fold {fold}: verified and imported {len(frame)} OOF studies', flush=True)
    assert all(c == contracts[1] for c in contracts)
    assert all(i == inputs[1] for i in inputs)
    oof = pd.concat(predictions, ignore_index=True)
    assert len(oof) == 4349 and not oof[config.ID_COL].duplicated().any()
    values = oof[config.LABELS].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    labels_path = ROOT / 'data/meta/train_labels.parquet'
    assert artifacts.sha256(labels_path) == contracts[1]['labels_sha256']
    labels = pd.read_parquet(labels_path).set_index(config.ID_COL).loc[oof[config.ID_COL], config.LABELS]
    scores = {n: roc_auc_score(labels[n].to_numpy() > .5, oof[n]) for n in config.LABELS}
    fold_scores = {str(f): float(np.mean([
        roc_auc_score(labels[n].to_numpy()[oof.fold == f] > .5, oof.loc[oof.fold == f, n])
        for n in config.LABELS])) for f in range(5)}
    write_json(imported / 'oof_metrics.json', {
        'pooled_mined_label_macro': float(np.mean(list(scores.values()))),
        'per_finding': scores, 'per_fold_macro': fold_scores, 'studies': len(oof),
        'note': 'OOF proxy against mined labels; not leaderboard performance. No weight tuning.'})
    write_json(package / 'ensemble.json', {'members': [{'checkpoint': f'convnext_fold{i}.pt', 'weight': .2} for i in range(5)]})
    write_json(package / 'dataset-metadata.json', {'id': DATASET, 'title': 'Knee Resolution384 Five Fold Private', 'licenses': [{'name': 'other'}]})
    write_json(imported / 'import_receipt.json', {'members': records, 'contract': contracts[1], 'recipe': 'five validation-selected checkpoints; equal probability weights; no gold selection', 'excluded_incomplete_archive': 'fold3-20260922T163223Z-1-001.zip', 'retained_in_archives': 'EMA/SWA weights, Lightning checkpoints, bytecode', 'gpu_validation': 'pending'})
    oof.to_parquet(imported / 'oof_all_folds.parquet', index=False)
    stage_notebook(package, ROOT / 'outputs/kaggle_resolution384_submission')
    print('Staged locally; no upload, GPU run, or competition submission.')


if __name__ == '__main__':
    main()
