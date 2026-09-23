"""Stage and launch the reviewed-label fold-1 Kaggle pair."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_kaggle_training_pair import INPUT_SEARCH, ALL_VALIDATION_COUNTS

RUNNER = r'''
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

def main():
    root = Path('/kaggle/working')
    started = time.monotonic()
    progress = {'state': 'running', 'fold': FOLD, 'completed': [], 'labels_sha256': LABELS_SHA256}
    def save():
        progress['elapsed_seconds'] = time.monotonic() - started
        (root / 'pair_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    try:
        for arm, source in SCRIPTS.items():
            directory = root / f'{arm}_fold{FOLD}'
            directory.mkdir(exist_ok=False)
            script = directory / 'train.py'
            script.write_text(source)
            progress['active'] = arm; save()
            print(f'Starting {arm}, fold {FOLD}, 12 epochs', flush=True)
            subprocess.run([sys.executable, '-u', str(script)], cwd=directory,
                           env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True, timeout=18000)
            progress['completed'].append(arm); save()
        sys.path.insert(0, str(root / f'control_fold{FOLD}'))
        import numpy as np
        import pandas as pd
        from src import config
        from src.model import artifacts
        namespace = {'__name__': 'pair_comparison', '__file__': str(root / 'compare_oof.py')}
        exec(compile(COMPARE_SOURCE, str(root / 'compare_oof.py'), 'exec'), namespace)
        # Each arm writes its weights to its own output directory, which is not
        # the directory its script was run in. Locate every file by content.
        find = lambda arm, name: namespace['find_arm_file'](root, arm, name)
        a, b = [json.loads(find(arm, 'label_review_contract.json').read_text())
                for arm in ('control', 'candidate')]
        assert a['arm'] == 'control' and b['arm'] == 'candidate'
        assert a['correction_count'] == 0 and b['correction_count'] == CORRECTION_COUNT
        assert a['corrections_sha256'] == b['corrections_sha256'] == CORRECTIONS_SHA256
        assert a['original_labels_sha256'] == b['original_labels_sha256'] == LABELS_SHA256
        assert not a['validation_labels_changed'] and not b['validation_labels_changed']
        assert not a['folds_changed'] and not b['folds_changed']
        contracts = [json.loads(find(arm, 'run_contract.json').read_text())
                     for arm in ('control', 'candidate')]
        assert contracts[0] == contracts[1], 'Settings or initialization differ'
        meta = [artifacts.load_metadata(find(arm, f'convnext_fold{FOLD}.pt'))
                for arm in ('control', 'candidate')]
        for key in ('train_uids', 'val_uids', 'fold', 'labels'):
            assert meta[0][key] == meta[1][key], f'Provenance differs: {key}'
        folds = [pd.read_parquet(find(arm, 'folds.parquet'))
                 for arm in ('control', 'candidate')]
        pd.testing.assert_frame_equal(folds[0], folds[1])
        label_paths = attached_files('train_labels.parquet')
        assert len(label_paths) == 1 and hashlib.sha256(label_paths[0].read_bytes()).hexdigest() == LABELS_SHA256
        labels = pd.read_parquet(label_paths[0])
        frames = [namespace['read_predictions']([find(arm, f'oof_fold{FOLD}.parquet')], folds[0])
                  for arm in ('control', 'candidate')]
        expected = set(meta[0]['val_uids'])
        assert len(expected) == VALIDATION_STUDIES and all(set(frame.index) == expected for frame in frames)
        table, summary = namespace['compare'](labels, *frames, draws=1000, seed=0)
        assert np.isfinite(table.to_numpy()).all()
        delta = summary['macro']['candidate'] - summary['macro']['baseline']
        passed = delta >= .003 and bool((table.candidate_delta >= -.02).all())
        summary['screening_gate'] = {'passed': passed, 'macro_delta': delta,
            'required_macro_delta': .003, 'maximum_finding_drop': .02,
            'next_step': 'repeat on fold 3' if passed else 'review before more training'}
        summary['reviewed_label_contracts'] = [a, b]
        summary['checkpoint_sha256'] = [m['sha256'] for m in meta]
        table.to_csv(root / 'per_finding.csv', index_label='finding')
        (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(table.round(5).to_string(), flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        progress.update(state='complete', active=None, screening_passed=passed)
    except BaseException as exc:
        progress.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()

if __name__ == '__main__': main()
'''


def prepare(root=ROOT):
    root = Path(root)
    staged = root / 'outputs/reviewed_labels_v1'
    receipt = json.loads((staged / 'receipt.json').read_text())
    corrections = json.loads((staged / 'corrections.json').read_text())
    control, candidate = [(staged / f'{name}_fold1.py').read_text() for name in ('control', 'candidate')]
    if receipt['state'] != 'staged_not_trained' or len(corrections) != 11:
        raise ValueError('Reviewed-label candidate is not ready')
    if hashlib.sha256((root / 'data/meta/train_labels.parquet').read_bytes()).hexdigest() != receipt['original_labels_sha256']:
        raise ValueError('Original labels changed since review')
    comparison = (root / 'tools/compare_oof.py').read_text()
    source = (INPUT_SEARCH + f'\nFOLD = 1\nVALIDATION_STUDIES = {ALL_VALIDATION_COUNTS[1]}\n'
              f'LABELS_SHA256 = {receipt["original_labels_sha256"]!r}\n'
              f'CORRECTION_COUNT = {len(corrections)}\n'
              f'CORRECTIONS_SHA256 = {hashlib.sha256(json.dumps(corrections, sort_keys=True).encode()).hexdigest()!r}\n'
              f'SCRIPTS = {{"control": {control!r}, "candidate": {candidate!r}}}\n'
              f'COMPARE_SOURCE = {comparison!r}\n' + RUNNER)
    ast.parse(source)
    output = root / 'outputs/kaggle_reviewed_labels_fold1'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'train.py').write_text(source)
    previous = json.loads((root / 'outputs/kaggle_sources/knee-finetune/kernel-metadata.json').read_text())
    metadata = {key: previous[key] for key in ('competition_sources', 'kernel_sources', 'dataset_sources', 'docker_image')}
    metadata.update(id='pssv27/knee-reviewed-labels-fold1-pair', title='Knee Reviewed Labels Fold1 Pair',
                    code_file='train.py', language='python', kernel_type='script', is_private=True,
                    enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    launch = {'kernel': metadata['id'], 'state': 'staged_not_launched', 'fold': 1,
              'correction_count': len(corrections), 'labels_sha256': receipt['original_labels_sha256'],
              'candidate_labels_sha256': receipt['candidate_labels_sha256'],
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'runtime': 'two sequential 12-epoch T4 runs; prior 224-pixel T4 protocol jobs are the appropriate estimate'}
    (output / 'staging_receipt.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(launch, indent=2))
    return output


if __name__ == '__main__': prepare()
