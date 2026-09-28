"""Stage candidate-only primary completion or a separate seed-1 repeat; never launch."""

import argparse
import hashlib
import json
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_kaggle_training_pair import training_source
from tools.prepare_resolution_training import resolution_source
from tools.colab_protocol_runtime import relocate, ColabPaths


def prepare(family, fold, seed=0, root=ROOT):
    if family not in ('spatial', 'resolution384'):
        raise ValueError('Only retained campaign candidates can be expanded')
    if (seed == 1 and fold != 1) or (seed == 0 and fold not in (0, 2, 4)) or seed not in (0, 1):
        raise ValueError('Primary completion uses folds 0,2,4; repeat uses fold 1 seed 1')
    root = Path(root)
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    labels_hash = hashlib.sha256((root / 'data/meta/train_labels.parquet').read_bytes()).hexdigest()
    name = f'{family}_fold{fold}_seed{seed}'
    sources = {}
    for phase, smoke in (('smoke', True), ('full', False)):
        kwargs = dict(fold=fold, smoke=smoke, training_seed=seed, campaign_completion=seed == 0)
        source = (training_source(notebook, 'spatial', labels_hash, **kwargs) if family == 'spatial'
                  else resolution_source(notebook, labels_hash, 384, **kwargs))
        sources[phase] = relocate(source, f'/kaggle/working/{name}/{phase}')
    output = root / 'outputs' / name
    output.mkdir(parents=True, exist_ok=True)
    for phase, source in sources.items():
        (output / f'{phase}.py').write_text(source)
    arm = f'spatial_fold{fold}' if family == 'spatial' else f'resolution384_fold{fold}'
    smoke_arm = 'ddp_smoke' if family == 'spatial' else arm
    runner = 'import ast, json, hashlib, subprocess, sys, time\nfrom pathlib import Path\n'
    runner += inspect.getsource(ColabPaths) + '\n' + inspect.getsource(relocate)
    runner += f'''
SOURCES = {sources!r}
OUTPUT_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working')
attempt = OUTPUT_ROOT / {name!r}
attempt.mkdir(parents=True, exist_ok=False)
for phase, source in SOURCES.items():
    parent = attempt / phase
    directory = parent / ({smoke_arm!r} if phase == 'smoke' else {arm!r})
    directory.mkdir(parents=True)
    script = directory / 'train.py'
    source = relocate(source, OUTPUT_ROOT)
    script.write_text(source)
    receipt = {{'state': 'running', 'family': {family!r}, 'fold': {fold}, 'seed': {seed},
               'smoke': phase == 'smoke', 'script_sha256': hashlib.sha256(source.encode()).hexdigest()}}
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, '-u', str(script)], cwd=directory, check=True)
        if phase == 'full':
            required = ['run_contract.json', 'folds.parquet', 'convnext_fold{fold}.pt',
                        'convnext_fold{fold}.pt.json', 'oof_fold{fold}.parquet']
            receipt['artifact_sha256'] = {{}}
            for filename in required:
                with (directory / filename).open('rb') as handle:
                    receipt['artifact_sha256'][filename] = hashlib.file_digest(handle, 'sha256').hexdigest()
        receipt['state'] = 'complete'
    except BaseException as exc:
        receipt.update(state='failed', error=type(exc).__name__)
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        (parent / 'resolution_run.json').write_text(json.dumps(receipt, indent=2) + '\\n')
'''
    (output / 'run_candidate.py').write_text(runner)
    receipt = {'state': 'staged_not_launched', 'family': family, 'fold': fold, 'seed': seed,
               'identity': name, 'split_seed': 0, 'epochs': 12, 'effective_batch': 8,
               'labels_sha256': labels_hash,
               'sources_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
               'launch_requires': ('both standalone screens and incumbent allocation gate; acceptance of seed rule'
                                   if seed else 'both standalone screens, incumbent blend and accepted seed rule passed'),
               'repeat_in_submission': False if seed else None}
    (output / 'staging_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('spatial', 'resolution384'), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.family, args.fold, args.seed))
