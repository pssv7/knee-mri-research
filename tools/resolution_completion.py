"""384-only completion notebook: retain training, remove historical verdicts."""

import ast
import inspect


def clean_training_source(source):
    tree = ast.parse(source)
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, n in enumerate(tree.body) if assignment(n, '_expected'))
    end = next(i for i, n in enumerate(tree.body) if assignment(n, 'checkpointer'))
    tree.body[start:end] = ast.parse('print(f"384-only training: fold {FOLD}; gold is diagnostic only")').body
    start = next((i for i, n in enumerate(tree.body) if assignment(n, 'FROZEN')), None)
    end = next(i for i, n in enumerate(tree.body)
               if isinstance(n, ast.If) and ast.unparse(n.test) == 'AVERAGES') if start is not None else None
    if start is not None:
        tree.body[start:end] = ast.parse('''
print("Gold-58 diagnostic AUC — selected checkpoint")
scores = []
for i, name in enumerate(config.LABELS):
    value = auc(truth[name].to_numpy().astype(int), gold_probs[:, i])
    scores.append(value)
    print(f"{name:18s} {value:.4f}")
macro = float(np.nanmean(scores))
print(f"Gold macro: {macro:.6f}")
''').body
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for i, child in enumerate(n.body):
                if assignment(child, '_val'):
                    n.body = n.body[:i]
                    break
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if 'GPU T4 x2 roughly halves' in n.value:
                n.value = 'Single-GPU training; accumulation holds effective batch at eight.'
            elif 'cache geometry:' in n.value:
                n.value = 'Default config geometry (overridden below):'
            elif 'Resolution arm: 384px;' in n.value:
                n.value = 'Training resolution: 384px; gradient checkpointing enabled'
            elif 'P2 -- selection-free weights' in n.value:
                n.value = '\nGold diagnostics for exported weights (not used for selection)'
            elif "gold's macro SE" in n.value:
                n.value = 'Gold is diagnostic only; no automatic weight selection or submission.'
            elif 'Matched-run preflight passed' in n.value:
                n.value = '384 training preflight passed'
            elif 'only the P2 comparison is unavailable' in n.value:
                n.value = 'Averaged exports are missing; keep the selected checkpoint and inspect the export log.'
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    compile(result, '<384 training>', 'exec')
    return result


def run_remaining_folds(sources_by_fold, drive_root, local_root, folds, fold1_directory):
    """Bind completion to the successful fold-1 contract; skip verified completions."""
    from pathlib import Path
    import json
    folds = list(folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {0, 2, 3, 4}:
        raise ValueError('Choose unique remaining folds from 0, 2, 3, 4; fold 1 is preserved')
    reference_dir = Path(fold1_directory)
    reference = json.loads((reference_dir / 'run_contract.json').read_text())
    receipt = json.loads((reference_dir.parent / 'resolution_run.json').read_text())
    assert receipt['state'] == 'complete' and not receipt['smoke']
    assert reference['fold'] == 1 and reference['cache_size'] == 384 and reference['seed'] == 0
    meta = json.loads((reference_dir / 'convnext_fold1.pt.json').read_text())
    assert digest(reference_dir / 'convnext_fold1.pt') == meta['sha256']
    original_inputs = json.loads((reference_dir.parent.parent / 'input_receipt.json').read_text())
    inputs = json.loads((Path(local_root) / 'input_receipt.json').read_text())
    for key in ('metadata_sha256', 'side_predictions_sha256'):
        assert inputs[key] == original_inputs[key], f'Fold-1 input mismatch: {key}'
    assert inputs['shards'] == original_inputs['shards'], 'Fold-1 cache receipts differ'
    results = {}
    for fold in folds:
        required = {f'convnext_fold{fold}.pt', f'convnext_fold{fold}.pt.json',
                    f'oof_fold{fold}.parquet', f'gold_predictions_fold{fold}.parquet',
                    'run_contract.json', 'folds.parquet'}
        fold_root = Path(drive_root) / f'fold{fold}'
        completed = []
        for progress_path in fold_root.glob('attempt_*/progress.json'):
            progress = json.loads(progress_path.read_text())
            if progress.get('state') != 'complete':
                continue
            assert required <= set(progress['sha256']), 'Incomplete completed-run artifact receipt'
            output = progress_path.parent / 'full' / f'resolution384_fold{fold}'
            contract = json.loads((output / 'run_contract.json').read_text())
            ignored = {'fold', 'steps', 'warmup_steps'}
            assert {k: v for k, v in contract.items() if k not in ignored} == {
                k: v for k, v in reference.items() if k not in ignored}, 'Completed run contract mismatch'
            for name, checksum in progress['sha256'].items():
                assert digest(output / name) == checksum, f'Completed artifact changed: {output / name}'
            completed.append(str(output))
        if len(completed) > 1:
            raise RuntimeError(f'Multiple completed attempts for fold {fold}; choose a separate DRIVE_OUT')
        if completed:
            results[fold] = completed[0]
            print(f'Skipping verified completed fold {fold}: {completed[0]}', flush=True)
            continue
        sources = {phase: 'REFERENCE_CONTRACT = ' + repr(reference) + '\n' + source
                   for phase, source in sources_by_fold[fold].items()}
        results[fold] = run_training(sources, fold_root, local_root, fold=fold, size=384)
    return results


def configure_notebook(cells):
    def set_source(cell, text):
        cell['source'] = (text.strip() + '\n').splitlines(keepends=True)
    set_source(cells[0], '''# 384-only ConvNeXt: remaining folds 0, 2, 3, 4

Continue the successful 384 fold-1 recipe: ConvNeXt-Tiny, protocol selection,
16 slices × five series, 12 epochs, seed 0, effective batch 8, gradient checkpointing.
No 224 training, frozen-model comparisons, blend scoring, or automatic submission.
This is the user-requested 384-only expansion, not a claim that the earlier screening gate passed.

Use one A100 GPU and the existing six Drive shards with original receipts and metadata.
Allow about 90 GiB free local disk. Fold 1 took about 2.8 hours including scoring;
budget roughly 11–12 GPU hours for four folds, plus staging. Colab may disconnect:
run one fold per session by setting FOLDS_TO_TRAIN = [0], then [2], [3], [4].

The completed fold-1 directory is read-only. New outputs go to a separate Drive root.
Completed new folds are hash-checked and skipped on rerun; interrupted folds restart
from epoch zero in a fresh attempt (no optimizer/EMA resume). Keep all exports.
Checkpoint selection uses validation only; gold and EMA/SWA scores are diagnostic.
Future five-fold submission would average one consistent checkpoint type per fold;
this notebook trains and exports only, and does not build or submit that ensemble.''')
    set_source(cells[1], '''DRIVE_CACHE = "/content/drive/MyDrive/knee_cache_384_v1"
DRIVE_OUT = "/content/drive/MyDrive/knee_runs/resolution384_remaining_v1"
LOCAL_INPUTS = "/content/knee_resolution_inputs"
FOLD1_DIRECTORY = "/content/drive/MyDrive/knee_runs/resolution_pair/fold1/384/attempt_b5e73e547b5a/full/resolution384_fold1"
FOLDS_TO_TRAIN = [0, 2, 3, 4]
''')
    for cell in cells:
        text = ''.join(cell['source'])
        if text.startswith('## 4.'):
            set_source(cell, '## 4. Train remaining 384 folds\nChecks the saved fold-1 contract and inputs. Each remaining fold gets a smoke test, then full training. Completed folds are skipped.')
        elif text.startswith('RUN_DIRS = run_resolution_folds'):
            set_source(cell, inspect.getsource(run_remaining_folds) + '\nRUN_DIRS = run_remaining_folds(TRAINING_SOURCES_BY_FOLD, DRIVE_OUT, LOCAL_INPUTS, FOLDS_TO_TRAIN, FOLD1_DIRECTORY)')
