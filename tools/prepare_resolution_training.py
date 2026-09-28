"""Stage isolated matched resolution arms; each fits within its own GPU session."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.prepare_kaggle_training_pair import training_source, ROOT


def resolution_source(notebook, labels_hash, size, fold=1, smoke=False, training_seed=0, campaign_completion=False):
    if size not in (224, 384) or fold not in ((0, 2, 4) if campaign_completion else (1, 3)):
        raise ValueError('supported screening geometry is 224/384 on folds 1/3')
    source = training_source(notebook, 'baseline', labels_hash, fold=fold,
                             smoke=smoke, match_slice=True, training_seed=training_seed,
                             campaign_completion=campaign_completion)
    directory = f'/kaggle/working/resolution{size}_fold{fold}'
    source = source.replace(f'/kaggle/working/baseline_fold{fold}', directory)
    if smoke:
        source = source.replace('/kaggle/working/ddp_smoke', directory)
    tree = ast.parse(source)
    inserted_config = inserted_checkpointing = inserted_contract = False
    for pos in range(len(tree.body) - 1, -1, -1):
        node = tree.body[pos]
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if names == ['RUN'] and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'replace':
            node.value.keywords.append(ast.keyword(arg='name', value=ast.Constant(f'resolution{size}')))
        if names == ['EXPERIMENT']:
            tree.body[pos:pos] = ast.parse(f'config.CACHE_SIZE = {size}\nprint("Resolution arm: {size}px; gradient checkpointing enabled in both arms", flush=True)').body
            inserted_config = True
        if names == ['module']:
            tree.body[pos+1:pos+1] = ast.parse('module.model.backbone.set_grad_checkpointing(True)').body
            inserted_checkpointing = True
        if names == ['_contract']:
            tree.body[pos+1:pos+1] = ast.parse(f'_contract.update(cache_size={size}, cache_slices=24, gradient_checkpointing=True, labels_sha256={labels_hash!r})').body
            inserted_contract = True
    class Geometry(ast.NodeTransformer):
        def visit_Compare(self, node):
            if ast.unparse(node) == 'len(CACHE_PATHS) == 3':
                return ast.parse(f'len(CACHE_PATHS) == {6 if size == 384 else 3}', mode='eval').body
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            if ast.unparse(node) == '(24, 224, 224)':
                return ast.parse(f'(24, {size}, {size})', mode='eval').body
            return self.generic_visit(node)
    tree = Geometry().visit(tree)
    assert inserted_config and inserted_checkpointing and inserted_contract
    # Check receipts before any dataset construction or GPU model allocation.
    if size == 384:
        for pos, node in enumerate(tree.body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'manifest' for t in node.targets):
                check = '''from src.data.resolution import validate_cache_inputs
validate_cache_inputs(CACHE_FILES_FOR_VALIDATION, expected_size=384)
'''.replace('CACHE_FILES_FOR_VALIDATION', 'cache_paths')
                tree.body[pos:pos] = ast.parse(check).body
                break
        else:
            raise ValueError('cache assembly location changed')
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    from tools.training_measurement import instrument
    source = instrument(source)
    ast.parse(source)
    return source


def prepare(size, fold=1, smoke=False, root=ROOT):
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    labels_hash = hashlib.sha256((root / 'data/meta/train_labels.parquet').read_bytes()).hexdigest()
    source = resolution_source(notebook, labels_hash, size, fold, smoke)
    name = f'resolution{size}_fold{fold}'
    directory = f'/kaggle/working/{name}'
    # Main guard applies only to the parent; Lightning re-executes train.py.
    runner = f'''import pathlib, subprocess, sys, time, json
directory = pathlib.Path({directory!r})
directory.mkdir(exist_ok=False)
source = {source!r}
script = directory / 'train.py'
script.write_text(source)
started = time.monotonic()
receipt = {{'state': 'running', 'size': {size}, 'fold': {fold}, 'smoke': {smoke!r}}}
path = pathlib.Path('/kaggle/working/resolution_run.json')
path.write_text(json.dumps(receipt, indent=2))
try:
    subprocess.run([sys.executable, '-u', str(script)], cwd=directory, check=True)
    receipt['state'] = 'complete'
except BaseException as exc:
    receipt.update(state='failed', error=f'{{type(exc).__name__}}: {{exc}}')
    raise
finally:
    receipt['elapsed_seconds'] = time.monotonic() - started
    path.write_text(json.dumps(receipt, indent=2) + '\\n')
'''
    ast.parse(runner)
    suffix = '-smoke' if smoke else ''
    output = root / f'outputs/kaggle_{name}{suffix}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_resolution.py').write_text(runner)
    metadata = json.loads((root / 'outputs/kaggle_slice_transformer_fold1/kernel-metadata.json').read_text())
    metadata.update(id=f'pssv27/knee-resolution-{size}-fold{fold}{suffix}',
                    title=f'Knee Resolution {size} Fold{fold}' + (' Smoke' if smoke else ''),
                    code_file='run_resolution.py')
    if size == 384:
        metadata['kernel_sources'] = [f'pssv27/knee-cache-384-shard-{i}' for i in range(6)] + ['pssv27/knee-side-classifier']
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'staging_receipt.json').write_text(json.dumps({
        'script_sha256': hashlib.sha256(runner.encode()).hexdigest(),
        'labels_sha256': labels_hash, 'size': size, 'fold': fold, 'smoke': smoke,
        'gradient_checkpointing': True, 'effective_batch': 8, 'epochs': 12,
        'parent_timeout': None}, indent=2) + '\n')
    print(output)


def prepare_probe(root=ROOT):
    notebook = json.loads((root / 'notebooks/08_finetune.ipynb').read_text())
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    tree = ast.parse(cells[1])
    files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    source = f'''import pathlib, sys
FILES = {files!r}
for name, content in FILES.items():
    path = pathlib.Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
sys.path.insert(0, str(pathlib.Path.cwd()))
''' + (root / 'tools/resolution_gpu_probe.py').read_text()
    ast.parse(source)
    output = root / 'outputs/kaggle_resolution_gpu_probe'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'probe.py').write_text(source)
    metadata = json.loads((root / 'outputs/kaggle_slice_transformer_fold1/kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-resolution-gpu-probe', title='Knee Resolution GPU Probe',
                    code_file='probe.py', dataset_sources=[], kernel_sources=[],
                    competition_sources=[], enable_internet=False)
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(224, 384))
    parser.add_argument('--fold', type=int, choices=(1, 3), default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        prepare_probe()
    elif args.size is not None:
        prepare(args.size, args.fold, args.smoke)
    else:
        parser.error('--size or --probe is required')
