"""Stage a 100-study raw-DICOM benchmark of the exact five-fold submission code."""
import ast
import json
from pathlib import Path
import pandas as pd
from src import config
from src.model.artifacts import sha256
from tools.prepare_resolution_submission import replace_once, write_json

ROOT = Path(__file__).resolve().parent.parent

def main():
    submission = ROOT / 'outputs/kaggle_resolution384_submission'
    notebook = json.loads((submission / 'submission.ipynb').read_text())
    oof = pd.read_parquet(ROOT / 'outputs/resolution384_downloads/oof_all_folds.parquet')
    sample = pd.concat([oof[oof.fold == f].sort_values(config.ID_COL).sample(20, random_state=384) for f in range(5)])
    references = {r[config.ID_COL]: {'fold': int(r['fold']), 'probabilities': [float(r[n]) for n in config.LABELS]} for _, r in sample.iterrows()}
    source = ''.join(notebook['cells'][5]['source'])
    source = replace_once(source, 'test_csv = pd.read_csv(find_competition_file("test.csv"))', 'test_csv = pd.read_csv(find_competition_file("train.csv"))\ntest_csv = test_csv.set_index(config.ID_COL).loc[list(REFERENCES)].reset_index()')
    source = replace_once(source, 'find_competition_file("test_series.csv")', 'find_competition_file("train_series.csv")')
    source = replace_once(source, 'config.DATA_ROOT / "test_series"', 'config.DATA_ROOT / "train_series"')
    source = 'REFERENCES = ' + repr(references) + '\n' + source
    notebook['cells'][5]['source'] = source.splitlines(keepends=True)
    benchmark = '''
checks = []
for i, uid in enumerate(ids):
    ref = REFERENCES[uid]
    member_index = next(j for j, m in enumerate(explicit_members) if m['metadata']['fold'] == ref['fold'])
    delta = float(np.max(np.abs(np.asarray(tuned_rows[i][member_index]) - ref['probabilities'])))
    checks.append({'uid': uid, 'fold': ref['fold'], 'max_abs_delta': delta})
projection = loop_elapsed / len(ids) * 1300 / 3600
benchmark_report = {'studies': len(ids), 'fallbacks': failures, 'loop_seconds': loop_elapsed,
    'projected_1300_study_hours': projection, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'checkpoints': actual_checkpoints, 'reference_checks': checks,
    'max_reference_delta': max(c['max_abs_delta'] for c in checks),
    'reference_gate_passed': all(c['max_abs_delta'] < .01 for c in checks),
    'runtime_gate_passed': projection < 6.3, 'submission_notebook_sha256': SUBMISSION_NOTEBOOK_HASH}
pathlib.Path('/kaggle/working/benchmark_report.json').write_text(json.dumps(benchmark_report, indent=2))
print(json.dumps({k: v for k, v in benchmark_report.items() if k != 'reference_checks'}, indent=2))
assert benchmark_report['reference_gate_passed'], 'Held-out prediction reproduction failed'
assert benchmark_report['runtime_gate_passed'], 'Runtime gate failed'
'''
    benchmark = 'SUBMISSION_NOTEBOOK_HASH = ' + repr(sha256(submission / 'submission.ipynb')) + '\n' + benchmark
    notebook['cells'].append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':benchmark.splitlines(keepends=True)})
    notebook['cells'][0]['source'] = ['# 384 five-fold GPU benchmark\n100 fixed held-out studies; prediction reproduction and raw-DICOM runtime. Not a competition submission.\n']
    for c in notebook['cells']:
        if c['cell_type'] == 'code': ast.parse(''.join(c['source']))
    output = ROOT / 'outputs/kaggle_resolution384_benchmark'
    output.mkdir(exist_ok=True)
    write_json(output / 'benchmark.ipynb', notebook)
    metadata = json.loads((submission / 'kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-resolution384-five-fold-benchmark',title='Knee Resolution384 Five Fold Benchmark',code_file='benchmark.ipynb')
    write_json(output / 'kernel-metadata.json', metadata)

if __name__ == '__main__': main()
