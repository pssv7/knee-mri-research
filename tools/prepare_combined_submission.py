"""Stage the 20-model submission: the 0.914 incumbent plus the five 384 models.

    .venv/bin/python tools/prepare_combined_submission.py

Both parts already ran on the hidden test set: the fixed 16/24/protocol
ensemble (15 models, submission 56196386, public 0.914) and the 384 five-fold
ensemble (5 models, submission 56480789, public 0.911). Rather than merge
their code, the kernel runs each staged notebook unchanged in its own Python
subprocess and averages their outputs with equal weight per model, 15/20
and 5/20. On out-of-fold predictions that average is +0.0052 macro over the
incumbent, paired 95% interval [+0.0042, +0.0063] (mined labels).

Separate processes matter: the two notebooks embed different versions of
the repo modules and set different cache geometry, and one interpreter
can't hold both.

Each part changes in exactly two ways:
- it selects its own `ensemble.json` by the hash it already asserts, since
  both model datasets are attached;
- it writes under `/kaggle/working/<part>/` instead of `/kaggle/working/`.

The blend refuses to write a submission unless both parts report their
expected member count, zero fallback studies, the same studies in the same
order, and the same columns. Staged locally; nothing is uploaded or
submitted here.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARTS = {
    'part15': {'staged': 'outputs/kaggle_protocol_submission',
               'ran': 'outputs/kaggle_protocol_submission_receipt/knee-16-24-protocol-fixed-blend.ipynb',
               'members': 15, 'submission': 56196386},
    'part384': {'staged': 'outputs/kaggle_resolution384_submission',
                'ran': 'outputs/kaggle_resolution384_submission_receipt/knee-resolution384-five-fold.ipynb',
                'members': 5, 'submission': 56480789},
}
WEIGHTS = {'part15': 15 / 20, 'part384': 5 / 20}
OUTPUT = 'outputs/kaggle_combined20_submission'
KERNEL = 'pssv27/knee-combined-20-model'
PROJECTED_HOURS = {'part15': 1.925325424773607, 'part384': 2.372179122567177}


def code_cells(path):
    return [''.join(c['source']) for c in json.loads(Path(path).read_text())['cells'] if c['cell_type'] == 'code']


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected exactly one occurrence of {old[:60]!r}')
    return source.replace(old, new)


def part_script(name, root=ROOT):
    """One staged notebook as a script, with only the two permitted edits."""
    spec = PARTS[name]
    cells = code_cells(root / spec['staged'] / 'submission.ipynb')
    if cells != code_cells(root / spec['ran']):
        raise ValueError(f'{name}: staged notebook differs from the one that ran on Kaggle')
    header = cells[0]
    match = re.search(r'assert hashlib\.sha256\(manifests\[0\]\.read_bytes\(\)\)\.hexdigest\(\) == \'([0-9a-f]{64})\'', header)
    if not match:
        raise ValueError(f'{name}: manifest hash assertion not found')
    manifest_hash = match.group(1)
    header = replace_once(
        header, 'manifests = attached_files("ensemble.json")',
        'manifests = [p for p in attached_files("ensemble.json")\n'
        f'             if hashlib.sha256(p.read_bytes()).hexdigest() == {manifest_hash!r}]')
    script = '\n\n'.join([header, *cells[1:]])
    if script.count('/kaggle/working/') == 0:
        raise ValueError(f'{name}: no output paths found')
    script = script.replace('/kaggle/working/', f'/kaggle/working/{name}/')
    compile(script, name, 'exec')
    return script, manifest_hash


def blend(frames, reports, weights=WEIGHTS, expected_members=None):
    """Equal weight per model across the parts; refuses any inconsistency."""
    import numpy as np
    expected_members = expected_members or {n: s['members'] for n, s in PARTS.items()}
    names = list(weights)
    first = frames[names[0]]
    for name in names:
        frame, report = frames[name], reports[name]
        if report.get('members') != expected_members[name]:
            raise ValueError(f'{name}: {report.get("members")} members, expected {expected_members[name]}')
        if report.get('fallbacks') != 0:
            raise ValueError(f'{name}: {report.get("fallbacks")} fallback studies')
        if frame.columns.tolist() != first.columns.tolist():
            raise ValueError(f'{name}: columns differ')
        if frame.iloc[:, 0].tolist() != first.iloc[:, 0].tolist():
            raise ValueError(f'{name}: studies or their order differ')
        values = frame.iloc[:, 1:].to_numpy(float)
        if not (np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()):
            raise ValueError(f'{name}: predictions outside [0, 1]')
    if abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError('weights must sum to 1')
    out = first.copy()
    out.iloc[:, 1:] = sum(weights[n] * frames[n].iloc[:, 1:].to_numpy(float) for n in names)
    return out


def kernel_source(scripts, hashes):
    blend_source = __import__('inspect').getsource(blend)
    return f'''import json, pathlib, subprocess, sys, time, hashlib
import pandas as pd
STARTED = time.monotonic()
PART_SCRIPTS = {scripts!r}
WEIGHTS = {WEIGHTS!r}
PARTS = {{name: {{"members": m}} for name, m in {({n: s["members"] for n, s in PARTS.items()})!r}.items()}}
{blend_source}
timings = {{}}
for name, script in PART_SCRIPTS.items():
    folder = pathlib.Path("/kaggle/working") / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(script)
    started = time.monotonic()
    subprocess.run([sys.executable, str(folder / "run.py")], cwd=folder, check=True)
    timings[name] = time.monotonic() - started
    print(f"{{name}} finished in {{timings[name] / 60:.1f}} min", flush=True)

frames = {{n: pd.read_csv(f"/kaggle/working/{{n}}/submission.csv") for n in PART_SCRIPTS}}
reports = {{n: json.loads(pathlib.Path(f"/kaggle/working/{{n}}/submission_report.json").read_text()) for n in PART_SCRIPTS}}
submission = blend(frames, reports, WEIGHTS, {{n: p["members"] for n, p in PARTS.items()}})
submission.to_csv("/kaggle/working/submission.csv", index=False)
report = {{"rows": len(submission), "members": sum(r["members"] for r in reports.values()),
          "weights": WEIGHTS, "part_seconds": timings, "elapsed_seconds": time.monotonic() - STARTED,
          "manifests": {hashes!r},
          "parts": {{n: {{k: r[k] for k in ("members", "fallbacks", "checkpoints", "submission_sha256")}}
                    for n, r in reports.items()}},
          "submission_sha256": hashlib.sha256(pathlib.Path("/kaggle/working/submission.csv").read_bytes()).hexdigest()}}
pathlib.Path("/kaggle/working/submission_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({{k: v for k, v in report.items() if k != "parts"}}, indent=2))
'''


def prepare(root=ROOT):
    scripts, hashes = {}, {}
    for name in PARTS:
        scripts[name], hashes[name] = part_script(name, root)
    source = kernel_source(scripts, hashes)
    compile(source, 'kernel', 'exec')
    notebook = {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# 20-model blend: 16/24/protocol (15) + 384 (5)\n',
            'Each benchmarked submission notebook runs unchanged in its own process; '
            'outputs are averaged with equal weight per model (15/20, 5/20).\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': source.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                     'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    (out / 'submission.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    metas = [json.loads((root / s['staged'] / 'kernel-metadata.json').read_text()) for s in PARTS.values()]
    if len({m['docker_image'] for m in metas}) != 1:
        raise ValueError('the two parts ran on different images')
    datasets = sorted({d for m in metas for d in m['dataset_sources']})
    metadata = {'id': KERNEL, 'title': 'Knee Combined 20 Model', 'code_file': 'submission.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
                'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4', 'docker_image': metas[0]['docker_image'],
                'dataset_sources': datasets, 'kernel_sources': metas[0]['kernel_sources'],
                'competition_sources': metas[0]['competition_sources']}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    receipt = {'kernel': KERNEL, 'weights': WEIGHTS, 'manifests': hashes,
               'parts': {n: {'staged_notebook_sha256': hashlib.sha256(
                   (root / s['staged'] / 'submission.ipynb').read_bytes()).hexdigest(),
                   'members': s['members'], 'submission': s['submission']} for n, s in PARTS.items()},
               'projected_1300_study_hours': sum(PROJECTED_HOURS.values()),
               'oof': {'incumbent_15': 0.885228, 'combined_20': 0.890410,
                       'paired_delta_95ci': [0.0042, 0.0063], 'labels': 'mined; diagnostic, not leaderboard'},
               'state': 'staged_not_run'}
    (out / 'staging_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return out, receipt


if __name__ == '__main__':
    out, receipt = prepare()
    print(f"staged {out.relative_to(ROOT)}; projected {receipt['projected_1300_study_hours']:.2f} h for 1,300 studies")
