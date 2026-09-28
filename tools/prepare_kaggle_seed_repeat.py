"""Stage a seed repeat of the frozen recipe as one private Kaggle T4x2 kernel.

The frozen recipe is bit-deterministic on Kaggle T4 / torch 2.10 (the mirror and
silence screens' concurrent controls reproduced the historical control exactly),
so screen deltas mix the change with an unmeasured trajectory variation. This
runs the unmodified recipe with seeds 1 and 2, one per T4, to set beside the
historical seed 0. The seed is the only change: head initialisation, data order,
intensity jitter and dropout follow it; the pretrained trunk does not.

Declared reading: per finding, seed SD over seeds {0, 1, 2}; a screen's
per-finding delta is "within seed variation" when |delta| <= 2 x that SD. Three
seeds give a rough SD, and it is reported as such.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_colab_perturbation import payload
from tools.prepare_kaggle_mirror_screen import body as pair_body
from tools.prepare_kaggle_perturbation import SOURCES

KERNEL_ID = 'pssv27/knee-seed-repeat-v1'
OUT = 'outputs/kaggle_seed_repeat_v1'
SEEDS = (1, 2)


def seed_source(baseline, seed):
    """Frozen runner with only the seed replaced. Fails closed on drift."""
    source = baseline
    edits = {
        "    pl.seed_everything(spec['seed'], workers=True)\n":
            f"    pl.seed_everything({seed}, workers=True)\n",
        "    contract = {**spec, 'training_recipe': recipe,":
            f"    contract = {{**spec, 'seed': {seed}, 'seed_repeat_of': spec['seed'], 'training_recipe': recipe,",
        "'arm': 'control' if spec.get('sampling_policy', 'legacy') == 'legacy' else 'candidate'":
            f"'arm': 'seed{seed}'",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f'Frozen runner adapter needs review: {old!r}')
        source = source.replace(old, new)
    compile(source, f'train_seed{seed}.py', 'exec')
    return source


MAIN = r'''
children, logs = {}, {}
try:
    assemble()
    reference = json.loads((code / 'reference/run_contract.json').read_text())
    inputs = json.loads((code / 'reference/inputs.json').read_text())
    inputs['series_meta']['path'] = locate(inputs['series_meta'], True)
    inputs['side_predictions']['path'] = locate(inputs['side_predictions'], True)
    for shard in inputs['shards']:
        for entry in shard.values():
            entry['path'] = locate(entry, entry['bytes'] < 10**8)
    (out / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    import torch
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_name(i) != 'Tesla T4' for i in range(2)):
        raise RuntimeError('Expected two Tesla T4s, one per seed')
    sys.path.insert(0, str(code))
    from src.model import net
    net.build_backbone(reference['training_recipe']['backbone'], pretrained=True)  # cache warm-up only
    receipt.update(status='running', seeds=list(SEEDS), training_launched=True,
                   historical_seed=reference['seed'], historical_runtime=reference['runtime_versions'])
    for device, seed in enumerate(SEEDS):
        arm = f'seed{seed}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONHASHSEED='0')
        logs[arm] = (out / f'{arm}_execution.log').open('w')
        command = [sys.executable, '-u', str(code / f'tools/train_seed{seed}.py'), '--design',
                   str(code / 'design/control'), '--inputs', str(out / 'inputs.json'), '--out', str(out / arm), '--execute']
        children[arm] = subprocess.Popen(command, cwd=code, env=env, stdout=logs[arm], stderr=subprocess.STDOUT)
    while True:
        codes = {arm: p.poll() for arm, p in children.items()}
        print('seed status', json.dumps(codes), round((time.monotonic() - started) / 60, 1), 'min', flush=True)
        failed = {a: c for a, c in codes.items() if c not in (None, 0)}
        if failed:
            raise RuntimeError(f'Seed run failed; stopped without retry: {failed}')
        if all(c == 0 for c in codes.values()):
            break
        time.sleep(120)
    import numpy as np
    import pandas as pd
    from src.validation.proxy import ID, LABELS, paired_proxy_evaluation, strict_aucs, validate_prediction_provenance
    design = code / 'design/control'
    roles = pd.read_csv(design / 'roles.csv')
    outer = roles[roles.role == 'outer'].sort_values(ID)
    truth = (pd.read_parquet(design / 'proxy_labels.parquet').set_index(ID).loc[outer[ID], LABELS] >= .5).to_numpy(float)
    runs = {'seed0': (code / 'reference', reference)}
    for seed in SEEDS:
        runs[f'seed{seed}'] = (out / f'seed{seed}', json.loads((out / f'seed{seed}/run_contract.json').read_text()))
    arrays, summary = {}, {'per_seed': {}}
    for name, (directory, contract) in runs.items():
        predictions = pd.read_parquet(directory / 'outer_predictions.parquet')
        validate_prediction_provenance(roles, contract, predictions)
        arrays[name] = predictions.set_index(ID).loc[outer[ID], LABELS].to_numpy(float)
        aucs, macro = strict_aucs(truth, arrays[name])
        summary['per_seed'][name] = {'seed': contract['seed'], 'macro': macro,
                                     'per_finding': dict(zip(LABELS, map(float, aucs))),
                                     'initial_weights_sha256': contract['initial_weights_sha256'],
                                     'checkpoint_sha256': contract['checkpoint_sha256']}
    table = np.array([[summary['per_seed'][n]['per_finding'][f] for f in LABELS] for n in runs])
    summary['seed_sd'] = dict(zip(LABELS, map(float, table.std(axis=0, ddof=1))))
    macros = [summary['per_seed'][n]['macro'] for n in runs]
    summary['macro_sd'] = float(np.std(macros, ddof=1))
    summary['macro_range'] = [float(min(macros)), float(max(macros))]
    names = list(runs)
    summary['pairwise'] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            e = paired_proxy_evaluation(truth, arrays[names[i]], arrays[names[j]], outer.component.to_numpy(), draws=2000)
            summary['pairwise'][f'{names[j]}_minus_{names[i]}'] = {'macro_delta': e['macro_delta'],
                'ci': e['paired_sampling_95ci'],
                'per_finding': {f: v['candidate'] - v['control'] for f, v in e['per_finding'].items()}}
    (out / 'seed_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    receipt.update(status='completed', macro_by_seed=dict(zip(names, macros)), macro_sd=summary['macro_sd'])
except Exception as exc:
    receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    for p in children.values():
        if p.poll() is None:
            p.terminate()
    for f in logs.values():
        f.close()
    receipt['elapsed_seconds'] = time.monotonic() - started
    (out / 'screen_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
'''


def script(root=ROOT):
    root = Path(root)
    archive, _, _ = payload(root, include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        files = {name: bundle.read(name) for name in
                 ['src/model/perturb.py', 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']}
        frozen_runner = bundle.read('tools/train_nested_proxy_control.py').decode()
    for seed in SEEDS:
        files[f'tools/train_seed{seed}.py'] = seed_source(frozen_runner, seed).encode()
    manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    embedded = {name: base64.b64encode(data).decode() for name, data in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    head, rest = pair_body().split('\ndef assemble():', 1)
    helpers = rest.split('\nGATE = ', 1)[0]
    source = (f'BUNDLE_SHA256 = {digest!r}\nMANIFEST = {manifest!r}\nEMBEDDED = {embedded!r}\n'
              f'SEEDS = {SEEDS!r}\n' + head + '\ndef assemble():' + helpers + MAIN)
    compile(source, 'seed_repeat.py', 'exec')
    return source, digest


def main(root=ROOT, destination=None):
    out = Path(destination) if destination else Path(root) / OUT
    out.mkdir(parents=True, exist_ok=False)
    source, digest = script(root)
    (out / 'seed_repeat.py').write_text(source)
    metadata = {'id': KERNEL_ID, 'title': 'Knee Seed Repeat V1', 'code_file': 'seed_repeat.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
                'kernel_sources': SOURCES, 'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(out, digest)
    return out


if __name__ == '__main__':
    main()
