"""Stage the mirror-augmentation screen as one private Kaggle T4x2 kernel.

Two arms train concurrently, one per T4, on the frozen nested outer0 design:
the unmodified frozen runner (control) and the same runner whose training loader
mirrors each study with probability 0.5, labels kept (candidate). Both share
seed, pretrained initialisation, inputs and runtime, which the frozen evaluator
requires. The fresh control is also compared with the historical Kaggle control
to measure run-to-run variation. Declared gate, unchanged from the earlier
screens: macro delta >= +0.005 and no finding below -0.020; eligible for a
separate confirmation only if the paired lower bound is also above zero.
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
from tools.prepare_kaggle_perturbation import BODY as PROBE_BODY, SOURCES

KERNEL_ID = 'pssv27/knee-mirror-screen-v1'
OUT = 'outputs/kaggle_mirror_screen_v1'
PROBABILITY = 0.5
AUGMENTATION = f'mirror_keep_labels_p{PROBABILITY}'


def mirror_source(baseline):
    """Fail closed if the frozen runner no longer has the expected shape."""
    source = baseline
    edits = {
        "    train_loader, select_loader = loader('train', True), loader('selection')\n":
            "    train_loader, select_loader = loader('train', True), loader('selection')\n"
            "    from src.model.mirror_augment import mirror_loader\n"
            f"    train_loader = mirror_loader(train_loader, probability={PROBABILITY})\n",
        "'arm': 'control' if spec.get('sampling_policy', 'legacy') == 'legacy' else 'candidate'":
            f"'arm': 'candidate', 'augmentation': {AUGMENTATION!r}",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f'Frozen runner adapter needs review: {old!r}')
        source = source.replace(old, new)
    compile(source, 'train_mirror.py', 'exec')
    return source


MAIN = r'''
GATE = {'macro_delta_min': 0.005, 'finding_delta_min': -0.02}


def launch(arm, script, device):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONHASHSEED='0')
    log = (out / f'{arm}_execution.log').open('w')
    command = [sys.executable, '-u', str(code / script), '--design', str(code / 'design/control'),
               '--inputs', str(out / 'inputs.json'), '--out', str(out / arm), '--execute']
    return subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT), log


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
        raise RuntimeError('Expected two Tesla T4s, one per arm')
    sys.path.insert(0, str(code))
    from src.model import net
    net.build_backbone(reference['training_recipe']['backbone'], pretrained=True)  # cache warm-up only
    receipt.update(status='running', augmentation=AUGMENTATION, gate=GATE, training_launched=True,
                   historical_control_runtime=reference['runtime_versions'])
    for device, (arm, script) in enumerate([('control', 'tools/train_nested_proxy_control.py'),
                                            ('candidate', 'tools/train_mirror.py')]):
        children[arm], logs[arm] = launch(arm, script, device)
    while True:
        codes = {arm: p.poll() for arm, p in children.items()}
        print('arm status', json.dumps(codes), round((time.monotonic() - started) / 60, 1), 'min', flush=True)
        failed = {a: c for a, c in codes.items() if c not in (None, 0)}
        if failed:
            raise RuntimeError(f'Arm failed; pair stopped without retry: {failed}')
        if all(c == 0 for c in codes.values()):
            break
        time.sleep(120)
    from tools.evaluate_nested_proxy import evaluate
    import pandas as pd
    from src.validation.proxy import ID, LABELS, paired_proxy_evaluation
    report = evaluate(code / 'design/control', out / 'control', out / 'candidate',
                      out / 'outer_proxy_evaluation.json', draws=2000)
    per = report['per_finding']
    point = (report['macro_delta'] is not None and report['macro_delta'] >= GATE['macro_delta_min'] and
             all(v['candidate'] - v['control'] >= GATE['finding_delta_min'] for v in per.values()))
    ci = report['paired_sampling_95ci']
    decision = {'point_screen_passed': point, 'paired_lower_bound_positive': ci is not None and ci[0] > 0,
                'eligible_for_separate_confirmation': point and ci is not None and ci[0] > 0,
                'promotion_allowed': False, 'automatic_followup_training': False, 'gate': GATE}
    (out / 'screen_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    # Run-to-run variation: fresh control against the historical Kaggle control.
    design = code / 'design/control'
    roles = pd.read_csv(design / 'roles.csv')
    outer = roles[roles.role == 'outer'].sort_values(ID)
    truth = (pd.read_parquet(design / 'proxy_labels.parquet').set_index(ID).loc[outer[ID], LABELS] >= .5).astype(float)
    arrays = [pd.read_parquet(p).set_index(ID).loc[outer[ID], LABELS].to_numpy(float)
              for p in [code / 'reference/outer_predictions.parquet', out / 'control/outer_predictions.parquet']]
    rerun = paired_proxy_evaluation(truth.to_numpy(), *arrays, outer.component.to_numpy(), draws=2000)
    (out / 'control_rerun_vs_historical.json').write_text(json.dumps(rerun, indent=2) + '\n')
    receipt.update(status='completed', control_macro=report['control_macro'],
                   candidate_macro=report['candidate_macro'], macro_delta=report['macro_delta'],
                   paired_sampling_95ci=ci, screen_decision=decision,
                   control_rerun_minus_historical=rerun['macro_delta'])
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


def body():
    """Probe kernel's setup and verified-assembly helpers, then the pair driver."""
    start, stop = '\nimport base64, hashlib', '\ndef phase('
    if PROBE_BODY.count(start) != 1 or PROBE_BODY.count(stop) != 1:
        raise ValueError('Probe kernel body changed; review the reuse')
    head = PROBE_BODY.split(stop, 1)[0]
    helpers = PROBE_BODY.split(stop, 1)[1].split('\ndef assemble():', 1)[1].split('\ntry:\n', 1)[0]
    return head + '\ndef assemble():' + helpers + MAIN


def script(root=ROOT):
    root = Path(root)
    archive, _, _ = payload(root, include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        files = {name: bundle.read(name) for name in
                 ['src/model/perturb.py', 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']}
        frozen_runner = bundle.read('tools/train_nested_proxy_control.py').decode()
    files['src/model/mirror_augment.py'] = (root / 'src/model/mirror_augment.py').read_bytes()
    files['tools/train_mirror.py'] = mirror_source(frozen_runner).encode()
    manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    embedded = {name: base64.b64encode(data).decode() for name, data in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source = (f'BUNDLE_SHA256 = {digest!r}\nMANIFEST = {manifest!r}\nEMBEDDED = {embedded!r}\n'
              f'AUGMENTATION = {AUGMENTATION!r}\n' + body())
    compile(source, 'mirror_screen.py', 'exec')
    return source, digest


def main(root=ROOT, destination=None):
    out = Path(destination) if destination else Path(root) / OUT
    out.mkdir(parents=True, exist_ok=False)
    source, digest = script(root)
    (out / 'mirror_screen.py').write_text(source)
    metadata = {'id': KERNEL_ID, 'title': 'Knee Mirror Screen V1', 'code_file': 'mirror_screen.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
                'kernel_sources': SOURCES, 'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(out, digest)
    return out


if __name__ == '__main__':
    main()
