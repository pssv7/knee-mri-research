"""Stage the perturbation probe as one private Kaggle T4 kernel.

Same hash-verified bundle and runner as the Colab notebook
(tools/prepare_colab_perturbation.py). Kaggle attaches the three 224 shards,
the side classifier and the Tiny control's own pair output, so the checkpoint
is taken by hash from the run that produced the reference predictions, on the
same GPU model. Inference only: no training, labels or submission.
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

KERNEL_ID = 'pssv27/knee-perturbation-probe-v1'
OUT = 'outputs/kaggle_perturbation_probe_v1'
SOURCES = ['pssv27/knee-cache-224-01-shard-1', 'pssv27/knee-cache-224-01-shard-2',
           'pssv27/knee-cache-224-01-shard-3', 'pssv27/knee-side-classifier',
           'pssv27/knee-distinct-sampling-pair-v1']
# Bundle files the attached outputs cannot supply: this probe's own code.
EMBED = ['src/model/perturb.py', 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']

BODY = r'''
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightning==2.6.5', 'timm==1.0.26'], check=True)
# Import check in a child process: the probe itself runs in one.
check = subprocess.run([sys.executable, '-c', 'import torch, timm, lightning.pytorch as pl; '
    'print("torch", torch.__version__, "| timm", timm.__version__, "| lightning", pl.__version__, '
    '"| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'],
    capture_output=True, text=True)
print(check.stdout, check.stderr, flush=True)
if check.returncode != 0:
    raise RuntimeError('Packages do not import in a child process')
import base64, hashlib, io, json, os, time, zipfile
from pathlib import Path

out = Path(os.environ.get('PROBE_WORKING', '/kaggle/working'))
root = Path(os.environ.get('PROBE_INPUT', '/kaggle/input'))
code = out / 'probe_code'
started = time.monotonic()
receipt = {'status': 'starting', 'bundle_sha256': BUNDLE_SHA256, 'labels_used': False,
           'training_launched': False, 'competition_submission': False}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


COMPETITION = 'rsna-knee-abnormality-detection'


def attached(name):
    """Files called `name` in attached sources, never walking the DICOM tree."""
    found = [Path(p) for p in [root / 'competitions' / COMPETITION / name, root / COMPETITION / name]]
    for top, dirs, names in os.walk(root):
        parts = Path(top).relative_to(root).parts
        if parts[:1] == ('competitions',) or parts[:1] == (COMPETITION,):
            dirs[:] = []
            continue
        if name in names:
            found.append(Path(top) / name)
    return sorted({p for p in found if p.is_file()})


def locate(entry, verify):
    name = Path(entry['path']).name
    hits = [p for p in attached(name) if p.stat().st_size == entry['bytes']]
    if verify:
        hits = [p for p in hits if sha(p) == entry['sha256']]
    if not hits:
        raise FileNotFoundError(f'No attached input matches {name}')
    # Shards are fully hashed again by the runner's preflight.
    return str(hits[0])


def phase(name, checkpoint, inputs_path, limit=None):
    command = [sys.executable, '-u', str(code / 'tools/perturbation_probe.py'),
               '--design', str(code / 'design/control'), '--inputs', str(inputs_path),
               '--checkpoint', str(checkpoint), '--reference', str(code / 'reference'),
               '--execute', '--out', str(out / name)] + (['--limit', str(limit)] if limit else [])
    env = dict(os.environ, PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='0')
    with (out / f'{name}.log').open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        if process.wait() != 0:
            raise RuntimeError(f'{name} failed; see {name}.log')
    contract = json.loads((out / name / 'run_contract.json').read_text())
    if not contract['reproduction']['passed']:
        raise RuntimeError(f'{name}: identity pass does not reproduce the saved predictions')
    return contract


def assemble():
    """New code from this script; everything else by hash from attached outputs."""
    by_hash = {}
    for top, dirs, names in os.walk(root):
        parts = Path(top).relative_to(root).parts
        if parts[:1] == ('competitions',) or parts[:1] == (COMPETITION,):
            dirs[:] = []
            continue
        for name in names:
            path = Path(top) / name
            if path.stat().st_size < 5 * 10**7:
                by_hash.setdefault(sha(path), path)
    for name, expected in MANIFEST.items():
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in EMBEDDED:
            target.write_bytes(base64.b64decode(EMBEDDED[name]))
        elif expected in by_hash:
            target.write_bytes(by_hash[expected].read_bytes())
        else:
            raise FileNotFoundError(f'No attached file matches frozen {name}')
        if sha(target) != expected:
            raise ValueError(f'Assembled file differs from the frozen bundle: {name}')


try:
    assemble()
    reference = json.loads((code / 'reference/run_contract.json').read_text())
    checkpoints = [p for p in attached('selected.pt') if sha(p) == reference['checkpoint_sha256']]
    if not checkpoints:
        raise FileNotFoundError('The Tiny control selected.pt is not attached')
    inputs = json.loads((code / 'reference/inputs.json').read_text())
    inputs['series_meta']['path'] = locate(inputs['series_meta'], True)
    inputs['side_predictions']['path'] = locate(inputs['side_predictions'], True)
    for shard in inputs['shards']:
        for entry in shard.values():
            entry['path'] = locate(entry, entry['bytes'] < 10**8)
    inputs_path = out / 'inputs.json'
    inputs_path.write_text(json.dumps(inputs, indent=2) + '\n')
    receipt.update(checkpoint=str(checkpoints[0]), checkpoint_sha256=reference['checkpoint_sha256'])
    smoke = phase('smoke', checkpoints[0], inputs_path, limit=5)
    receipt['smoke'] = {'reproduction': smoke['reproduction'], 'elapsed_seconds': smoke['elapsed_seconds']}
    print('Smoke passed; projected full minutes:',
          round(smoke['elapsed_seconds'] / 5 * len(reference['outer_uids']) / 60, 1), flush=True)
    full = phase('full', checkpoints[0], inputs_path)
    receipt.update(status='completed', full={'reproduction': full['reproduction'],
                   'elapsed_seconds': full['elapsed_seconds'], 'studies': full['studies'],
                   'runtime_versions': full['runtime_versions']})
except Exception as exc:
    receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    receipt['elapsed_seconds'] = time.monotonic() - started
    (out / 'probe_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
'''


def script(root=ROOT):
    """Kaggle caps kernel source at 1 MB, so only the new code travels inline.

    Every other bundle file already sits in the attached Tiny control output
    (its `pair_source/` and `control/`), and the kernel takes each one by the
    same SHA-256 the Colab bundle pins. The digest identifies the manifest.
    """
    archive, _, _ = payload(root, include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        embedded = {name: base64.b64encode(bundle.read(name)).decode() for name in EMBED}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source = (f'BUNDLE_SHA256 = {digest!r}\n'
              f'MANIFEST = {manifest!r}\n'
              f'EMBEDDED = {embedded!r}\n' + BODY)
    compile(source, 'probe_kernel.py', 'exec')
    return source, digest


def main(root=ROOT, destination=None):
    out = Path(destination) if destination else Path(root) / OUT
    out.mkdir(parents=True, exist_ok=False)
    source, digest = script(root)
    (out / 'probe_kernel.py').write_text(source)
    metadata = {'id': KERNEL_ID, 'title': 'Knee Perturbation Probe V1', 'code_file': 'probe_kernel.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
                'kernel_sources': SOURCES, 'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(out, digest)
    return out


if __name__ == '__main__':
    main()
