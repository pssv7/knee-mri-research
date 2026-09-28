"""Package a private, CPU-only full-cache preflight. Does not launch a job."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

BODY = r'''
import base64
import io
import json
from pathlib import Path
import sys
import time
import zipfile

started = time.monotonic()
work = Path('/kaggle/working/nested_proxy_preflight')
work.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(work)
sys.path.insert(0, str(work))
from src.validation.proxy import sha, write_json_new
from tools.train_nested_proxy_control import run

receipt = {'training_launched': False, 'gpu_requested': False,
           'endpoint': 'input_integrity_only', 'image_gain_demonstrated': False}
receipt_path = Path('/kaggle/working/preflight_receipt.json')
try:
    root = Path('/kaggle/input')
    def find_one(pattern):
        matches = sorted(root.rglob(pattern))
        if len(matches) != 1:
            raise ValueError(f'Expected one {pattern}, found {len(matches)}')
        return matches[0]
    def entry(path):
        print('Hashing', path.name, path.stat().st_size, flush=True)
        return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    arrays = sorted(root.rglob('cache_train_series_*.npy'))
    if len(arrays) != 3:
        raise ValueError(f'Expected three original cache arrays, found {len(arrays)}')
    shards = []
    for array in arrays:
        manifest_name = array.name.replace('cache_', 'manifest_', 1).replace('.npy', '.parquet')
        shards.append({'array': entry(array), 'manifest': entry(find_one(manifest_name))})
    inputs = {'series_meta': entry(find_one('train_series.csv')),
              'side_predictions': entry(find_one('side_predictions.parquet')),
              'shards': shards}
    input_path = work / 'inputs.json'
    write_json_new(input_path, inputs)
    receipt['input_config_sha256'] = sha(input_path)
    design = work / 'outputs/validation_fallback_v2/outer0'
    receipt['spec_sha256'] = sha(design / 'spec.json')
    receipt.update(run(design, input_path, None, execute=False))
except Exception as exc:
    receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    receipt['elapsed_seconds'] = time.monotonic() - started
    write_json_new(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
'''


def prepare():
    out = ROOT / 'outputs/kaggle_nested_proxy_preflight_v1'
    out.mkdir(exist_ok=False)
    design = ROOT / 'outputs/validation_fallback_v2/outer0'
    spec = json.loads((design / 'spec.json').read_text())
    from src.validation.proxy import sha
    paths = set(spec['runtime_code_sha256']) | {
        'src/__init__.py', 'src/labels/__init__.py', 'src/validation/__init__.py',
        'outputs/validation_fallback_v2/exclusions.csv'}
    paths.update(str(p.relative_to(ROOT)) for p in design.iterdir() if p.is_file())
    for name, expected in spec['runtime_code_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen code changed: {name}')
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(paths):
            archive.write(ROOT / name, name)
    code = 'PAYLOAD = ' + repr(base64.b64encode(payload.getvalue()).decode()) + '\n' + BODY
    compile(code, 'preflight.py', 'exec')
    (out / 'preflight.py').write_text(code)
    metadata = {'id': 'pssv27/knee-nested-proxy-preflight-v1',
        'title': 'Knee Nested Proxy Preflight V1', 'code_file': 'preflight.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
        'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': [f'pssv27/knee-cache-224-01-shard-{i}' for i in (1, 2, 3)] +
                          ['pssv27/knee-side-classifier']}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'package_manifest.json').write_text(json.dumps({
        'files': {name: sha(ROOT / name) for name in sorted(paths)},
        'script_sha256': sha(out / 'preflight.py'),
        'purpose': 'CPU-only input preflight; no training or model evaluation'}, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    prepare()
