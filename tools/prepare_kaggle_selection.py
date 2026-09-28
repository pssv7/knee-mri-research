"""Stage the series-selection check as one private Kaggle T4 kernel.

Reuses the perturbation kernel body (hash-verified assembly from the attached
Tiny control output, 5-study smoke, reproduction stop) with the selection
runner. Embeds the new code and a per-series header table (slice thickness,
slice count, pixel spacing) built locally from the DICOM index for the outer0
series; everything else is taken from the attached output by SHA-256.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.selection_policies import series_quality
from tools.prepare_colab_perturbation import payload
from tools.prepare_kaggle_perturbation import BODY as PERTURBATION_BODY, SOURCES

KERNEL_ID = 'pssv27/knee-selection-probe-v1'
OUT = 'outputs/kaggle_selection_probe_v1'
NEW = ['src/model/selection_policies.py', 'tools/selection_probe.py']
QUALITY = 'design/series_quality.parquet'


def quality_table(root=ROOT):
    roles = pd.read_csv(root / 'outputs/distinct_sampling_pair_v1/control/roles.csv')
    outer = set(roles.loc[roles.role == 'outer', 'StudyInstanceUID'])
    series = pd.read_csv(root / 'data/meta/train_series.csv')
    wanted = set(series.loc[series.StudyInstanceUID.isin(outer), 'SeriesInstanceUID'])
    index = pd.read_parquet(root / 'data/meta/resolution_index/index_train.parquet',
                            columns=['SeriesInstanceUID', 'SOPInstanceUID', 'SliceThickness', 'PixelSpacing'])
    table = series_quality(index[index.SeriesInstanceUID.isin(wanted)])
    if set(table.SeriesInstanceUID) != wanted:
        raise ValueError('DICOM index does not cover every outer0 series')
    buffer = io.BytesIO()
    table.sort_values('SeriesInstanceUID').to_parquet(buffer, index=False)
    return buffer.getvalue()


def body():
    old = "str(code / 'tools/perturbation_probe.py')"
    if PERTURBATION_BODY.count(old) != 1:
        raise ValueError('Perturbation kernel body changed; review the reuse')
    return PERTURBATION_BODY.replace(old, "str(code / 'tools/selection_probe.py')")


def script(root=ROOT):
    root = Path(root)
    archive, _, _ = payload(root, include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        files = {name: bundle.read(name) for name in
                 ['src/model/perturb.py', 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']}
    files.update({name: (root / name).read_bytes() for name in NEW})
    files[QUALITY] = quality_table(root)
    manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    embedded = {name: base64.b64encode(data).decode() for name, data in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source = (f'BUNDLE_SHA256 = {digest!r}\nMANIFEST = {manifest!r}\nEMBEDDED = {embedded!r}\n' + body())
    compile(source, 'selection_kernel.py', 'exec')
    return source, digest


def main(root=ROOT, destination=None):
    out = Path(destination) if destination else Path(root) / OUT
    out.mkdir(parents=True, exist_ok=False)
    source, digest = script(root)
    (out / 'selection_kernel.py').write_text(source)
    metadata = {'id': KERNEL_ID, 'title': 'Knee Selection Probe V1', 'code_file': 'selection_kernel.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
                'kernel_sources': SOURCES, 'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(out, digest)
    return out


if __name__ == '__main__':
    main()
