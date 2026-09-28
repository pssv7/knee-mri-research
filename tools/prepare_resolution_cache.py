"""Stage six private CPU raw-DICOM cache jobs without launching them."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ('src/__init__.py', 'src/config.py', 'src/data/__init__.py',
           'src/data/cache.py', 'src/data/dicom_io.py', 'src/data/resolution.py')


def prepare(root=ROOT):
    files = {path: (root / path).read_text() for path in MODULES}
    reference = json.loads((root / 'outputs/kaggle_slice_transformer_fold1/kernel-metadata.json').read_text())
    for shard in range(6):
        source = f'''import pathlib, sys
FILES = {files!r}
for name, content in FILES.items():
    path = pathlib.Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
sys.path.insert(0, str(pathlib.Path.cwd()))
import pandas as pd
from src import config
from src.data.resolution import build_shard, sha256
roots = [p for p in pathlib.Path('/kaggle/input').iterdir()
         if p.name not in ('competitions', config.COMPETITION)]
paths = [p for root in roots for p in root.rglob('index_train.parquet')]
assert len(paths) == 1, paths
assert sha256(paths[0]) == '145c2791a20a0be9f3673079c74717a4db75434025ca17a8b386b23126d524a7', 'index changed'
index = pd.read_parquet(paths[0])
series = pd.read_csv(config.DATA_ROOT / 'train_series.csv')
assert len(series) == 24371 and series.StudyInstanceUID.nunique() == 4407
build_shard(index, series, config.DATA_ROOT / 'train_series', '/kaggle/working',
            shard={shard}, size=384, shards=6, workers=2)
'''
        ast.parse(source)
        output = root / f'outputs/kaggle_resolution_cache_{shard}'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'build_cache.py').write_text(source)
        metadata = {'id': f'pssv27/knee-cache-384-shard-{shard}',
                    'title': f'Knee Cache 384 Shard {shard}', 'code_file': 'build_cache.py',
                    'language': 'python', 'kernel_type': 'script', 'is_private': True,
                    'enable_gpu': False, 'enable_internet': False,
                    'competition_sources': ['rsna-knee-abnormality-detection'],
                    'dataset_sources': ['pssv27/knee-index'], 'kernel_sources': [],
                    'docker_image': reference['docker_image']}
        (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(output)


if __name__ == '__main__':
    prepare()
