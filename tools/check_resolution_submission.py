"""CPU smoke of the staged 384 notebook with real fold weights and local DICOMs."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from tools import check_submission
    from src import config
    torch.set_num_threads(2)
    notebook = json.loads((ROOT / 'outputs/kaggle_resolution384_submission/submission.ipynb').read_text())
    package = ROOT / 'outputs/ensemble_resolution384'
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data = base / 'input/competitions/rsna-knee'
        data.mkdir(parents=True)
        working = base / 'working'
        working.mkdir()
        inputs = base / 'input/models'
        inputs.mkdir()
        check_submission.N_STUDIES = 1
        check_submission.SERIES_PER_STUDY = 1
        check_submission.build_fake_competition(data)
        frame = pd.read_csv(data / 'test.csv').iloc[:1]
        frame.to_csv(data / 'test.csv', index=False)
        for path in package.iterdir():
            (inputs / path.name).symlink_to(path)
        (inputs / 'side_model.pt').symlink_to(ROOT / 'side_model.zip')
        (inputs / 'train_labels.parquet').symlink_to(ROOT / 'data/meta/train_labels.parquet')
        os.environ['KNEE_DATA_ROOT'] = str(data)
        old_cwd = Path.cwd()
        os.chdir(working)
        namespace = {'__name__': '__main__'}
        try:
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                source = source.replace('/kaggle/input', str(base / 'input')).replace('/kaggle/working', str(working))
                source = source.replace('assert device == "cuda", "GPU required"', 'assert device == "cpu"')
                source = source.replace('torch.cuda.get_device_name()', '"local CPU smoke"')
                exec(compile(source, f'<staged cell {i}>', 'exec'), namespace)
                print(f'Cell {i}: passed', flush=True)
            assert namespace['failures'] == 0
            # Force the learned side path too; local DICOM tags may otherwise bypass it.
            namespace['test_tag_sides'].clear()
            side = namespace['predict_side'](namespace['volumes'], uids=namespace['uids'])
            assert side in ('L', 'R')
            headers = [namespace['dicom_io'].read_header(p) for p in sorted((data / 'test_series').rglob('*.dcm'))]
            expected_224 = namespace['cache'].build_series_volume(headers, size=224)['volume']
            actual_224 = namespace['side_volumes_224'][namespace['uids'][0]]
            assert np.array_equal(expected_224, actual_224)
            assert namespace['volumes'][0].shape == (24, 384, 384)
            report = json.loads((working / 'submission_report.json').read_text())
            report.update(test='local synthetic study from real dev DICOM; actual five fold weights', side_224_exact=True, side_cnn_path=True, gpu_runtime_validated=False)
            (ROOT / 'outputs/kaggle_resolution384_submission/local_smoke_report.json').write_text(json.dumps(report, indent=2) + '\n')
        finally:
            os.chdir(old_cwd)
    print('Actual five-fold notebook smoke passed; GPU/runtime benchmark remains pending.')


if __name__ == '__main__':
    main()
