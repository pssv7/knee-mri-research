"""Mirror augmentation with labels kept, its generated runner, and the kernel."""
import json

import numpy as np
import pytest

from src import config
from src.model import augment
from src.model.dataset import META_COLUMNS
from src.model.mirror_augment import MirrorStudies, mirror_loader
from test_colab_perturbation import ROOT, Trunk, miniature, surrogate_gpu


class Studies:
    def __init__(self, n=4):
        rng = np.random.default_rng(0)
        self.items = []
        for i in range(n):
            meta = np.zeros((2, len(META_COLUMNS)), dtype=np.float32)
            meta[0, augment.SAGITTAL_COLUMN] = 1.0
            meta[:, META_COLUMNS.index('is_right')] = 1.0
            self.items.append({config.ID_COL: f's{i}',
                               'volumes': rng.integers(0, 256, (2, 6, 8, 8), dtype=np.uint8),
                               'meta': meta, 'targets': np.arange(12, dtype=np.float32),
                               'weights': np.ones(12, dtype=np.float32)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in item.items()}


def test_mirror_flips_every_series_on_its_plane_axis_and_keeps_labels():
    base = Studies()
    item = MirrorStudies(base, 1.0)[0]
    original = base[0]
    assert item['mirrored'] is True
    assert np.array_equal(item['volumes'][0], original['volumes'][0][::-1])       # sagittal: slice axis
    assert np.array_equal(item['volumes'][1], original['volumes'][1][:, :, ::-1])  # coronal/axial: columns
    for key in ('targets', 'weights', 'meta'):
        assert np.array_equal(item[key], original[key])  # labels kept, is_right as acquired
    kept = MirrorStudies(base, 0.0)[0]
    assert kept['mirrored'] is False and np.array_equal(kept['volumes'], original['volumes'])


def test_mirror_rate_and_determinism():
    data = MirrorStudies(Studies(1), 0.5)
    np.random.seed(0)
    first = [data[0]['mirrored'] for _ in range(400)]
    np.random.seed(0)
    assert first == [data[0]['mirrored'] for _ in range(400)]
    assert 0.4 < np.mean(first) < 0.6
    with pytest.raises(ValueError):
        MirrorStudies(Studies(1), 1.5)


def test_loader_copies_the_training_settings():
    from torch.utils.data import DataLoader
    from src.model.dataset import collate_studies
    original = DataLoader(Studies(), batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_studies)
    wrapped = mirror_loader(original, 0.5)
    assert isinstance(wrapped.dataset, MirrorStudies) and wrapped.dataset.base is original.dataset
    assert (wrapped.batch_size, wrapped.num_workers, wrapped.collate_fn) == (1, 0, collate_studies)
    batch = next(iter(wrapped))
    assert batch['volumes'].shape == (1, 2, 6, 8, 8)


def cpu_training(monkeypatch):
    import lightning.pytorch as pl
    surrogate_gpu(monkeypatch)
    trainer_class = pl.Trainer
    def cpu_trainer(**kwargs):
        kwargs.update(accelerator='cpu', precision='32-true', enable_progress_bar=False)
        return trainer_class(**kwargs)
    monkeypatch.setattr(pl, 'Trainer', cpu_trainer)
    # Worker processes would not inherit the monkeypatches; train in-process.
    from torch.utils import data
    original = data.DataLoader.__init__
    def in_process(self, *args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False)
        original(self, *args, **kwargs)
    monkeypatch.setattr(data.DataLoader, '__init__', in_process)


def test_both_arms_train_on_cpu_and_pass_the_frozen_evaluator(tmp_path, monkeypatch):
    from tools.prepare_kaggle_mirror_screen import AUGMENTATION, mirror_source
    from tools.evaluate_nested_proxy import evaluate
    design, inputs, spec, roles = miniature(tmp_path)
    frozen = (ROOT / 'tools/train_nested_proxy_control.py').read_text()
    cpu_training(monkeypatch)
    import torch
    torch.set_num_threads(1)
    results = {}
    for arm, source in [('control', frozen), ('candidate', mirror_source(frozen))]:
        scope = {'__file__': str(ROOT / 'tools/train_nested_proxy_control.py'), '__name__': f'arm_{arm}'}
        exec(source.split("if __name__ == '__main__':", 1)[0], scope)
        results[arm] = scope['run'](design, inputs, tmp_path / 'runs' / arm, execute=True)
    control = json.loads((tmp_path / 'runs/control/run_contract.json').read_text())
    candidate = json.loads((tmp_path / 'runs/candidate/run_contract.json').read_text())
    assert control['arm'] == 'control' and 'augmentation' not in control
    assert candidate['arm'] == 'candidate' and candidate['augmentation'] == AUGMENTATION
    assert control['initial_weights_sha256'] == candidate['initial_weights_sha256']
    report = evaluate(design, tmp_path / 'runs/control', tmp_path / 'runs/candidate', tmp_path / 'evaluation.json', draws=200)
    assert report['sampling_policies'] == ['legacy', 'legacy'] and report['macro_delta'] is not None


def test_runner_adapter_fails_closed_on_a_changed_runner():
    from tools.prepare_kaggle_mirror_screen import mirror_source
    frozen = (ROOT / 'tools/train_nested_proxy_control.py').read_text()
    with pytest.raises(ValueError, match='needs review'):
        mirror_source(frozen.replace("loader('train', True)", "loader('train', augment=True)"))


def test_kernel_fits_kaggle_and_embeds_both_new_files(tmp_path):
    from tools.prepare_kaggle_mirror_screen import main
    out = main(destination=tmp_path / 'kernel')
    source = (out / 'mirror_screen.py').read_text()
    assert len(source.encode()) < 1_000_000
    compile(source, 'mirror_screen.py', 'exec')
    scope = {}
    exec(source.split('\nimport subprocess, sys\n', 1)[0], scope)
    assert {'src/model/mirror_augment.py', 'tools/train_mirror.py'} <= set(scope['EMBEDDED'])
    assert set(scope['EMBEDDED']) <= set(scope['MANIFEST'])
    assert 'def assemble():' in source and 'def phase(' not in source
    assert "('candidate', 'tools/train_mirror.py')" in source
    metadata = json.loads((out / 'kernel-metadata.json').read_text())
    assert metadata['machine_shape'] == 'NvidiaTeslaT4' and metadata['is_private']
