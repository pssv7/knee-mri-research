import json
from types import SimpleNamespace

import pytest

from tools.training_measurement import TrainingMeasurement


@pytest.mark.parametrize('peak,passed', [(35, True), (37, False)])
def test_training_and_validation_peaks_are_separate_and_gate_is_enforced(tmp_path, monkeypatch, peak, passed):
    from tools import training_measurement as module
    memory = {'value': peak * 2**30}
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: memory['value'],
                           max_memory_reserved=lambda: memory['value'], get_device_name=lambda: 'fixture')
    monkeypatch.setattr(module.torch, 'cuda', cuda)
    callback = TrainingMeasurement(tmp_path / 'measurement.json')
    trainer = SimpleNamespace(current_epoch=0, precision='16-mixed', accumulate_grad_batches=8,
                              is_global_zero=True, sanity_checking=False)
    callback.on_train_epoch_start(trainer, None)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    if not passed:
        with pytest.raises(RuntimeError, match='36-GiB'):
            callback.on_validation_start(trainer, None)
    else:
        callback.on_validation_start(trainer, None)
        memory['value'] = 3 * 2**30
        callback.on_validation_batch_end(trainer, None, None, None, 0)
        callback.on_validation_end(trainer, None)
    report = json.loads((tmp_path / 'measurement.json').read_text())
    assert report['training_memory_passed'] is passed
    assert report['phases']['train'][0]['peak_reserved_gib'] == peak
    if passed:
        assert report['phases']['validation'][0]['peak_reserved_gib'] == 3
