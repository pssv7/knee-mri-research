import json

import numpy as np
import pytest
import torch

from src.application.adapters import TorchPredictor
from src.application.pipeline import LocalResultStore, Pipeline
from src.config import ID_COL, LABELS
from src.model.dataset import META_COLUMNS


def request(**changes):
    return {"schema_version": 1, "request_id": "request-1", ID_COL: "study-1",
            "model_version": "test-v1", "preprocessing_id": "test-prep-v1", **changes}


def pipeline(tmp_path, **changes):
    options = dict(model_version="test-v1", preprocessing_id="test-prep-v1",
                   prepare=lambda req: {ID_COL: req.study_id, "volumes": [1]},
                   predict=lambda item: np.linspace(0, 1, 12),
                   store=LocalResultStore(tmp_path))
    return Pipeline(**(options | changes))


def test_roundtrip_and_retry(tmp_path):
    app = pipeline(tmp_path)
    first, retry = app.execute([request(), request()])
    assert first == retry
    assert first["status"] == "completed"
    assert list(first["result"]["probabilities"]) == LABELS
    with open(first["result_uri"]) as handle:
        assert json.load(handle) == first["result"]
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"request_id": " "},
    {"model_version": "wrong"}, {"preprocessing_id": "wrong"}, {"report": "text"},
])
def test_bad_metadata_isolated(tmp_path, change):
    failed, good = pipeline(tmp_path).execute([request(**change), request()])
    assert failed["status"] == "failed"
    assert failed["stage"] == "parse_metadata"
    assert good["status"] == "completed"


@pytest.mark.parametrize("values", [[0.1] * 11, [float("nan")] * 12,
                                     [float("inf")] * 12, [-0.1] * 12, [1.1] * 12])
def test_invalid_predictions_never_saved(tmp_path, values):
    response = pipeline(tmp_path, predict=lambda _: values).execute([request()])[0]
    assert response["stage"] == "infer"
    assert response["status"] == "failed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("study,volumes", [("wrong", [1]), ("study-1", [])])
def test_preparation_validation(tmp_path, study, volumes):
    response = pipeline(tmp_path, prepare=lambda _: {
        ID_COL: study, "volumes": volumes}).execute([request()])[0]
    assert response["stage"] == "prepare"
    assert not list(tmp_path.iterdir())


def test_storage_failure_is_not_completion(tmp_path):
    def fail(_):
        raise OSError("disk unavailable")
    response = pipeline(tmp_path, store=fail).execute([request()])[0]
    assert response["status"] == "failed"
    assert response["stage"] == "send_results"


def test_conflicting_request_does_not_overwrite(tmp_path):
    app = pipeline(tmp_path)
    first, conflict = app.execute([request(), request(**{ID_COL: "study-2"})])
    assert conflict["status"] == "failed"
    with open(first["result_uri"]) as handle:
        assert json.load(handle)[ID_COL] == "study-1"


def test_request_id_cannot_escape_store(tmp_path):
    response = pipeline(tmp_path).execute([request(request_id="../../escape")])[0]
    assert response["status"] == "completed"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_torch_adapter_matches_direct_model():
    from src.model.net import KneeNet

    torch.manual_seed(123)
    # Small real KneeNet with an inexpensive backbone, using actual raw-volume path.
    backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten(), torch.nn.Linear(3, 8))
    model = KneeNet(backbone, feature_dim=8, n_heads=2, series_layers=1).eval()
    volumes = np.arange(2 * 4 * 8 * 8, dtype=np.uint8).reshape(2, 4, 8, 8)
    meta = np.zeros((2, len(META_COLUMNS)), dtype=np.float32)
    item = {ID_COL: "study", "volumes": volumes, "meta": meta,
            "report": "must never enter inference"}
    with torch.inference_mode():
        expected = model(torch.from_numpy(volumes)[None], torch.from_numpy(meta)[None],
                         torch.ones(1, 2, dtype=torch.bool)).sigmoid()[0].tolist()
    assert TorchPredictor(model)(item) == pytest.approx(expected, abs=1e-7)
