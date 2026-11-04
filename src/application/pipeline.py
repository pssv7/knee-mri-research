"""Local application core matching the reference parse/infer/send-results stages."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

from src.config import ID_COL, LABELS


@dataclass(frozen=True)
class Request:
    request_id: str
    study_id: str
    model_version: str
    preprocessing_id: str
    schema_version: int = 1

    @classmethod
    def parse(cls, value: Mapping):
        fields = {"request_id", ID_COL, "model_version", "preprocessing_id", "schema_version"}
        if set(value) != fields:
            raise ValueError("request must contain exactly: " + ", ".join(sorted(fields)))
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported request schema_version")
        for key in fields - {"schema_version"}:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        return cls(value["request_id"], value[ID_COL], value["model_version"],
                   value["preprocessing_id"])


@dataclass(frozen=True)
class Result:
    request_id: str
    study_id: str
    model_version: str
    preprocessing_id: str
    probabilities: dict
    schema_version: int = 1

    def payload(self):
        value = asdict(self)
        value[ID_COL] = value.pop("study_id")
        return value


class LocalResultStore:
    """Atomic local JSON writes; same request and result is an idempotent retry.

    Conflicting reuse of a request ID is an error, never an overwrite. This is
    the development sink; object storage and event delivery are separate adapters.
    """

    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, result: Result) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(result.request_id.encode()).hexdigest() + ".json"
        target = self.root / key
        data = json.dumps(result.payload(), allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != data:
                    raise ValueError("request_id already has a different result")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(target)


class Pipeline:
    """Injected adapters keep the orchestration usable locally and in a server.

    prepare(request) returns an existing load_study-style mapping;
    predict(prepared) returns twelve probabilities; store(result) returns a URI.
    A completed response is returned only after store succeeds.
    """

    def __init__(self, *, model_version: str, preprocessing_id: str,
                 prepare: Callable, predict: Callable, store: Callable):
        self.model_version = model_version
        self.preprocessing_id = preprocessing_id
        self.prepare = prepare
        self.predict = predict
        self.store = store

    def execute(self, requests: Sequence[Mapping]) -> list[dict]:
        responses = []
        for raw in requests:
            stage = "parse_metadata"
            request_id = raw.get("request_id") if isinstance(raw, Mapping) else None
            if not isinstance(request_id, str):
                request_id = None
            try:
                request = Request.parse(raw)
                if request.model_version != self.model_version:
                    raise ValueError("model_version does not match loaded model")
                if request.preprocessing_id != self.preprocessing_id:
                    raise ValueError("preprocessing_id does not match loaded recipe")
                stage = "prepare"
                prepared = self.prepare(request)
                if prepared[ID_COL] != request.study_id:
                    raise ValueError("prepared study identity does not match request")
                if len(prepared["volumes"]) == 0:
                    raise ValueError("study has no usable series")
                stage = "infer"
                values = list(self.predict(prepared))
                if len(values) != len(LABELS):
                    raise ValueError("prediction must contain twelve probabilities")
                values = [float(value) for value in values]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                    raise ValueError("prediction contains an invalid probability")
                result = Result(request.request_id, request.study_id,
                                request.model_version, request.preprocessing_id,
                                dict(zip(LABELS, values)))
                stage = "send_results"
                uri = self.store(result)
                if not isinstance(uri, str) or not uri:
                    raise ValueError("store did not acknowledge a result URI")
                responses.append({"request_id": request_id, "status": "completed",
                                  "result_uri": uri, "result": result.payload()})
            except Exception as error:
                responses.append({"request_id": request_id, "status": "failed",
                                  "stage": stage, "error_type": type(error).__name__,
                                  "error": str(error)})
        return responses
