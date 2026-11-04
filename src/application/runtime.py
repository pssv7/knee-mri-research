"""Configured local batch inference with a content-addressed model registry."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src import config
from src.application.pipeline import LocalResultStore, Pipeline
from src.model import artifacts, dataset


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_settings(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    required = {"schema_version", "ensemble", "manifest", "series_metadata", "sides", "shards"}
    if set(value) != required or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid application configuration schema")
    for key in ("ensemble", "manifest", "series_metadata", "sides"):
        if key == "sides" and value[key] is None:
            continue
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} must be a path")
        value[key] = str((path.parent / value[key]).resolve())
    if not isinstance(value["shards"], dict) or not value["shards"]:
        raise ValueError("shards must map manifest cache_path values to .npy files")
    for key, shard in value["shards"].items():
        if not isinstance(shard, str) or not shard:
            raise ValueError("shard location must be a path")
        value["shards"][key] = str((path.parent / shard).resolve())
    return value


def register_model(manifest_path):
    """Verify explicit members and bind both weights and recipe to their version."""
    path = Path(manifest_path)
    entries = json.loads(path.read_text())["members"]
    if not entries:
        raise ValueError("ensemble has no members")
    members, hashes, folds = [], set(), set()
    for entry in entries:
        checkpoint = (path.parent / entry["checkpoint"]).resolve()
        metadata = artifacts.load_metadata(checkpoint)
        weight = float(entry.get("weight", 1))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weights must be finite and positive")
        fold = (metadata["experiment"]["name"], metadata["fold"])
        if metadata["sha256"] in hashes or fold in folds:
            raise ValueError("duplicate ensemble member")
        hashes.add(metadata["sha256"])
        folds.add(fold)
        members.append({"checkpoint_sha256": metadata["sha256"], "weight": weight,
                        "metadata_sha256": artifacts.sha256(str(checkpoint) + ".json"),
                        "recipe": metadata["experiment"]})
    # Changes to shared preprocessing/model code must also change the version.
    source_files = ["src/config.py", "src/model/artifacts.py", "src/model/dataset.py",
                    "src/model/net.py", "src/application/runtime.py",
                    "src/application/pipeline.py"]
    sources = {name: artifacts.sha256(config.PROJECT_ROOT / name) for name in source_files}
    recipe = {"adapter": "cached-artifacts-v1", "cache_size": config.CACHE_SIZE,
              "cache_slices": config.CACHE_SLICES, "sources": sources,
              "members": [m["recipe"] for m in members]}
    registry = {"schema_version": 1, "labels": config.LABELS, "members": members,
                "preprocessing": recipe}
    return {**registry, "model_version": identity(registry),
            "preprocessing_id": identity(recipe)}


def read_frame(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


class FullCacheSource:
    """Read acquired volumes; existing predict_members owns all preprocessing."""

    def __init__(self, settings):
        manifest = read_frame(settings["manifest"])
        if "mirrored" in manifest and manifest["mirrored"].fillna(False).astype(str).str.lower().isin(
                ["true", "1", "1.0"]).any():
            raise ValueError("application requires unmirrored source caches")
        index = dataset.StudyIndex(manifest, read_frame(settings["series_metadata"]),
                                   sides=read_frame(settings["sides"]) if settings["sides"] else None)
        self.rows = index.series
        self.shards = {key: np.load(path, mmap_mode="r", allow_pickle=False)
                       for key, path in settings["shards"].items()}

    def __call__(self, request):
        rows = self.rows[self.rows[config.ID_COL] == request.study_id].copy()
        volumes = []
        for _, row in rows.iterrows():
            position = row["position"]
            shard = self.shards[row["cache_path"]]
            if isinstance(position, (bool, np.bool_)) or not np.isfinite(position) or int(position) != position or not 0 <= position < len(shard):
                raise ValueError("invalid cache position")
            volume = np.asarray(shard[int(position)])
            if volume.dtype != np.uint8 or volume.shape != (config.CACHE_SLICES, config.CACHE_SIZE, config.CACHE_SIZE):
                raise ValueError("cache dtype or geometry mismatch")
            volumes.append(volume)
        return {config.ID_COL: request.study_id, "volumes": volumes, "rows": rows}


class EnsemblePredictor:
    def __init__(self, members, device):
        self.members, self.device = members, device
        weights = np.asarray([member["weight"] for member in members], dtype=np.float64)
        # Scaling first prevents overflow for large but finite weights.
        weights /= weights.max()
        self.weights = weights / weights.sum()

    def __call__(self, prepared):
        probabilities = artifacts.predict_members(self.members, prepared["volumes"],
                                                   prepared["rows"], self.device)
        return np.average(probabilities, axis=0, weights=self.weights)


def run_batch(settings_path, studies, output, device="cpu"):
    if not studies or any(not isinstance(uid, str) or not uid.strip() for uid in studies):
        raise ValueError("study IDs must be nonempty strings")
    if len(studies) != len(set(studies)):
        raise ValueError("duplicate study IDs")
    settings = load_settings(settings_path)
    registry = register_model(settings["ensemble"])
    output = Path(output)
    # A run directory is immutable; refuse accidental replacement of earlier results.
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
               "settings": settings, "device": device, "registry": registry,
               "study_ids": studies}
    receipt["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "timm", "numpy", "pandas"):
        try:
            receipt["environment"][package] = version(package)
        except PackageNotFoundError:
            receipt["environment"][package] = None

    def save_receipt():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "run.json")

    save_receipt()
    try:
        # Pin metadata and cache contents for reproducibility, including side decisions.
        inputs = [settings[key] for key in ("manifest", "series_metadata", "sides") if settings[key]]
        inputs += list(settings["shards"].values())
        receipt["input_sha256"] = {path: artifacts.sha256(path) for path in inputs}
        source = FullCacheSource(settings)
        members = artifacts.load_ensemble(settings["ensemble"], device)
        if register_model(settings["ensemble"]) != registry:
            raise ValueError("model artifacts changed during loading")
        app = Pipeline(model_version=registry["model_version"],
                       preprocessing_id=registry["preprocessing_id"], prepare=source,
                       predict=EnsemblePredictor(members, device), store=LocalResultStore(output / "results"))
        responses = []
        with (output / "responses.jsonl").open("x") as journal:
            for study in studies:
                request = {"schema_version": 1, config.ID_COL: study,
                           "request_id": identity([registry["model_version"], study, receipt["input_sha256"]]),
                           "model_version": registry["model_version"],
                           "preprocessing_id": registry["preprocessing_id"]}
                response = app.execute([request])[0]
                journal.write(json.dumps(response, allow_nan=False) + "\n")
                journal.flush()
                responses.append(response)
        failed = sum(row["status"] != "completed" for row in responses)
        receipt.update(status="failed" if failed else "completed", completed=len(responses) - failed,
                       failed=failed)
        if not failed:
            rows = [{config.ID_COL: row["result"][config.ID_COL], **row["result"]["probabilities"]}
                    for row in responses]
            pd.DataFrame(rows, columns=config.SUBMISSION_COLUMNS).to_csv(output / "predictions.csv", index=False)
        receipt["output_sha256"] = {str(path.relative_to(output)): artifacts.sha256(path)
                                    for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run.json"}
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_receipt()
    return receipt
