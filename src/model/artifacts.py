"""Explicit model provenance and inference for heterogeneous ensembles."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.model import dataset, net


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(checkpoint, experiment, fold, train_uids, val_uids):
    """Bind input geometry and split membership to the actual exported weights."""
    train_uids, val_uids = sorted(train_uids), sorted(val_uids)
    if set(train_uids) & set(val_uids):
        raise ValueError("training and validation studies overlap")
    payload = {
        "schema_version": 1, "labels": config.LABELS,
        "sha256": sha256(checkpoint), "experiment": experiment,
        "cache_size": config.CACHE_SIZE, "cache_slices": config.CACHE_SLICES,
        "fold": int(fold), "train_uids": train_uids, "val_uids": val_uids,
    }
    path = Path(str(checkpoint) + ".json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_metadata(checkpoint):
    path = Path(str(checkpoint) + ".json")
    data = json.loads(path.read_text())
    if data["schema_version"] != 1 or data["labels"] != config.LABELS:
        raise ValueError(f"unsupported schema or label order: {path}")
    if data["sha256"] != sha256(checkpoint):
        raise ValueError(f"checkpoint checksum mismatch: {checkpoint}")
    if data["cache_size"] != config.CACHE_SIZE or data["cache_slices"] != config.CACHE_SLICES:
        raise ValueError("checkpoint requires a different source cache geometry")
    spec = data["experiment"]
    if not 1 <= spec["slices"] <= config.CACHE_SLICES:
        raise ValueError("checkpoint slice count is outside the cache")
    dataset.select_series(pd.DataFrame(), spec["max_series"], spec["series_selection"])
    return data


def load_ensemble(manifest_path, device="cpu"):
    """Load exactly the members named in JSON, never a directory-wide glob.

    Format: {"members": [{"checkpoint": "run/fold0.pt", "weight": 1.0}]}.
    Paths are relative to this manifest. Each checkpoint requires its .pt.json
    sidecar, written by notebook 08. Weights are global, not per finding.
    """
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text())["members"]
    if not entries:
        raise ValueError("ensemble has no members")
    members, digests, identities = [], set(), set()
    for entry in entries:
        path = (manifest_path.parent / entry["checkpoint"]).resolve()
        weight = float(entry.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("ensemble weights must be finite and positive")
        metadata = load_metadata(path)
        identity = (metadata["experiment"]["name"], metadata["fold"])
        if metadata["sha256"] in digests or identity in identities:
            raise ValueError("duplicate checkpoint or experiment/fold in ensemble")
        digests.add(metadata["sha256"])
        identities.add(identity)
        spec = metadata["experiment"]
        state = net.load_knee_state(path)
        trunk = net.build_backbone(spec["backbone"], pretrained=False)
        defaults = net.backbone_defaults(spec["backbone"])
        model = net.KneeNet(trunk, trunk.num_features,
                            input_mean=defaults["input_mean"],
                            input_std=defaults["input_std"],
                            **net.aggregation_defaults(state))
        model.load_state_dict(state)
        members.append({"model": model.to(device).eval(), "spec": spec,
                        "weight": weight, "metadata": metadata})
    return members


@torch.inference_mode()
def predict_members(members, volumes, rows, device="cpu"):
    """Predict from full acquired volumes and one aligned metadata row per series."""
    if len(rows) != len(volumes) or not len(rows):
        raise ValueError("volumes and metadata must be nonempty and aligned")
    order = np.argsort(rows["SeriesInstanceUID"].astype(str).to_numpy(), kind="stable")
    rows = rows.iloc[order].reset_index(drop=True)
    volumes = [volumes[i] for i in order]
    outputs = []
    for member in members:
        spec = member["spec"]
        chosen = dataset.select_series(rows, spec["max_series"], spec["series_selection"])
        picked = rows.iloc[chosen]
        stack = []
        for i, (_, row) in zip(chosen, picked.iterrows()):
            volume = dataset.mirror_volume(volumes[i], row.get("side_predicted"),
                                            row.get("Anatomical_Plane"))
            if volume.shape != (config.CACHE_SLICES, config.CACHE_SIZE, config.CACHE_SIZE):
                raise ValueError(f"unexpected volume geometry: {volume.shape}")
            keep = np.linspace(0, len(volume) - 1, spec["slices"]).round().astype(int)
            stack.append(volume[keep])
        tensor = torch.from_numpy(np.stack(stack)).unsqueeze(0).to(device)
        meta = torch.from_numpy(np.stack([
            dataset.series_metadata(row) for _, row in picked.iterrows()
        ])).unsqueeze(0).to(device)
        mask = torch.ones(1, len(stack), dtype=torch.bool, device=device)
        device_type = torch.device(device).type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            logits = member["model"](tensor, meta, mask)
        probs = logits.float().sigmoid().cpu().numpy()[0]
        if not np.isfinite(probs).all():
            raise ValueError("non-finite model prediction")
        outputs.append(probs)
    return np.stack(outputs)
