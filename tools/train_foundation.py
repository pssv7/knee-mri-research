"""Preflight, smoke, or train one explicitly selected foundation-model arm.

Uses the unchanged nested input preflight. No gold data, automatic follow-up
training, public-score selection, or submission is supported.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning.pytorch as pl
import timm

from src.model import dataset, lightning as knee, foundation as fm
from src.validation.proxy import ID, LABELS, sha, write_json_new, validate_prediction_provenance
from tools.train_nested_proxy_control import preflight


def verify_experiment(experiment, root=ROOT):
    experiment = Path(experiment)
    spec = json.loads((experiment / "experiment.json").read_text())
    if spec.get("schema_version") != 1 or spec.get("stage") not in {"frozen", "finetune"}:
        raise ValueError("Unknown foundation experiment schema/stage")
    if not spec.get("runtime_code_sha256"):
        raise ValueError("Missing source provenance")
    for name, expected in spec["runtime_code_sha256"].items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"Experiment runtime changed: {name}; prepare a new experiment")
    baseline = json.loads((experiment / "baseline/spec.json").read_text())
    for key in ["roles_sha256", "proxy_labels_sha256"]:
        if spec[key] != baseline[key]:
            raise ValueError("Foundation and baseline designs differ")
    return spec


class FeatureDataset(Dataset):
    """Completed, hash-bound slice features; training does not reread MRI pixels."""

    def __init__(self, directory, expected_binding):
        self.directory = Path(directory)
        receipt = json.loads((self.directory / "features.json").read_text())
        if receipt["binding"] != expected_binding:
            raise ValueError("Feature cache belongs to another encoder, input, or split")
        for name, digest in receipt["sha256"].items():
            if sha(self.directory / name) != digest:
                raise ValueError("Feature cache bytes changed")
        self.uids = receipt["uids"]
        if len(self.uids) != len(set(self.uids)):
            raise ValueError("Duplicate feature study")
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
                       for name in ["features", "meta", "count", "targets", "weights"]}

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        n = int(self.arrays["count"][index])
        return {ID: self.uids[index], "volumes": self.arrays["features"][index, :n],
                "meta": self.arrays["meta"][index, :n],
                "targets": self.arrays["targets"][index], "weights": self.arrays["weights"][index]}


def extract_cache(model, source, out, binding, device="cuda"):
    """Extract only the supplied role; caller supplies train/selection before fit."""
    from tqdm.auto import tqdm
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    n = len(source)
    shapes = {"features": (n, 5, 16, fm.ENCODER_DIM),
              "meta": (n, 5, len(dataset.META_COLUMNS)), "count": (n,),
              "targets": (n, len(LABELS)), "weights": (n, len(LABELS))}
    arrays = {key: np.lib.format.open_memmap(out / f"{key}.npy", mode="w+",
              dtype=np.int64 if key == "count" else np.float32, shape=shape)
              for key, shape in shapes.items()}
    uids = []
    model.to(device).eval()
    with torch.inference_mode():
        for i in tqdm(range(n), desc=f"Extract {out.name}"):
            item = source[i]
            volumes = torch.as_tensor(item["volumes"], device=device)
            count = len(volumes)
            if not 1 <= count <= 5 or volumes.shape[1:] != (16, 224, 224):
                raise ValueError("Unexpected selected volume geometry")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                features = model.extract_slices(volumes).float().cpu().numpy()
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite foundation features")
            arrays["features"][i, :count] = features
            arrays["meta"][i, :count] = item["meta"]
            arrays["count"][i] = count
            for key in ["targets", "weights"]:
                arrays[key][i] = item[key]
            uids.append(item[ID])
    for value in arrays.values():
        value.flush()
    write_json_new(out / "features.json", {"binding": binding, "uids": uids,
                   "sha256": {f"{key}.npy": sha(out / f"{key}.npy") for key in shapes}})
    return FeatureDataset(out, binding)


def make_module(model, recipe, total_steps):
    module = knee.KneeLitModule(torch.nn.Identity(), model.feature_dim,
        total_steps=total_steps, lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"], backbone_lr_scale=recipe["backbone_lr_scale"],
        max_lr_multiplier=recipe["max_lr_multiplier"], schedule=recipe["schedule"],
        warmup_steps=int(round(total_steps / recipe["epochs"] * recipe["warmup_epochs"])))
    module.model = model
    return module


class FiniteSteps(pl.Callback):
    def __init__(self):
        self.steps = 0
        self.losses = []

    def on_before_optimizer_step(self, trainer, module, optimizer):
        gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or nonfinite gradients")
        self.steps += 1

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        self.losses.append(float(loss.detach().cpu()))


def load_selected(model, selected, stage):
    state = torch.load(selected, map_location="cpu", weights_only=True)
    if stage == "frozen":
        head = fm.FoundationKneeNet(cached=True)
        head.load_state_dict(state, strict=True)
        combined = model.state_dict()
        combined.update(state)
        model.load_state_dict(combined, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return model


def predict(model, source, device="cuda"):
    rows = []
    model.to(device).eval()
    loader = DataLoader(source, batch_size=1, num_workers=0, collate_fn=dataset.collate_studies)
    with torch.inference_mode():
        for batch in loader:
            args = [batch[key].to(device) for key in ("volumes", "meta", "mask")]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                scores = model(*args).float().sigmoid().cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("Nonfinite predictions")
            rows.extend({ID: uid, **dict(zip(LABELS, map(float, values)))}
                        for uid, values in zip(batch[ID], scores))
    return pd.DataFrame(rows)


def verify_smoke(receipt, spec, experiment_hash, inputs_hash, encoder_hash, arm, versions):
    if not receipt:
        raise ValueError("Training requires a passing GPU smoke receipt")
    checks = {"status": "passed", "arm": arm, "stage": spec["stage"],
              "experiment_sha256": experiment_hash, "input_config_sha256": inputs_hash,
              "encoder_sha256": encoder_hash, "runtime_versions": versions, "optimizer_steps": 2}
    for key, expected in checks.items():
        if receipt.get(key) != expected:
            raise ValueError(f"GPU smoke is missing or incompatible: {key}")
    if receipt.get("roundtrip_max_abs", float("inf")) > 1e-5:
        raise ValueError("GPU smoke reload mismatch")


def run(experiment, inputs, repo, weights, arm, out=None, action="preflight", smoke_receipt=None):
    experiment = Path(experiment)
    spec = verify_experiment(experiment)
    if arm not in spec["arms"]:
        raise ValueError("Unknown comparison arm")
    base, roles, labels, manifest, series, sides, paths, _ = preflight(experiment / "baseline", inputs)
    recipe = spec["recipe"]
    lock = ROOT / "configs/orthofoundation_assets.json"
    fm.verify_source(repo, json.loads(lock.read_text()))
    digest = sha(weights)
    if arm == "orthofoundation" and digest != fm.ORTHO_SHA256:
        raise ValueError("Wrong OrthoFoundation checkpoint")
    if arm == "dinov3" and not digest.startswith("8aa4cbdd"):
        raise ValueError("Need official DINOv3-L LVD1689M checkpoint 8aa4cbdd")
    if action == "preflight":
        return {"status": "data_and_asset_preflight_passed", "counts": roles.role.value_counts().to_dict(),
                "encoder_sha256": digest, "training_launched": False}
    if action not in {"smoke", "train"} or out is None:
        raise ValueError("Choose preflight, smoke, or train with a fresh output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU training campaign is started")
    versions = {"torch": torch.__version__, "timm": timm.__version__, "lightning": pl.__version__,
                "gpu": torch.cuda.get_device_name(), "devices": 1, "precision": recipe["precision"]}
    if action == "train":
        receipt = json.loads(Path(smoke_receipt).read_text()) if smoke_receipt else None
        verify_smoke(receipt, spec, sha(experiment / "experiment.json"), sha(inputs), digest, arm, versions)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(spec["seed"], workers=True)
    model = fm.make_model(repo, weights, arm, lock, seed=spec["seed"],
                          frozen=spec["stage"] == "frozen", chunk_size=recipe["slice_chunk"])
    initial_head = fm.state_sha(fm.head_state(model))
    index = dataset.StudyIndex(manifest, series, labels, sides, max_series=5, series_selection="protocol")
    if set(index.studies) != set(roles[ID]):
        raise ValueError("Dataset dropped studies")

    def volumes(role, augment=None):
        if augment is None:
            augment = role == "train" and spec["stage"] == "finetune"
        return knee.StudyVolumeDataset(index, sorted(roles.loc[roles.role == role, ID]), paths, 16,
            augment=augment, resample=False, duplicate_free=False)

    train_images = volumes("train")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    binding = {"encoder_sha256": digest, "inputs_sha256": sha(inputs),
               "experiment_sha256": sha(experiment / "experiment.json"), "role": "train"}

    if action == "smoke":
        chosen = []
        for i in range(len(train_images)):
            if len(train_images[i]["volumes"]) == 5:
                chosen.append(i)
                if len(chosen) == 2:
                    break
        if len(chosen) != 2:
            raise RuntimeError("Need two training studies with five series")
        training_data = Subset(train_images, chosen)
    else:
        training_data = train_images

    if spec["stage"] == "frozen":
        train_data = extract_cache(model, training_data, out / "train_features", binding)
        selection_data = None if action == "smoke" else extract_cache(
            model, volumes("selection"), out / "selection_features", {**binding, "role": "selection"})
        model.cpu()
        head = fm.FoundationKneeNet(cached=True)
        head.load_state_dict(fm.head_state(model), strict=True)
        train_model = head
    else:
        train_data, selection_data, train_model = training_data, None, model
        if action == "train":
            selection_data = volumes("selection")
    if action == "smoke":
        train_data = Subset(train_data, list(range(len(train_data))) * 8)
    steps = knee.optimiser_steps(len(train_images), 1, 1, 8, recipe["epochs"])
    module = make_module(train_model, recipe, steps)
    # Reset after all construction/extraction: head training RNG is paired too.
    pl.seed_everything(spec["seed"], workers=True)
    loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=dataset.collate_studies)
    finite = FiniteSteps()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(out / "checkpoints"), filename="selected",
        monitor="val_macro", mode="max", save_top_k=1, save_last=True)
    trainer = pl.Trainer(max_epochs=1 if action == "smoke" else recipe["epochs"],
        max_steps=2 if action == "smoke" else -1, accelerator="gpu", devices=1,
        precision=recipe["precision"], accumulate_grad_batches=8,
        callbacks=[finite] if action == "smoke" else [checkpoint, finite],
        enable_checkpointing=action != "smoke", enable_model_summary=False,
        logger=False if action == "smoke" else pl.loggers.CSVLogger(str(out), name="training"),
        limit_val_batches=0 if action == "smoke" else 1.0,
        num_sanity_val_steps=0)
    select_loader = None if selection_data is None else DataLoader(selection_data,
        batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_studies)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=select_loader)
    if action == "smoke":
        if finite.steps != 2 or len(finite.losses) != 16:
            raise RuntimeError("Smoke must finish exactly two optimizer steps / sixteen microbatches")
        selected = out / "transient.pt"
        torch.save(module.model.state_dict(), selected)
    else:
        if not checkpoint.best_model_path or trainer.current_epoch < recipe["epochs"]:
            raise RuntimeError("Training did not complete all prescribed epochs")
        selected = out / "selected.pt"
        knee.export_from_checkpoint(checkpoint.best_model_path, str(selected))

    # Reload head weights into the image path, including preprocessing buffers.
    module.cpu()
    load_selected(model, selected, spec["stage"])
    # Reload checks must see identical pixels, including in the fine-tuning arm.
    check_source = Subset(volumes("train", augment=False), [0])
    before = predict(model, check_source)[LABELS].to_numpy()
    model.cpu()
    torch.cuda.empty_cache()
    restored = fm.make_model(repo, weights, arm, lock, seed=spec["seed"], frozen=True,
                             chunk_size=recipe["slice_chunk"])
    load_selected(restored, selected, spec["stage"])
    after = predict(restored, check_source)[LABELS].to_numpy()
    difference = float(np.max(np.abs(before - after)))
    if not np.allclose(before, after, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Fresh encoder/checkpoint reload changed predictions")
    if action == "smoke":
        selected.unlink()
        receipt = {"status": "passed", "arm": arm, "stage": spec["stage"],
            "optimizer_steps": finite.steps, "finite_losses": finite.losses,
            "roundtrip_max_abs": difference, "runtime_versions": versions,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "elapsed_seconds": time.monotonic() - started, "encoder_sha256": digest,
            "initial_head_sha256": initial_head, "experiment_sha256": binding["experiment_sha256"],
            "input_config_sha256": sha(inputs),
            "image_gain_claim_allowed": False, "trained_weights_retained": False}
        write_json_new(out / "smoke_receipt.json", receipt)
        return receipt

    # First construction/read of outer image dataset follows checkpoint selection.
    predictions = predict(restored, volumes("outer"))
    predictions.to_parquet(out / "outer_predictions.parquet", index=False)
    contract = {**base, "foundation_schema_version": 1, "arm": arm, "stage": spec["stage"],
        "seed": spec["seed"], "training_recipe": recipe, "runtime_versions": versions,
        "experiment_sha256": binding["experiment_sha256"], "encoder_sha256": digest,
        "initial_head_sha256": initial_head, "input_config_sha256": sha(inputs),
        "checkpoint_sha256": sha(selected), "outer_predictions_sha256": sha(out / "outer_predictions.parquet"),
        "runtime_code_sha256": spec["runtime_code_sha256"], "roundtrip_max_abs": difference,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "backbone": "dinov3_vitl16", "encoder_origin": arm,
        "checkpoint_contents": "aggregation_head" if spec["stage"] == "frozen" else "complete_image_model",
        **{f"{role}_uids": sorted(roles.loc[roles.role == role, ID]) for role in ["train", "selection", "outer"]}}
    validate_prediction_provenance(roles, contract, predictions)
    write_json_new(out / "run_contract.json", contract)
    return {"status": "one_arm_completed", "arm": arm, "stage": spec["stage"],
            "output": str(out), "image_gain_claim_allowed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["experiment", "inputs", "repo", "weights"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--arm", choices=["orthofoundation", "dinov3"], required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--action", choices=["preflight", "smoke", "train"], default="preflight")
    a = p.parse_args()
    print(json.dumps(run(a.experiment, a.inputs, a.repo, a.weights, a.arm, a.out, a.action, a.smoke_receipt), indent=2))
