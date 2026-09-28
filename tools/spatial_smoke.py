"""Measure one full cycle of the spatial model against its 2D control.

The plan commits to measuring "a complete optimizer update, validation,
checkpoint export/reload and inference" before any full training, because the
cost of the spatial blocks is unknown until something runs them. This does
exactly that at whatever scale the machine allows: the same script produces
the local correctness evidence on CPU and the A100 timing and peak-memory
numbers the campaign estimate needs.

What it asserts, per arm:

    forward and backward over real 24-slice geometry, finite throughout
    the optimiser actually steps (parameters move, no NaN)
    a validation pass under inference mode
    export to a bare KneeNet state dict, reload, and identical scores
    provenance metadata that `artifacts.load_metadata` accepts

And across the two arms, that a freshly built spatial model and its control
agree to numerical tolerance -- the zero-initialised residual is what makes
the screening comparison a test of the 3D blocks rather than of a different
starting point. That check is skipped once weights have been trained.

Timings are one measured step, not a benchmark: they bound the campaign
estimate and nothing finer.
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.model import artifacts, experiments, net
from src.model.dataset import META_COLUMNS
from src.model.lightning import KneeLitModule, export_state_dict


def synthetic_batch(studies, series, slices, size, device, seed=0):
    """Volumes in the cache's own dtype and geometry, with one plane per series.

    Synthetic pixels are fine here: this measures the shapes, the memory and
    the plumbing, none of which can tell a knee from noise.
    """
    generator = torch.Generator().manual_seed(seed)
    volumes = torch.randint(0, 255, (studies, series, slices, size, size),
                            dtype=torch.uint8, generator=generator)
    meta = torch.zeros(studies, series, len(META_COLUMNS))
    planes = [META_COLUMNS.index(f"is_{name}")
              for name in ("sagittal", "coronal", "axial")]
    for i in range(series):
        meta[:, i, planes[i % len(planes)]] = 1.0
    mask = torch.ones(studies, series, dtype=torch.bool)
    if series > 1:
        # One study short a series, so masking is exercised on every run.
        mask[0, -1] = False
    targets = (torch.rand(studies, len(config.LABELS), generator=generator))
    weights = torch.ones_like(targets)
    return {config.ID_COL: [f"study{i}" for i in range(studies)],
            "volumes": volumes.to(device), "meta": meta.to(device),
            "mask": mask.to(device), "targets": targets.to(device),
            "weights": weights.to(device)}


def build(experiment, pretrained, device):
    spec = experiments.get_experiment(experiment)
    trunk = net.build_backbone(spec.backbone, pretrained=pretrained)
    defaults = net.backbone_defaults(spec.backbone)
    module = KneeLitModule(
        trunk, trunk.num_features, total_steps=4,
        spatial_encoder=spec.spatial_encoder,
        plane_adapters=spec.plane_adapters,
        classwise_pool=spec.classwise_pool,
        slice_transformer=spec.slice_transformer, **defaults)
    return spec, module.to(device)


def elapsed(device, started):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def run_arm(experiment, args, device, out_dir):
    spec, module = build(experiment, args.pretrained, device)
    batch = synthetic_batch(args.studies, args.series, spec.slices,
                            config.CACHE_SIZE, device, seed=args.seed)
    if getattr(args, 'full_series', False):
        batch['mask'].fill_(True)
    mixed = getattr(args, 'precision', '32') == '16-mixed'
    accumulation = getattr(args, 'accumulation', 1)
    def precision_context():
        return torch.autocast('cuda', dtype=torch.float16) if mixed else contextlib.nullcontext()
    trainable = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    added = sum(p.numel() for name, p in module.model.named_parameters()
                if name.startswith(("spatial_encoder.", "study_pool.adapters.",
                                    "study_pool.pools.", "study_pool.fuse.",
                                    "study_pool.plane_embedding")))
    optimiser = torch.optim.AdamW(module.parameters(), lr=1e-4)

    # Scored before anything trains: the arms are only comparable at
    # initialisation, since one optimiser step moves two different parameter
    # sets and they are expected to diverge from there.
    module.eval()
    with torch.inference_mode(), precision_context():
        initial = module(batch["volumes"], batch["meta"], batch["mask"])
    if not torch.isfinite(initial).all():
        raise ValueError(f"{experiment}: non-finite logits at initialisation")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    module.train()
    started = time.perf_counter()
    before = [p.detach().cpu().clone() for p in module.model.parameters()]
    scaler = torch.amp.GradScaler('cuda', enabled=mixed)
    for microbatch in range(accumulation):
        with precision_context():
            loss = module.training_step(batch, microbatch) / accumulation
        if not torch.isfinite(loss):
            raise ValueError(f"{experiment}: non-finite training loss")
        scaler.scale(loss).backward()
    scaler.step(optimiser)
    scaler.update()
    optimiser.zero_grad(set_to_none=True)
    step_seconds = elapsed(device, started)
    moved = sum(not torch.equal(a, b.detach().cpu())
                for a, b in zip(before, module.model.parameters()))
    if not moved:
        raise ValueError(f"{experiment}: the optimiser changed nothing")
    if any(not torch.isfinite(p).all() for p in module.model.parameters()):
        raise ValueError(f"{experiment}: non-finite parameters after one step")
    training_peak = (torch.cuda.max_memory_allocated(device) / 2**30
                     if device.type == "cuda" else None)
    del before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    module.eval()
    started = time.perf_counter()
    with torch.inference_mode(), precision_context():
        logits = module(batch["volumes"], batch["meta"], batch["mask"])
    validation_seconds = elapsed(device, started)
    validation_peak = (torch.cuda.max_memory_allocated(device) / 2**30
                       if device.type == "cuda" else None)
    if not torch.isfinite(logits).all():
        raise ValueError(f"{experiment}: non-finite validation logits")

    # Export, reload into a bare KneeNet rebuilt from the weights alone, and
    # require the same scores. This is the path the submission takes, and the
    # place two earlier checkpoints quietly changed meaning.
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{experiment}.pt"
    export_state_dict(module, str(checkpoint))
    logits = logits.float().cpu()
    initial = initial.float().cpu()
    del module, optimiser, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    state = net.load_knee_state(checkpoint)
    defaults = net.aggregation_defaults(state)
    if defaults["spatial_encoder"] != spec.spatial_encoder:
        raise ValueError(f"{experiment}: the spatial blocks did not survive export")
    if defaults["plane_adapters"] != spec.plane_adapters:
        raise ValueError(f"{experiment}: the plane adapters did not survive export")
    trunk = net.build_backbone(spec.backbone, pretrained=False)
    normalisation = {key: net.backbone_defaults(spec.backbone)[key]
                     for key in ("input_mean", "input_std")}
    reloaded = net.KneeNet(trunk, trunk.num_features, **normalisation, **defaults)
    reloaded.load_state_dict(state, strict=True)
    reloaded = reloaded.to(device).eval()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), precision_context():
        again = reloaded(batch["volumes"], batch["meta"], batch["mask"])
    inference_seconds = elapsed(device, started)
    gap = (logits - again.float().cpu()).abs().max().item()
    if gap > 1e-4:
        raise ValueError(f"{experiment}: reloaded weights score differently ({gap})")

    artifacts.write_metadata(checkpoint, spec.to_dict(), fold=1,
                             train_uids=["train0"], val_uids=["val0"])
    artifacts.load_metadata(checkpoint)

    return {"experiment": experiment, "slices": spec.slices,
            "trainable_parameters": trainable, "added_parameters": added,
            "train_step_seconds": round(step_seconds, 3),
            "validation_seconds": round(validation_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "reload_max_logit_gap": gap,
            "training_peak_memory_gib": training_peak,
            "validation_peak_memory_gib": validation_peak,
            "inference_peak_memory_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                          if device.type == 'cuda' else None),
            "precision": getattr(args, 'precision', '32'),
            "accumulation": accumulation,
            "effective_batch": args.studies * accumulation,
            "training_memory_gate_passed": training_peak <= 36 if training_peak is not None else None,
            "peak_memory_gib": (round(torch.cuda.max_memory_allocated(device)
                                      / 2**30, 3) if device.type == "cuda" else None),
            "initial_logits": initial.float().cpu().numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["spatial_control", "spatial"])
    parser.add_argument("--studies", type=int, default=1)
    parser.add_argument("--series", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true",
                        help="download timm weights; off by default so this runs offline")
    parser.add_argument('--precision', choices=('32', '16-mixed'), default='32')
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--full-series', action='store_true')
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/spatial_smoke")
    args = parser.parse_args()
    if args.accumulation < 1 or (args.precision == '16-mixed' and not args.device.startswith('cuda')):
        parser.error('Positive accumulation required; mixed precision requires CUDA')

    device = torch.device(args.device)
    out_dir = Path(args.out)
    results, logits = [], {}
    for experiment in args.arms:
        torch.manual_seed(args.seed)
        measured = run_arm(experiment, args, device, out_dir)
        logits[experiment] = measured.pop("initial_logits")
        results.append(measured)
        print(json.dumps(measured, indent=2), flush=True)

    report = {"device": args.device, "arms": results,
              "batch": {"studies": args.studies, "series": args.series,
                        "size": config.CACHE_SIZE},
              "pretrained_backbone": args.pretrained}
    if {"spatial_control", "spatial"} <= set(logits):
        gap = float(np.abs(logits["spatial_control"] - logits["spatial"]).max())
        report["control_vs_spatial_initial_max_logit_gap"] = gap
        report["identical_at_initialisation"] = gap < 1e-4
        if gap >= 1e-4:
            raise ValueError(
                "an untrained spatial model must reproduce its control; "
                f"maximum logit gap {gap}")
    if len(results) == 2:
        report["train_step_ratio"] = round(
            results[1]["train_step_seconds"] / results[0]["train_step_seconds"], 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r['training_memory_gate_passed'] is False for r in results):
        raise RuntimeError('Representative training exceeds the 36-GiB campaign gate')
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
