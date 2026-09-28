"""Package the user-confirmed 16/24-slice runs without retraining or uploading."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.model import artifacts, net, splits
from src.model.experiments import Experiment
from tools.compare_oof import read_predictions


def package(root, output, resume=False):
    root, output = Path(root), Path(output)
    sources = [(slices, fold, root / directory / f"fold{fold}" / f"lit_fold{fold}.ckpt")
               for slices, directory in [(16, "models_train"), (24, "experiment /slice_24")]
               for fold in range(5)]
    for _, _, source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing package: {output}")
    labels = pd.read_parquet(root / "data/meta/train_labels.parquet")
    folded = splits.make_folds(labels)
    gold = splits.load_gold_uids(root / "data/meta/train.csv")
    splits.assert_no_gold_leakage(folded[config.ID_COL], gold)
    output.mkdir(parents=True, exist_ok=resume)
    folded[[config.ID_COL, "fold"]].to_parquet(output / "reconstructed_folds.parquet", index=False)
    backbone = "convnext_tiny.fb_in22k_ft_in1k"
    trunk = net.build_backbone(backbone, pretrained=False)
    model = net.KneeNet(trunk, trunk.num_features)
    manifest, provenance = {"members": []}, []
    for slices, fold, source in sources:
        state = net.load_knee_state(source)
        if any(net.aggregation_defaults(state).values()):
            raise ValueError(f"Expected the legacy shared-pooling architecture: {source}")
        model.load_state_dict(state, strict=True)
        if any(not value.isfinite().all() for value in state.values() if value.is_floating_point()):
            raise ValueError(f"Non-finite checkpoint weights: {source}")
        oof = source.parent / f"oof_fold{fold}.parquet"
        oof_verified = False
        if oof.exists():
            predictions = read_predictions([oof], folded)
            expected = set(folded.loc[folded.fold == fold, config.ID_COL])
            if set(predictions.index) != expected:
                raise ValueError(f"Incomplete or incorrect validation membership: {oof}")
            oof_verified = True
        spec = Experiment(name=f"legacy_{slices}slice", backbone=backbone,
                          slices=slices, max_series=5).to_dict()
        spec["provenance"] = {
            "slice_count": "user confirmed 2026-09-09; both runs scored 0.906",
            "max_series": "five, from legacy project configuration; absent from checkpoint",
            "split": "reconstructed legacy split, NOT a verified training-membership record",
        }
        dest = output / f"slices_{slices}" / f"convnext_fold{fold}.pt"
        dest.parent.mkdir(exist_ok=True)
        if resume and Path(str(dest) + ".json").exists():
            metadata = artifacts.load_metadata(dest)
            if metadata["experiment"] != spec or metadata["fold"] != fold:
                raise ValueError(f"Existing export has different provenance: {dest}")
            exported = net.load_knee_state(dest)
            if exported.keys() != state.keys() or any(not torch.equal(exported[k], state[k]) for k in state):
                raise ValueError(f"Existing export differs from source: {dest}")
            del exported
        else:
            with dest.open("wb") as handle:
                torch.save(state, handle)
            # Do not represent reconstructed training UIDs as verified provenance.
            artifacts.write_metadata(dest, spec, fold, [], [])
        artifacts.load_metadata(dest)
        manifest["members"].append({"checkpoint": str(dest.relative_to(output)), "weight": 0.1})
        provenance.append({"source": str(source.relative_to(root)), "source_sha256": artifacts.sha256(source),
                           "checkpoint": str(dest.relative_to(output)), "fold": fold,
                           "slices": slices, "oof_membership_matches_reconstruction": oof_verified})
        print(f"Packaged {slices} slices, fold {fold}; OOF checked: {oof_verified}", flush=True)
        del state
    (output / "ensemble.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Fixed 50/50 family blend\n\n"
        "Ten selected checkpoints: five folds each at 16 and 24 slices.\n"
        "No EMA/SWA duplicates. All use five series and legacy uniform selection.\n"
        "Both families scored 0.906 separately; this blend has NOT been submitted.\n"
        "Point notebook 10 ENSEMBLE_JSON at this directory's ensemble.json.\n"
        "Keep the directory layout and all .pt.json sidecars intact.\n\n"
        "Legacy split membership is reconstructed, not established from training logs.\n"
        "Sidecar train_uids/val_uids are intentionally empty for these legacy exports.\n"
        "See provenance.json for OOF availability at packaging time.\n"
        "See docs/benchmark_16_24_results.md for subsequent recovery and runtime results.\n"
        "Benchmark 100 diverse studies on the submission GPU before submitting.\n"
        "A successful local CPU plumbing test is not a runtime or score guarantee.\n")
    print(f"Manifest: {output / 'ensemble.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=config.PROJECT_ROOT)
    parser.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "outputs/ensemble_16_24")
    parser.add_argument("--resume", action="store_true", help="Verify completed exports and rewrite incomplete ones")
    args = parser.parse_args()
    package(args.root, args.out, args.resume)
