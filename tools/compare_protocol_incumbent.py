"""Compare fixed protocol addition with the incumbent's same-fold OOF proxy."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.model import artifacts
from tools.compare_oof import compare, read_predictions


def fixed_predictions(families):
    if len(families) != 3:
        raise ValueError("Expected 16-slice, 24-slice and protocol families")
    a, b, p = [frame.sort_index() for frame in families]
    if not a.index.is_unique or not all(a.index.equals(f.index) for f in (b, p)):
        raise ValueError("Model families must contain exactly the same unique held-out studies")
    if not all(a["fold"].equals(f["fold"]) for f in (b, p)):
        raise ValueError("Model family fold assignments differ")
    for frame in (a, b, p):
        values = frame[config.LABELS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid prediction probabilities")
    baseline, candidate = a.copy(), a.copy()
    baseline[config.LABELS] = (a[config.LABELS] + b[config.LABELS]) / 2
    candidate[config.LABELS] = (a[config.LABELS] + b[config.LABELS] + p[config.LABELS]) / 3
    return baseline, candidate


def completed_protocol_directory(root, fold):
    if fold not in range(5):
        raise ValueError("Choose protocol folds from 0 through 4")
    suffix = "results" if fold in (1, 3) else "comparison_download"
    results = Path(root) / f"outputs/kaggle_protocol_fold{fold}_{suffix}"
    progress_name = "pair_progress.json" if fold in (1, 3) else "completion_progress.json"
    progress = json.loads((results / progress_name).read_text())
    if progress["state"] != "complete":
        raise ValueError(f"Protocol fold {fold} has not completed")
    if fold not in (1, 3) and (progress["fold"] != fold
            or f"protocol_fold{fold}" not in progress["completed"]):
        raise ValueError("Completion receipt belongs to a different or unfinished fold")
    return results / f"protocol_fold{fold}"


def evaluate(root, selected_folds, output, draws=1000):
    root, output = Path(root), Path(output)
    if not selected_folds or len(set(selected_folds)) != len(selected_folds):
        raise ValueError("Choose unique protocol folds")
    if not set(selected_folds) <= set(range(5)):
        raise ValueError("Choose protocol folds from 0 through 4")
    reference = json.loads((root / "outputs/kaggle_protocol_fold3_results/protocol_fold3/run_contract.json").read_text())
    package = root / "outputs/ensemble_16_24"
    manifest = json.loads((package / "ensemble.json").read_text())
    provenance = json.loads((package / "provenance.json").read_text())
    if len(manifest["members"]) != 10 or any(m["weight"] != .1 for m in manifest["members"]):
        raise ValueError("Incumbent is no longer the fixed ten-model ensemble")
    paths_by_family = {"slices_16": [], "slices_24": [], "protocol": []}
    receipts = []
    folded = pd.read_parquet(package / "reconstructed_folds.parquet")
    for fold in sorted(selected_folds):
        for slices in (16, 24):
            entries = [p for p in provenance if p["fold"] == fold and p["slices"] == slices]
            if len(entries) != 1:
                raise ValueError("Ambiguous incumbent provenance")
            entry = entries[0]
            if sum(m["checkpoint"] == entry["checkpoint"] for m in manifest["members"]) != 1:
                raise ValueError("Provenance checkpoint is not a unique incumbent member")
            checkpoint = package / entry["checkpoint"]
            metadata = artifacts.load_metadata(checkpoint)
            if metadata["fold"] != fold or metadata["experiment"]["slices"] != slices:
                raise ValueError("Wrong incumbent checkpoint geometry or fold")
            source = root / entry["source"]
            if artifacts.sha256(source) != entry["source_sha256"]:
                raise ValueError("Incumbent source checkpoint changed")
            oof = source.parent / f"oof_fold{fold}.parquet"
            paths_by_family[f"slices_{slices}"].append(oof)
            receipts.append({"family": f"slices_{slices}", "fold": fold,
                             "checkpoint_sha256": metadata["sha256"],
                             "oof": str(oof.relative_to(root)), "oof_sha256": artifacts.sha256(oof)})
        directory = completed_protocol_directory(root, fold)
        contract = json.loads((directory / "run_contract.json").read_text())
        if contract["fold"] != fold or any(contract.get(k) != v for k, v in reference.items()
                if k not in {"fold", "steps", "warmup_steps"}):
            raise ValueError("Protocol settings or initialization differ from screening")
        metadata = artifacts.load_metadata(directory / f"convnext_fold{fold}.pt")
        expected_val = set(folded.loc[folded.fold == fold, config.ID_COL])
        expected_train = set(folded.loc[folded.fold != fold, config.ID_COL])
        if (metadata["fold"] != fold or set(metadata["val_uids"]) != expected_val
                or set(metadata["train_uids"]) != expected_train
                or metadata["experiment"] != reference["experiment"]):
            raise ValueError("Protocol checkpoint split or experiment differs")
        saved = pd.read_parquet(directory / "folds.parquet")
        pd.testing.assert_series_equal(saved.set_index(config.ID_COL)["fold"].sort_index(),
                                       folded.set_index(config.ID_COL)["fold"].sort_index())
        oof = directory / f"oof_fold{fold}.parquet"
        paths_by_family["protocol"].append(oof)
        receipts.append({"family": "protocol", "fold": fold,
                         "checkpoint_sha256": metadata["sha256"],
                         "oof": str(oof.relative_to(root)), "oof_sha256": artifacts.sha256(oof)})
    frames = [read_predictions(paths, folded) for paths in paths_by_family.values()]
    expected = set(folded.loc[folded.fold.isin(selected_folds), config.ID_COL])
    if any(set(frame.index) != expected for frame in frames):
        raise ValueError("Incomplete held-out fold predictions")
    baseline, candidate = fixed_predictions(frames)
    table, summary = compare(pd.read_parquet(root / "data/meta/train_labels.parquet"),
                             baseline, candidate, draws=draws)
    summary.update(
        recipes={"baseline": "equal 16/24-slice families",
                 "candidate": "equal 16/24-slice/protocol families"},
        source_receipts=receipts,
        scope=f"Same-fold OOF proxy on {len(selected_folds)} protocol folds; not a public-score result. "
              "Legacy OOF/checkpoint association relies on original artifacts and the recorded recovery, "
              "not independently proven legacy training exclusion. The secondary blend_50_50 output "
              "is not a selected recipe; no weight search or classwise addition.")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "per_finding.csv", index_label="finding")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    baseline.reset_index().to_parquet(output / "incumbent_oof.parquet", index=False)
    candidate.reset_index().to_parquet(output / "protocol_addition_oof.parquet", index=False)
    print(table.round(5).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "source_receipts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "data/meta/compare_protocol_incumbent")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    evaluate(args.root, args.folds, args.out, args.draws)
