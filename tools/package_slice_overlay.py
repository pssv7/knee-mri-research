"""Package the fixed 15-member incumbent plus three validated slice transformers.

The three transformers are new votes. The fold-1 and fold-3 control checkpoints
are already byte-identical protocol members of the incumbent, so they must not
be added again. Every one of the resulting 18 members receives weight 1/18.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import artifacts


ROOT = Path(__file__).resolve().parent.parent
DATASET = "pssv27/knee-ensemble-protocol-slice-overlay-private"
INCUMBENT = ROOT / "outputs/ensemble_16_24_protocol"
OUTPUT = ROOT / "outputs/ensemble_16_24_protocol_slice_overlay"
SOURCES = {
    1: ROOT / "outputs/slice_overlay_staging/fold1/slice_transformer_fold1/convnext_fold1.pt",
    2: ROOT / "outputs/fold2/convnext_fold2.pt",
    3: ROOT / "outputs/slice_overlay_staging/fold3/slice_transformer_fold3/convnext_fold3.pt",
}
CONTROL_HASHES = {
    1: "64b0d2515b3d214bf8175de3d083dd388794ff1013fb01a93d0a7c819022664c",
    3: "347bb18b82d5259d65e3e80574ad4e07af5b9de9a126f46a135deb6b9ddf185f",
}


def link_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if artifacts.sha256(source) != artifacts.sha256(destination):
            raise ValueError(f"existing package file differs: {destination}")
    else:
        destination.hardlink_to(source)


def check_candidate(fold: int, path: Path) -> dict:
    meta = artifacts.load_metadata(path)
    spec = meta["experiment"]
    expected = {
        "name": "slice_transformer", "backbone": "convnext_tiny.fb_in22k_ft_in1k",
        "slices": 16, "max_series": 5, "series_selection": "protocol",
        "resample": False, "classwise_pool": False, "slice_transformer": True,
    }
    if meta["fold"] != fold or spec != expected:
        raise ValueError(f"unexpected slice-transformer provenance: {path}")
    if len(meta["train_uids"]) + len(meta["val_uids"]) != 4349:
        raise ValueError(f"incomplete split provenance: {path}")
    return meta


def main() -> None:
    incumbent_manifest = json.loads((INCUMBENT / "ensemble.json").read_text())
    incumbent = incumbent_manifest["members"]
    if len(incumbent) != 15 or any(entry["weight"] != 1 / 15 for entry in incumbent):
        raise ValueError("expected the fixed fifteen-member incumbent")

    incumbent_meta = {}
    for entry in incumbent:
        path = INCUMBENT / entry["checkpoint"]
        meta = artifacts.load_metadata(path)
        key = (meta["experiment"]["name"], meta["fold"])
        if key in incumbent_meta:
            raise ValueError(f"duplicate incumbent member: {key}")
        incumbent_meta[key] = (path, meta)
    for fold, digest in CONTROL_HASHES.items():
        path, meta = incumbent_meta["protocol", fold]
        if meta["sha256"] != digest:
            raise ValueError(f"fold {fold} control is not the paired baseline: {path}")

    candidates = {fold: check_candidate(fold, path) for fold, path in SOURCES.items()}
    if len({meta["sha256"] for meta in candidates.values()}) != 3:
        raise ValueError("duplicate slice-transformer checkpoint")

    entries = []
    for entry in incumbent:
        source = INCUMBENT / entry["checkpoint"]
        meta = artifacts.load_metadata(source)
        entries.append((source, meta, entry["checkpoint"]))
    for fold, source in SOURCES.items():
        entries.append((source, candidates[fold], f"slice_transformer_fold{fold}.pt"))

    if len(entries) != 18:
        raise AssertionError("wrong overlay member count")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = []
    manifest = {"members": []}
    for source, meta, filename in entries:
        for suffix in ("", ".json"):
            link_verified(Path(str(source) + suffix), OUTPUT / (filename + suffix))
        manifest["members"].append({"checkpoint": filename, "weight": 1 / 18})
        provenance.append({"source": str(source.relative_to(ROOT)), "checkpoint": filename,
                           "sha256": meta["sha256"], "experiment": meta["experiment"],
                           "fold": meta["fold"]})
    (OUTPUT / "ensemble.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "recipe": "fixed equal 18-member overlay; no score-derived weight tuning",
        "incumbent_members": 15, "slice_transformer_members": 3,
        "why_no_controls": "fold 1/3 paired controls are incumbent protocol members; fold 2 control was not the incumbent hash",
        "members": provenance,
    }, indent=2) + "\n")
    (OUTPUT / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET,
        "title": "Knee Ensemble Protocol Slice Overlay Private",
        "licenses": [{"name": "other"}],
        "description": "Fixed 18-member overlay: the 15-member 0.914 incumbent plus selected slice-transformer folds 1–3.",
    }, indent=2) + "\n")
    print(f"Packaged {len(entries)} hard-linked members: {OUTPUT}")


if __name__ == "__main__":
    main()
