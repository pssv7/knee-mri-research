"""Create the small Kaggle dataset for the slice-overlay's three new weights."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import artifacts
from tools.package_slice_overlay import INCUMBENT, SOURCES

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/kaggle_slice_overlay_upload"
DATASET = "pssv27/knee-slice-transformer-overlay-private"
INCUMBENT_MOUNT = "knee-ensemble-protocol-private"


def link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.hardlink_to(source)
    elif artifacts.sha256(source) != artifacts.sha256(dest):
        raise ValueError(f"stale upload artifact: {dest}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    incumbent = json.loads((INCUMBENT / "ensemble.json").read_text())["members"]
    if len(incumbent) != 15:
        raise ValueError("expected fifteen-member incumbent")
    members, provenance = [], []
    for entry in incumbent:
        members.append({"checkpoint": f"../{INCUMBENT_MOUNT}/{entry['checkpoint']}", "weight": 1 / 18})
    for fold, source in SOURCES.items():
        meta = artifacts.load_metadata(source)
        filename = f"slice_transformer_fold{fold}.pt"
        for suffix in ("", ".json"):
            link(Path(str(source) + suffix), OUTPUT / (filename + suffix))
        members.append({"checkpoint": filename, "weight": 1 / 18})
        provenance.append({"fold": fold, "checkpoint": filename, "sha256": meta["sha256"],
                           "source": str(source.relative_to(ROOT))})
    if len(members) != 18:
        raise AssertionError("incorrect overlay member count")
    (OUTPUT / "overlay.json").write_text(json.dumps({"members": members}, indent=2) + "\n")
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "recipe": "fixed 18 equal members: 15 incumbent plus 3 slice transformers",
        "incumbent_dataset": "pssv27/knee-ensemble-protocol-private",
        "slice_members": provenance,
    }, indent=2) + "\n")
    (OUTPUT / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET, "title": "Knee Slice Transformer Overlay Private",
        "licenses": [{"name": "other"}],
        "description": "Three verified slice-transformer checkpoints and an overlay manifest referencing the existing private incumbent dataset.",
    }, indent=2) + "\n")
    print(f"Prepared {OUTPUT}")


if __name__ == "__main__":
    main()
