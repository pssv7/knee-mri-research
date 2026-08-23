"""Run the submission notebook end to end against a fake test set.

This is the notebook whose failure is most expensive: it runs once, against a
hidden test set, with no chance to iterate. So it is executed here for real --
DICOM in, submission.csv out -- with a miniature test tree built from the dev
series and randomly initialised checkpoints standing in for the trained ones.

    python3 tools/check_submission.py

What this catches: import errors, shape mismatches, the fallback path, header
mismatches, and anything that only appears when the cells run in sequence.
What it cannot catch: whether the predictions are any good.
"""

import json
import os
import pathlib
import sys
import tempfile
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV = ROOT / "data" / "dev"

N_STUDIES = 3
SERIES_PER_STUDY = 2


def build_fake_competition(base: pathlib.Path):
    """A miniature competition input tree: test DICOMs plus the three CSVs."""
    import pandas as pd
    import pydicom

    source = sorted(DEV.glob("*.dcm"))
    if not source:
        raise SystemExit("no dev DICOM found -- see docs/dev_data.md")

    sys.path.insert(0, str(ROOT))
    from src import config

    studies, series_rows = [], []
    for study in range(N_STUDIES):
        study_uid = f"1.2.826.0.1.3680043.8.498.fake.study{study}"
        studies.append(study_uid)
        for series in range(SERIES_PER_STUDY):
            series_uid = f"{study_uid}.series{series}"
            target = base / "test_series" / study_uid / series_uid
            target.mkdir(parents=True, exist_ok=True)
            for path in source[:8]:      # 8 slices is enough to exercise it
                dataset = pydicom.dcmread(str(path))
                dataset.StudyInstanceUID = study_uid
                dataset.SeriesInstanceUID = series_uid
                dataset.save_as(target / path.name)
            series_rows.append({
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": series % 2,
                "Fat_Suppression": (series + 1) % 2,
                "Anatomical_Plane": ["Sagittal", "Coronal"][series % 2],
            })

    pd.DataFrame({config.ID_COL: studies}).to_csv(base / "test.csv", index=False)
    pd.DataFrame(series_rows).to_csv(base / "test_series.csv", index=False)

    sample = {config.ID_COL: studies}
    for name in config.LABELS:
        sample[name] = [0.5] * len(studies)
    pd.DataFrame(sample)[config.SUBMISSION_COLUMNS].to_csv(
        base / "sample_submission.csv", index=False)

    # A study whose imaging is missing entirely, to exercise the fallback.
    missing = "1.2.826.0.1.3680043.8.498.fake.study.missing"
    frame = pd.read_csv(base / "test.csv")
    frame.loc[len(frame)] = [missing]
    frame.to_csv(base / "test.csv", index=False)

    print(f"  fake test set: {len(frame)} studies "
          f"({N_STUDIES} with imaging, 1 without)")
    return base


def build_fake_checkpoints(target: pathlib.Path):
    """Randomly initialised stand-ins for the trained weights."""
    import torch
    from src.model import embed, net, side_classifier

    target.mkdir(parents=True, exist_ok=True)
    torch.save(embed.build_encoder(pretrained=False).state_dict(),
               target / "dinov2_s_reg4.pt")
    torch.save(side_classifier.SideNet().state_dict(), target / "side_model.pt")
    for fold in range(2):
        torch.save(net.KneeNet(None, embed.EMBED_DIM).state_dict(),
                   target / f"knee_fold{fold}.pt")

    import pandas as pd
    from src import config
    labels = {config.ID_COL: ["x"]}
    for name in config.LABELS:
        labels[name] = [0.2]
        labels[f"{name}_weight"] = [1.0]
    pd.DataFrame(labels).to_parquet(target / "train_labels.parquet", index=False)
    print("  fake checkpoints written")


def code_cells(path):
    with open(path) as handle:
        notebook = json.load(handle)
    return {index: "".join(cell["source"])
            for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        data_root = base / "input" / "competitions" / "rsna-knee"
        inputs = base / "input" / "checkpoints"
        working = base / "working"
        working.mkdir(parents=True, exist_ok=True)
        data_root.mkdir(parents=True, exist_ok=True)

        print("setup")
        build_fake_competition(data_root)
        os.environ["KNEE_DATA_ROOT"] = str(data_root)
        build_fake_checkpoints(inputs)

        scratch = base / "scratch"
        scratch.mkdir()
        os.chdir(scratch)

        rewrites = {
            "/kaggle/working": str(working),
            '"/kaggle/input/*"': f'"{base / "input"}/*"',
            'f"/kaggle/input/competitions/*/{name}"':
                f'f"{data_root.parent}/*/{{name}}"',
            'f"/kaggle/input/*/{name}"': f'f"{base / "input"}/*/{{name}}"',
            'pathlib.Path("/kaggle/input")': f'pathlib.Path("{base / "input"}")',
        }

        print("\n07_submit.ipynb")
        cells = code_cells(ROOT / "notebooks" / "07_submit.ipynb")
        namespace = {"__name__": "__main__"}
        for index in sorted(cells):
            source = cells[index]
            for old, new in rewrites.items():
                source = source.replace(old, new)
            try:
                exec(source, namespace)  # noqa: S102
                print(f"  cell {index}: OK")
            except Exception:
                print(f"  cell {index}: FAILED")
                traceback.print_exc()
                os.chdir(ROOT)
                return 1

        submission = working / "submission.csv"
        print(f"\nsubmission.csv exists: {submission.exists()}")
        if submission.exists():
            print(submission.read_text()[:400])

    os.chdir(ROOT)
    print("\nsubmission notebook ran clean")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
