"""Stage the three remaining protocol folds after successful screening; no launch."""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.prepare_kaggle_training_pair import ALL_VALIDATION_COUNTS, training_source


RUNNER = '''
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    root = Path("/kaggle/working")
    start = time.monotonic()
    progress = {"fold": FOLD, "state": "running", "completed": []}
    def save():
        progress["elapsed_seconds"] = time.monotonic() - start
        (root / "completion_progress.json").write_text(json.dumps(progress, indent=2))
    try:
        for name, source in SCRIPTS.items():
            directory = root / name
            directory.mkdir(exist_ok=False)
            (directory / "train.py").write_text(source)
            progress["active"] = name
            save()
            subprocess.run([sys.executable, "-u", str(directory / "train.py")],
                           cwd=directory, check=True,
                           timeout=900 if name == "ddp_smoke" else 18000)
            progress["completed"].append(name)
            save()
        directory = root / f"protocol_fold{FOLD}"
        sys.path.insert(0, str(directory))
        import numpy as np
        import pandas as pd
        from src import config
        from src.model import artifacts
        folded = pd.read_parquet(directory / "folds.parquet")
        expected = set(folded.loc[folded.fold == FOLD, config.ID_COL])
        training = set(folded.loc[folded.fold != FOLD, config.ID_COL])
        assert len(expected) == VALIDATION_STUDIES and len(folded) == 4349
        for name in (f"convnext_fold{FOLD}.pt", f"avg_fold{FOLD}_ema.pt", f"avg_fold{FOLD}_swa.pt"):
            metadata = artifacts.load_metadata(directory / name)
            assert metadata["fold"] == FOLD and metadata["experiment"] == EXPECTED_SPEC
            assert set(metadata["val_uids"]) == expected
            assert set(metadata["train_uids"]) == training and not (training & expected)
        oof = pd.read_parquet(directory / f"oof_fold{FOLD}.parquet")
        assert len(oof) == len(expected) and set(oof[config.ID_COL]) == expected
        assert oof.fold.eq(FOLD).all()
        values = oof[config.LABELS].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        progress.update(state="complete", active=None, validation_studies=len(oof))
    except BaseException as exc:
        progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()

if __name__ == "__main__":
    main()
'''


def prepare(root=ROOT, folds=(0, 2, 4)):
    root = Path(root)
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {0, 2, 4}:
        raise ValueError("Choose unique remaining protocol folds from 0, 2 and 4")
    notebook_path = root / "notebooks/08_finetune.ipynb"
    notebook = json.loads(notebook_path.read_text())
    label_hash = hashlib.sha256((root / "data/meta/train_labels.parquet").read_bytes()).hexdigest()
    notebook_hash = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    # Require both completed allocation screens; never expand an unmeasured preset.
    for fold in (1, 3):
        results = root / f"outputs/kaggle_protocol_fold{fold}_results"
        comparison = json.loads((results / "comparison.json").read_text())
        progress = json.loads((results / "pair_progress.json").read_text())
        if (progress["state"] != "complete" or comparison["folds"] != [fold]
                or not comparison["screening_gate"]["passed"]):
            raise ValueError("Protocol screening folds must complete and pass before expansion")
        if progress["labels_sha256"] != label_hash:
            raise ValueError("Label file changed since protocol screening")
    screened = json.loads((root / "outputs/kaggle_protocol_fold3/staging_receipt.json").read_text())
    if screened["notebook_sha256"] != notebook_hash:
        raise ValueError("Training notebook changed since protocol screening")
    expected = json.loads((root / "outputs/kaggle_protocol_fold3_results/protocol_fold3/run_contract.json").read_text())
    from src.model.experiments import get_experiment
    spec = get_experiment("protocol").to_dict()
    if expected["experiment"] != spec:
        raise ValueError("Protocol preset changed since screening")
    previous = json.loads((root / "outputs/kaggle_sources/knee-finetune/kernel-metadata.json").read_text())
    for fold in folds:
        scripts = {name: training_source(notebook, "protocol", label_hash, fold=fold,
                                        smoke=smoke, complete_protocol=True)
                   for name, smoke in (("ddp_smoke", True), (f"protocol_fold{fold}", False))}
        scripts = {name: f"REFERENCE_CONTRACT = {expected!r}\n" + source
                   for name, source in scripts.items()}
        source = (f"FOLD = {fold}\nVALIDATION_STUDIES = {ALL_VALIDATION_COUNTS[fold]}\n"
                  f"EXPECTED_SPEC = {spec!r}\nSCRIPTS = {scripts!r}\n" + RUNNER)
        ast.parse(source)
        output = root / f"outputs/kaggle_protocol_fold{fold}_completion"
        output.mkdir(parents=True, exist_ok=True)
        metadata = {key: previous[key] for key in (
            "competition_sources", "kernel_sources", "dataset_sources", "docker_image")}
        metadata.update(id=f"pssv27/knee-protocol-fold{fold}-completion",
                        title=f"Knee Protocol Fold{fold} Completion", code_file="train.py",
                        language="python", kernel_type="script", is_private=True,
                        enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4")
        (output / "train.py").write_text(source)
        (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        receipt = {"kernel": metadata["id"], "fold": fold, "experiment": spec,
                   "labels_sha256": label_hash, "notebook_sha256": notebook_hash,
                   "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "state": "staged_not_launched"}
        (output / "staging_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Staged {output}; no remote job launched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 2, 4])
    args = parser.parse_args()
    prepare(folds=args.folds)
