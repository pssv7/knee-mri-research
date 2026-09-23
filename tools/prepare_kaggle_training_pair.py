"""Stage isolated, sequential matched experiments; never launch remotely."""

import ast
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KERNEL = "pssv27/knee-protocol-fold1-pair"
EXPERIMENTS = ("baseline", "protocol", "classwise", "convnextv2", "slice_transformer",
               "spatial_control", "spatial")
# The spatial candidate is screened against its own 24-slice control rather
# than against the 16-slice baseline: the plan varies the 3D path only, and a
# control that also differs in slice count would confound the two.
CONTROLS = {"spatial": "spatial_control"}
VALIDATION_COUNTS = {1: 871, 3: 869}
ALL_VALIDATION_COUNTS = {0: 871, 1: 871, 2: 870, 3: 869, 4: 868}

INPUT_SEARCH = '''
def attached_files(pattern):
    import pathlib
    found = []
    for root in sorted(pathlib.Path("/kaggle/input").iterdir()):
        if root.name not in ("competitions", "rsna-knee-abnormality-detection"):
            found.extend(root.rglob(pattern))
    return found
'''


class OutputPaths(ast.NodeTransformer):
    def __init__(self, directory):
        self.directory = directory

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/kaggle/working"):
            return ast.copy_location(ast.Constant(
                self.directory + node.value[len("/kaggle/working"):]), node)
        return node


PREFLIGHT = '''
assert (FOLD, EPOCHS, SEED, SLICES_PER_SERIES, MAX_SERIES) == (EXPECTED_FOLD, 12, EXPECTED_SEED, EXPECTED_SLICES, 5)
assert not RESAMPLE and not GROUPED_FOLDS
assert RUN.classwise_pool == (EXPERIMENT == "classwise")
assert not SILENCE_REWEIGHT and POS_WEIGHT is None and RESIDUAL_BETA is None
assert (LAMBDA_REPORT, LAMBDA_RANK, LAMBDA_STRATIFIED_RANK, LAMBDA_COUPLING) == (0., 0., 0., 0.)
assert len(index) == 4349 and len(gold_index) == 58 and len(CACHE_PATHS) == 3
assert len(train_studies) == 4349 - EXPECTED_VAL and len(val_studies) == EXPECTED_VAL
assert BATCH_STUDIES * DEVICES * ACCUM_STEPS == 8
assert BACKBONE == ("convnextv2_tiny.fcmae_ft_in22k_in1k" if EXPERIMENT == "convnextv2" else "convnext_tiny.fb_in22k_ft_in1k")
if MATCH_BACKBONE:
    assert FEATURE_DIM == 768
    assert BACKBONE_DEFAULTS == {"input_mean": None, "input_std": None, "backbone_lr_scale": 1.0}
if MATCH_SLICE or MATCH_SPATIAL:
    assert RUN.series_selection == "protocol"
    assert RUN.slice_transformer == (EXPERIMENT == "slice_transformer")
    assert RUN.spatial_encoder == (EXPERIMENT == "spatial")
    assert not RUN.plane_adapters, "Plane adapters are a separate ablation"
else:
    assert RUN.series_selection == ("protocol" if EXPERIMENT == "protocol" else "uniform")
assert all(volume.shape[1:] == (24, 224, 224) for volume in shards.values())
import hashlib, json, pathlib
_initial = hashlib.sha256()
for _key, _tensor in sorted(module.state_dict().items()):
    _initial.update(_key.encode())
    _initial.update(_tensor.detach().cpu().contiguous().numpy().tobytes())
_contract = {
    "experiment": RUN.to_dict(), "fold": FOLD, "epochs": EPOCHS, "seed": SEED,
    "devices": DEVICES, "effective_batch": EFFECTIVE_STUDIES,
    "steps": TOTAL_STEPS, "lr": LR, "schedule": SCHEDULE,
    "warmup_steps": WARMUP_STEPS, "initial_weights_sha256": _initial.hexdigest(),
    "torch": torch.__version__, "timm": timm.__version__, "lightning": pl.__version__,
}
if MATCH_CLASSWISE or MATCH_BACKBONE or MATCH_SLICE or MATCH_SPATIAL:
    # Hash only unchanged modules; keep the full initial-state hash for provenance.
    _excluded = (("model.spatial_encoder.",) if MATCH_SPATIAL else
                 ("model.slice_encoder.",) if MATCH_SLICE else
                ("model.backbone.",) if MATCH_BACKBONE else
                 ("model.study_pool.", "model.head."))
    _shared = hashlib.sha256()
    for _key, _tensor in sorted(module.state_dict().items()):
        if not _key.startswith(_excluded):
            _shared.update(_key.encode())
            _shared.update(_tensor.detach().cpu().contiguous().numpy().tobytes())
    _contract["shared_initial_weights_sha256"] = _shared.hexdigest()
if MATCH_BACKBONE:
    _contract["aggregation_seed"] = SEED
    _contract["feature_dim"] = FEATURE_DIM
    _contract["input_defaults"] = BACKBONE_DEFAULTS
if "REFERENCE_CONTRACT" in globals():
    _fold_specific = {"fold", "steps", "warmup_steps"}
    assert {k: v for k, v in _contract.items() if k not in _fold_specific} == {k: v for k, v in REFERENCE_CONTRACT.items() if k not in _fold_specific}, "Completion settings or initialization differ from screened protocol"
_baseline_contract = pathlib.Path(BASELINE_CONTRACT_PATH)
if EXPERIMENT != CONTROL_EXPERIMENT and _baseline_contract.exists():
    _control = json.loads(_baseline_contract.read_text())
    _ignored = {"experiment"} | ({"initial_weights_sha256"} if MATCH_CLASSWISE or MATCH_BACKBONE or MATCH_SLICE or MATCH_SPATIAL else set())
    assert {k: v for k, v in _contract.items() if k not in _ignored} == {k: v for k, v in _control.items() if k not in _ignored}, "Candidate settings or shared initialization differ from baseline"
if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    pathlib.Path("run_contract.json").write_text(json.dumps(_contract, indent=2) + "\\n")
print("Matched-run preflight passed", flush=True)
'''


def training_source(notebook, experiment, labels_sha256, recovery=None, smoke=False, fold=1,
                    match_classwise=False, match_backbone=False, complete_protocol=False,
                    match_slice=False, match_spatial=False, training_seed=0, campaign_completion=False):
    if training_seed not in (0, 1):
        raise ValueError("Campaign seeds are 0 (primary) and 1 (repeat)")
    if campaign_completion and (fold not in (0, 2, 4) or experiment not in ("spatial", "baseline")):
        raise ValueError("Campaign completion supports candidate folds 0, 2, 4 only")
    match_spatial = match_spatial or experiment in ("spatial", "spatial_control")
    if match_spatial and (recovery or complete_protocol or match_classwise
                          or match_backbone or match_slice
                          or experiment not in ("spatial_control", "spatial")):
        raise ValueError("Spatial comparison requires fresh 24-slice protocol arms only")
    match_slice = match_slice or experiment == "slice_transformer"
    if match_slice and (recovery or complete_protocol or match_classwise or match_backbone
                        or experiment not in ("baseline", "slice_transformer")):
        raise ValueError("Slice comparison requires fresh baseline/protocol inputs only")
    if complete_protocol:
        if (fold not in (0, 2, 4) or experiment != "protocol" or recovery
                or match_classwise or match_backbone):
            raise ValueError("Completion is only for unchanged protocol folds 0, 2 and 4")
    elif fold not in VALIDATION_COUNTS and not campaign_completion and not (match_slice and fold in ALL_VALIDATION_COUNTS):
        raise ValueError("Only screening folds 1 and 3 are supported")
    if recovery and fold != 1:
        raise ValueError("The saved recovery checkpoint belongs to fold 1")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Only the authorized experiments {EXPERIMENTS} are supported")
    match_classwise = match_classwise or experiment == "classwise"
    match_backbone = match_backbone or experiment == "convnextv2"
    if match_backbone and (match_classwise or experiment not in ("baseline", "convnextv2")):
        raise ValueError("Backbone comparison cannot include other architecture or selection changes")
    if recovery and match_classwise:
        raise ValueError("Classwise comparison requires a fresh native-DDP baseline")
    if recovery and match_backbone:
        raise ValueError("Backbone comparison requires a fresh baseline with matched aggregation seed")
    control = "spatial_control" if match_spatial else "baseline"
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if len(sources) != 10:
        raise ValueError("Notebook 08 structure changed; review the staging adapter")
    if not sources[0].startswith("!pip install -q timm\n"):
        raise ValueError("Notebook setup changed")
    # Kaggle may mount the competition tree either as /input/competitions/*
    # or directly as /input/rsna-knee-abnormality-detection. The notebook's
    # historical finder skips only the former; patch the embedded run source
    # so it never recursively walks the raw-DICOM tree while locating caches.
    for i, source in enumerate(sources):
        sources[i] = source.replace(
            'if os.path.basename(root) == "competitions":\n            continue',
            'if (os.path.basename(root) == "competitions" or\n'
            '                os.path.exists(os.path.join(root, "train.csv")) or\n'
            '                os.path.exists(os.path.join(root, "train_series.csv"))):\n'
            '            continue')
    # The pinned Kaggle image already supplies timm; do not upgrade between runs.
    sources[0] = sources[0].split("\n", 1)[1]
    directory = f"/kaggle/working/{'ddp_smoke' if smoke else experiment + '_fold' + str(fold)}"
    header = INPUT_SEARCH + f'''\nimport hashlib, os, pathlib
EXPECTED_FOLD = {fold}
EXPECTED_SEED = {training_seed}
EXPECTED_VAL = {ALL_VALIDATION_COUNTS[fold]}
MATCH_CLASSWISE = {match_classwise!r}
MATCH_BACKBONE = {match_backbone!r}
MATCH_SLICE = {match_slice!r}
MATCH_SPATIAL = {match_spatial!r}
EXPECTED_SLICES = {24 if match_spatial else 16}
CONTROL_EXPERIMENT = {control!r}
BASELINE_CONTRACT_PATH = "/kaggle/working/{control}_fold{fold}/run_contract.json"
os.environ["KNEE_EXPERIMENT"] = {experiment!r}
os.environ["PYTHONUNBUFFERED"] = "1"
paths = attached_files("train_labels.parquet")
assert len(paths) == 1, f"Expected one label file, found {{paths}}"
assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == {labels_sha256!r}, "Label file changed"
import faulthandler
faulthandler.enable()
# Keep fatal-signal diagnostics, but do not arm a background stack-dump timer.
# Fold 2 and fold 4 SIGSEGVs occurred inside scheduled dumps while training
# was progressing. Parent time limits are configured by the staging command.
'''
    if recovery:
        header += f'''
import shutil
_recovery = {recovery!r}
_checkpoints = attached_files("baseline_lit_fold1.ckpt")
assert len(_checkpoints) == 1, "Expected exactly one recoverable baseline checkpoint"
with _checkpoints[0].open("rb") as _handle:
    assert hashlib.file_digest(_handle, "sha256").hexdigest() == _recovery["checkpoint_sha256"]
shutil.copy2(_checkpoints[0], "lit_fold1.ckpt")
pathlib.Path("lit_fold1/metrics").mkdir(parents=True, exist_ok=True)
shutil.copy2(_checkpoints[0].parent / "baseline_metrics.csv", "lit_fold1/metrics/metrics.csv")
print("Recovered the completed baseline checkpoint; no baseline retraining", flush=True)
'''
    result = [header]
    guard_inserted = False
    aggregation_seed_inserted = False
    for i, source in enumerate(sources):
        if smoke and i >= 7:
            continue
        if source.startswith('FINDING, STRATIFY_BY = "Synovitis", "Effusion"'):
            continue
        if match_slice and i != 1:
            # The control must use the same successful protocol selector as
            # the slice-transformer candidate, before loaders are constructed.
            source = source.replace(
                'RUN = experiments.get_experiment(EXPERIMENT)',
                'RUN = experiments.get_experiment(EXPERIMENT)\n'
                'from dataclasses import replace\n'
                'RUN = replace(RUN, series_selection="protocol")')
        tree = ast.parse(source)
        if i != 1:  # Embedded repo source must remain byte-for-byte unchanged.
            tree = OutputPaths(directory).visit(tree)
        if match_backbone:
            for pos, node in enumerate(tree.body):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == "knee_lightning.KneeLitModule"):
                    keyword = next(k for k in node.value.keywords if k.arg == "backbone")
                    if ast.unparse(keyword.value) != "net.build_backbone(BACKBONE)":
                        raise ValueError("Backbone construction changed; review aggregation seeding")
                    keyword.value = ast.Name(id="_matched_backbone", ctx=ast.Load())
                    # Different trunk constructors consume different RNG draws.
                    before = ast.parse('_matched_backbone = net.build_backbone(BACKBONE)\npl.seed_everything(SEED, workers=True)').body
                    tree.body[pos:pos] = before
                    aggregation_seed_inserted = True
                    break
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "SEED" for t in node.targets):
                node.value = ast.Constant(training_seed)
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "FOLD" for t in node.targets):
                node.value = ast.Constant(fold)
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "STRATEGY" for t in node.targets):
                node.value = ast.parse('"ddp" if DEVICES > 1 else "auto"', mode="eval").body
            if smoke and isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.Trainer":
                node.keywords = [k for k in node.keywords if k.arg != "max_epochs"]
                node.keywords += [ast.keyword(arg=k, value=ast.Constant(v)) for k, v in {
                    "max_epochs": 1, "limit_train_batches": 8,
                    "limit_val_batches": 8, "num_sanity_val_steps": 0}.items()]
        for pos, node in enumerate(tree.body):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "trainer.fit"):
                before = PREFLIGHT
                if recovery:
                    before += '\nassert json.loads(pathlib.Path("run_contract.json").read_text()) == _recovery["contract"], "Recovered baseline settings changed"\n'
                if smoke:
                    before += '\naverager.total_steps = 2\naverager.swa_fraction = 1.0\n'
                fit = ([ast.parse('checkpointer.best_model_path = str(pathlib.Path("lit_fold1.ckpt").resolve())').body[0]]
                       if recovery else [node])
                # Script DDP re-executes this file. Only rank zero exports/scores.
                after = ast.parse('if not trainer.is_global_zero:\n    raise SystemExit(0)').body
                tree.body[pos:pos + 1] = ast.parse(before).body + fit + after
                guard_inserted = True
                break
        result.append(ast.unparse(ast.fix_missing_locations(tree)))
    if not guard_inserted:
        raise ValueError("Could not locate training preflight insertion point")
    if match_backbone and not aggregation_seed_inserted:
        raise ValueError("Could not locate backbone/aggregation initialization boundary")
    if smoke:
        result.append('assert all(pathlib.Path(p).exists() for p in [CHECKPOINT, averager.ema_path, averager.swa_path])\nprint("DDP_SMOKE_PASSED: fit returned and selected/EMA/SWA exports exist", flush=True)')
    if recovery:
        result.append('''
_recovered_truth = (labels.set_index(config.ID_COL).loc[oof_ids, config.LABELS].to_numpy() >= .5).astype(int)
_recovered_macro = knee_lightning.macro_auc(_recovered_truth, oof[config.LABELS].to_numpy())
assert np.isfinite(oof[config.LABELS].to_numpy()).all()
assert abs(_recovered_macro - _recovery["expected_val_macro"]) < .001, "Recovered OOF does not reproduce the selected checkpoint"
print(f"BASELINE_RECOVERED: OOF macro {_recovered_macro:.6f}", flush=True)
''')
    source = "\n\n".join(result) + "\n"
    if match_spatial:
        from tools.training_measurement import instrument
        source = instrument(source)
        source = source.replace("_contract.update(microbatch=", f"_contract.update(labels_sha256={labels_sha256!r}, microbatch=")
    ast.parse(source)
    return source


def prepare(root=ROOT, recover_baseline=None, fold=1, candidate="protocol", smoke_only=False,
            no_timeout=False):
    if candidate not in EXPERIMENTS[1:]:
        raise ValueError(f"Candidate must be one of {EXPERIMENTS[1:]}")
    if recover_baseline and candidate != "protocol":
        raise ValueError("Recovery is supported only for the protocol experiment")
    if recover_baseline and smoke_only:
        raise ValueError("Smoke-only runs must not attach recovery artifacts")
    if candidate == "spatial_control":
        raise ValueError("spatial_control is the control arm, not a candidate")
    allowed_folds = ALL_VALIDATION_COUNTS if candidate == "slice_transformer" else VALIDATION_COUNTS
    if fold not in allowed_folds or (recover_baseline and fold != 1):
        raise ValueError("Use fold 1 or 3; recovery is available only for fold 1")
    notebook_path = root / "notebooks/08_finetune.ipynb"
    notebook = json.loads(notebook_path.read_text())
    digest = hashlib.sha256((root / "data/meta/train_labels.parquet").read_bytes()).hexdigest()
    recovery = None
    if recover_baseline:
        directory = Path(recover_baseline) / "baseline_fold1"
        with (directory / "lit_fold1.ckpt").open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        recovery = {"checkpoint_sha256": checkpoint_hash,
                    "contract": json.loads((directory / "run_contract.json").read_text())}
        with (directory / "lit_fold1/metrics/metrics.csv").open() as handle:
            recovery["expected_val_macro"] = max(float(row["val_macro"])
                for row in csv.DictReader(handle) if row["val_macro"])
        upload = root / "outputs/kaggle_recovery_upload"
        upload.mkdir(parents=True, exist_ok=True)
        target = upload / "baseline_lit_fold1.ckpt"
        if not target.exists():
            os.link(directory / "lit_fold1.ckpt", target)
        with target.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == checkpoint_hash
        shutil.copy2(directory / "lit_fold1/metrics/metrics.csv", upload / "baseline_metrics.csv")
        shutil.copy2(directory / "run_contract.json", upload / "baseline_run_contract.json")
        (upload / "dataset-metadata.json").write_text(json.dumps({
            "id": "pssv27/knee-fold1-recovery-private", "title": "Knee Fold1 Recovery Private",
            "licenses": [{"name": "other"}],
            "description": "Private recovery of the completed baseline from the failed paired training job."
        }, indent=2) + "\n")
    control = CONTROLS.get(candidate, "baseline")
    required_macro_delta = .005 if candidate == "spatial" else .003
    experiments = (control, candidate)
    scripts = {name: training_source(notebook, name, digest,
               recovery=recovery if name == "baseline" else None, fold=fold,
               match_classwise=candidate == "classwise",
               match_backbone=candidate == "convnextv2",
               match_slice=candidate == "slice_transformer",
               match_spatial=candidate == "spatial") for name in experiments}
    smoke_source = training_source(notebook, candidate, digest, smoke=True, fold=fold)
    smoke_scripts = {"ddp_smoke": smoke_source}
    if candidate == "spatial":
        smoke_scripts = {f"smoke_{arm}": training_source(notebook, arm, digest, smoke=True, fold=fold).replace(
            "/kaggle/working/ddp_smoke", f"/kaggle/working/smoke_{arm}") for arm in experiments}
    runner = (root / "tools/kaggle_training_pair_body.py").read_text()
    comparison = (root / "tools/compare_oof.py").read_text()
    source = (INPUT_SEARCH + f"\nTRAINING_SCRIPTS = {scripts!r}\nSMOKE_SOURCE = {smoke_source!r}\nSMOKE_SCRIPTS = {smoke_scripts!r}\nCOMPARE_SOURCE = {comparison!r}\n"
              f"LABELS_SHA256 = {digest!r}\nRECOVERY = {recovery!r}\nFOLD = {fold}\nCANDIDATE = {candidate!r}\nCONTROL = {control!r}\nSMOKE_ONLY = {smoke_only!r}\nVALIDATION_STUDIES = {ALL_VALIDATION_COUNTS[fold]}\n"
              f"REQUIRED_MACRO_DELTA = {required_macro_delta!r}\n"
              f"SMOKE_TIMEOUT = {None if no_timeout else 900!r}\nRUN_TIMEOUT = {None if no_timeout else 18000!r}\n" + runner)
    ast.parse(source)
    kernel = "pssv27/knee-protocol-fold1-recovery" if recovery else KERNEL
    output = root / ("outputs/kaggle_training_recovery" if recovery else "outputs/kaggle_training_pair")
    if fold != 1:
        kernel = f"pssv27/knee-protocol-fold{fold}-pair"
        output = root / f"outputs/kaggle_protocol_fold{fold}"
    if candidate != "protocol":
        kernel = f"pssv27/knee-{candidate.replace('_', '-')}-fold{fold}-pair"
        output = root / f"outputs/kaggle_{candidate}_fold{fold}"
    if smoke_only:
        kernel = f"pssv27/knee-{candidate.replace('_', '-')}-fold{fold}-smoke"
        output = root / f"outputs/kaggle_{candidate}_fold{fold}_smoke"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_pair.py").write_text(source)
    previous = json.loads((root / "outputs/kaggle_sources/knee-finetune/kernel-metadata.json").read_text())
    metadata = {key: previous[key] for key in (
        "competition_sources", "kernel_sources", "dataset_sources", "docker_image")}
    if recovery:
        metadata["dataset_sources"].append("pssv27/knee-fold1-recovery-private")
    metadata.update(id=kernel, title="Knee Protocol Fold1 Recovery" if recovery else f"Knee {candidate.replace('_', ' ').title()} Fold{fold} Pair", code_file="train_pair.py",
                    language="python", kernel_type="script", is_private=True,
                    enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4")
    if smoke_only:
        metadata["title"] = f"Knee {candidate.title()} Fold{fold} Smoke"
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "staging_receipt.json").write_text(json.dumps({
        "kernel": kernel, "labels_sha256": digest, "recovery": recovery,
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "experiments": list(experiments), "fold": fold, "epochs": 12,
        "required_macro_delta": required_macro_delta, "maximum_finding_drop": .020,
        "smoke_only": smoke_only, "smoke_max_epochs": 1, "smoke_train_batches": 8,
        "per_run_timeout_seconds": None if no_timeout else 18000,
        "smoke_timeout_seconds": None if no_timeout else 900,
    }, indent=2) + "\n")
    print(f"Staged {output}; no GPU job or competition submission launched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-baseline", type=Path)
    parser.add_argument("--fold", type=int, choices=sorted(ALL_VALIDATION_COUNTS), default=1)
    parser.add_argument("--candidate", choices=EXPERIMENTS[1:], default="protocol")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-timeout", action="store_true",
                        help="Remove parent smoke/full timeouts; platform limit still applies")
    args = parser.parse_args()
    prepare(recover_baseline=args.recover_baseline, fold=args.fold, candidate=args.candidate,
            smoke_only=args.smoke_only, no_timeout=args.no_timeout)
