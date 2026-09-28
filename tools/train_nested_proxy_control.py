"""Preflight or train ONE fresh protocol224 arm on one frozen nested split.

Default is preflight only. --execute requires a CUDA environment with full
224 caches. No gold loader, outer callbacks, ensemble export, or candidate
label repair is supported here. Historical notebooks are left unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.proxy import ID, LABELS, sha, write_json_new, validate_roles


def checked_file(entry, base):
    path = (base / entry['path']).resolve()
    if not path.is_file() or sha(path) != entry['sha256']:
        raise ValueError(f'Missing or changed input: {path.name}')
    return path


def preflight(design, input_config):
    design, input_config = Path(design), Path(input_config)
    spec = json.loads((design / 'spec.json').read_text())
    code_root = Path(__file__).resolve().parents[1]
    if not spec.get('runtime_code_sha256'):
        raise ValueError('Missing frozen runtime source hashes')
    for rel, expected_hash in spec['runtime_code_sha256'].items():
        if sha(code_root / rel) != expected_hash:
            raise ValueError(f'Runtime source changed: {rel}')
    roles = pd.read_csv(design / 'roles.csv')
    labels = pd.read_parquet(design / 'proxy_labels.parquet')
    validate_roles(roles)
    for filename, key in [('roles.csv', 'roles_sha256'), ('proxy_labels.parquet', 'proxy_labels_sha256')]:
        if sha(design / filename) != spec[key]:
            raise ValueError('Frozen design hash mismatch')
    if labels[ID].duplicated().any() or set(labels[ID]) != set(roles[ID]):
        raise ValueError('Label/split membership mismatch')
    excluded_path = design.parent / 'exclusions.csv'
    if sha(excluded_path) != spec['protected_exclusions_sha256']:
        raise ValueError('Protected exclusions changed')
    excluded = pd.read_csv(excluded_path)
    if set(roles[ID]) & set(excluded[ID]) or set(roles.component) & set(excluded.component):
        raise ValueError('Protected studies or components leaked')
    recipe = {'backbone': 'convnext_tiny.fb_in22k_ft_in1k', 'cache_size': 224,
              'slices': 16, 'max_series': 5, 'series_selection': 'protocol',
              'epochs': 12, 'effective_batch': 8, 'learning_rate': .0001,
              'schedule': 'onecycle', 'warmup_epochs': 3.6}
    if any(spec.get(k) != v for k, v in recipe.items()):
        raise ValueError('Control recipe differs from the frozen baseline')
    if spec.get('initialization') != 'original_pretrained' or spec.get('gold_monitoring') is not False:
        raise ValueError('Invalid initialization or gold monitoring')
    if spec.get('model_count') != 1 or spec.get('checkpoint_selection_role') != 'selection' or spec.get('endpoint') != 'mined_report_label_macro_roc_auc':
        raise ValueError('Only a singleton inner-selected proxy control is supported')
    policy = spec.get('sampling_policy', 'legacy')
    if policy not in ('legacy', 'distinct_raw_17_23_v1'):
        raise ValueError('Unknown sampling policy')
    if policy != 'legacy' and not spec.get('raw_counts_sha256'):
        raise ValueError('Candidate requires independent header-count provenance')
    for role in ['train', 'selection', 'outer']:
        yy = labels.set_index(ID).loc[roles.loc[roles.role == role, ID], LABELS].to_numpy(float)
        if not np.isfinite(yy).all() or ((yy < 0) | (yy > 1)).any():
            raise ValueError('Invalid proxy targets')
        positives = (yy >= .5).sum(axis=0)
        if ((positives == 0) | (positives == len(yy))).any():
            raise ValueError('Every role must support all twelve AUCs')
    weights = labels[[f'{x}_weight' for x in LABELS]].to_numpy(float)
    if not np.isfinite(weights).all() or ((weights <= 0) | (weights > 1)).any():
        raise ValueError('Invalid sample weights')
    inputs = json.loads(input_config.read_text())
    base = input_config.parent
    series = pd.read_csv(checked_file(inputs['series_meta'], base))
    sides = pd.read_parquet(checked_file(inputs['side_predictions'], base))
    if series.SeriesInstanceUID.duplicated().any() or sides.SeriesInstanceUID.duplicated().any():
        raise ValueError('Duplicate series metadata or side prediction')
    expected = series[series[ID].isin(roles[ID])]
    if set(expected[ID]) != set(roles[ID]):
        raise ValueError('Missing study series metadata')
    if expected[['Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']].isna().any().any():
        raise ValueError('Missing acquisition metadata')
    valid_sides = sides.set_index('SeriesInstanceUID').reindex(expected.SeriesInstanceUID).side_predicted
    if not valid_sides.isin(['L', 'R']).all():
        raise ValueError('Missing or invalid laterality predictions')
    frames, cache_paths = [], []
    for shard in inputs['shards']:
        array_path = checked_file(shard['array'], base)
        manifest_path = checked_file(shard['manifest'], base)
        a = np.load(array_path, mmap_mode='r')
        m = pd.read_parquet(manifest_path)
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[1:] != (24, 224, 224):
            raise ValueError('Need original 24x224x224 uint8 cache')
        pos = m.position.to_numpy(float)
        if not np.isfinite(pos).all() or (pos != np.floor(pos)).any() or (pos < 0).any() or (pos >= len(a)).any() or m.position.duplicated().any():
            raise ValueError('Invalid cache positions')
        m['cache_path'] = str(array_path)
        frames.append(m)
        cache_paths.append(str(array_path))
    if not frames or len(cache_paths) != len(set(cache_paths)):
        raise ValueError('Missing or repeated cache shards')
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest[ID].isin(roles[ID])].copy()
    if manifest.SeriesInstanceUID.duplicated().any() or set(manifest.SeriesInstanceUID) != set(expected.SeriesInstanceUID):
        raise ValueError('Incomplete or duplicate image series coverage')
    actual_pairs = set(zip(manifest[ID], manifest.SeriesInstanceUID))
    if actual_pairs != set(zip(expected[ID], expected.SeriesInstanceUID)):
        raise ValueError('Series assigned to wrong studies')
    if 'error' in manifest and manifest.error.notna().any():
        raise ValueError('Failed cache entries present')
    if spec.get('raw_counts_sha256'):
        counts_path = design.parent / 'raw_counts.parquet'
        if sha(counts_path) != spec['raw_counts_sha256']:
            raise ValueError('Header-count provenance changed')
        counts = pd.read_parquet(counts_path)
        joined = manifest.merge(counts, on=[ID, 'SeriesInstanceUID'], validate='one_to_one')
        if len(joined) != len(manifest) or not (joined.n_slices_raw == joined.raw_slices).all():
            raise ValueError('Cache raw counts disagree with source index')
        if 'mirrored' not in manifest or not manifest.mirrored.eq(False).all():
            raise ValueError('Need explicitly unmirrored source caches')
    return spec, roles, labels, manifest, series, sides, cache_paths, recipe


def run(design, input_config, out, execute=False):
    data = preflight(design, input_config)
    spec, roles, labels, manifest, series, sides, paths, recipe = data
    if not execute:
        return {'status': 'preflight_passed', 'counts': roles.role.value_counts().to_dict(), 'training_launched': False}
    import hashlib
    import torch
    import timm
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    from src.model import dataset, lightning as knee, net

    if not torch.cuda.is_available():
        raise RuntimeError('Full training requires CUDA; no CPU campaign will be started')
    out = Path(out)
    if out.exists():
        raise FileExistsError('Training output exists; use a fresh directory')
    out.mkdir(parents=True)
    pl.seed_everything(spec['seed'], workers=True)
    index = dataset.StudyIndex(manifest, series, labels, sides, max_series=5, series_selection='protocol')
    if set(index.studies) != set(roles[ID]):
        raise ValueError('Dataset silently dropped studies')
    def loader(role, augment=False):
        return DataLoader(knee.StudyVolumeDataset(index, sorted(roles.loc[roles.role == role, ID]),
            paths, 16, augment=augment, resample=False,
            duplicate_free=spec.get('sampling_policy', 'legacy') != 'legacy'), batch_size=1, shuffle=augment,
            num_workers=2, persistent_workers=True, pin_memory=True, collate_fn=dataset.collate_studies)
    train_loader, select_loader = loader('train', True), loader('selection')
    steps = knee.optimiser_steps(len(train_loader.dataset), 1, 1, 8, 12)
    backbone = net.build_backbone(recipe['backbone'], pretrained=True)
    module = knee.KneeLitModule(backbone, backbone.num_features, total_steps=steps,
        lr=.0001, warmup_steps=int(round(steps / 12 * 3.6)), **net.backbone_defaults(recipe['backbone']))
    initial = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        initial.update(name.encode())
        initial.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(out / 'checkpoints'), filename='selected',
        monitor='val_macro', mode='max', save_top_k=1)
    trainer = pl.Trainer(max_epochs=12, accelerator='gpu', devices=1, precision='16-mixed',
        accumulate_grad_batches=8, callbacks=[checkpoint],
        logger=pl.loggers.CSVLogger(str(out), name='training'), enable_model_summary=False)
    started = time.monotonic()
    # The ONLY validation loader supplied during training is inner selection.
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=select_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError('No selected checkpoint was written')
    selected = out / 'selected.pt'
    knee.export_from_checkpoint(checkpoint.best_model_path, str(selected))
    module.model.load_state_dict(torch.load(selected, map_location='cpu', weights_only=True), strict=True)
    module.to('cuda').eval()
    rows = []
    # Instantiate the outer loader only after checkpoint selection is complete.
    with torch.inference_mode():
        for batch in loader('outer'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                p = module(batch['volumes'].cuda(), batch['meta'].cuda(), batch['mask'].cuda()).float().sigmoid().cpu().numpy()
            for uid, scores in zip(batch[ID], p):
                rows.append({ID: uid, **dict(zip(LABELS, map(float, scores)))})
    preds = pd.DataFrame(rows)
    pred_path = out / 'outer_predictions.parquet'
    preds.to_parquet(pred_path, index=False)
    contract = {**spec, 'training_recipe': recipe, 'initial_weights_sha256': initial.hexdigest(),
        'runtime_versions': {'torch': torch.__version__, 'timm': timm.__version__, 'lightning': pl.__version__,
                             'gpu': torch.cuda.get_device_name(), 'devices': 1, 'precision': '16-mixed'},
        'input_config_sha256': sha(input_config), 'checkpoint_sha256': sha(selected),
        'outer_predictions_sha256': sha(pred_path), 'elapsed_seconds': time.monotonic()-started,
        'train_uids': sorted(roles.loc[roles.role == 'train', ID]),
        'selection_uids': sorted(roles.loc[roles.role == 'selection', ID]),
        'outer_uids': sorted(roles.loc[roles.role == 'outer', ID]),
        'arm': 'control' if spec.get('sampling_policy', 'legacy') == 'legacy' else 'candidate',
        'training_script_sha256': sha(__file__)}
    from src.validation.proxy import validate_prediction_provenance
    validate_prediction_provenance(roles, contract, preds)
    write_json_new(out / 'run_contract.json', contract)
    return {'status': 'one_model_trained', 'arm': contract['arm'],
            'checkpoint': str(selected), 'image_gain_claim_allowed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    if a.execute and a.out is None:
        parser.error('--execute requires --out')
    print(json.dumps(run(a.design, a.inputs, a.out, a.execute), indent=2))
