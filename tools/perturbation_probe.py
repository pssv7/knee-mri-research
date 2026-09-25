"""Run the declared perturbation grid over one frozen model's outer studies.

Default is preflight only: frozen design, inputs, checkpoint and reference
hashes. --execute needs CUDA and the full 224 caches. Inference only -- no
training, no label use, no submission. See src/model/perturb.py for the grid
and what it may and may not be read as.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import perturb
from src.validation.proxy import ID, LABELS, sha, write_json_new
from tools.train_nested_proxy_control import preflight


def check_reference(spec, roles, recipe, checkpoint, reference):
    contract = json.loads((reference / 'run_contract.json').read_text())
    if sha(checkpoint) != contract['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the reference run contract')
    if sha(reference / 'outer_predictions.parquet') != contract['outer_predictions_sha256']:
        raise ValueError('Reference predictions changed')
    for key in ['roles_sha256', 'proxy_labels_sha256', 'seed', 'sampling_policy']:
        if contract.get(key) != spec.get(key):
            raise ValueError(f'Reference run differs from frozen design: {key}')
    if contract['training_recipe'] != recipe:
        raise ValueError('Reference recipe differs from the frozen recipe')
    outer = sorted(roles.loc[roles.role == 'outer', ID])
    if sorted(contract['outer_uids']) != outer:
        raise ValueError('Reference outer studies differ from the frozen roles')
    return contract, outer


def run(design, input_config, checkpoint, reference, out, execute=False, limit=None):
    design, input_config = Path(design), Path(input_config)
    checkpoint, reference = Path(checkpoint), Path(reference)
    spec, roles, labels, manifest, series, sides, paths, recipe = preflight(design, input_config)
    contract, outer = check_reference(spec, roles, recipe, checkpoint, reference)
    if limit is not None:
        outer = outer[:limit]
    if not execute:
        return {'status': 'preflight_passed', 'studies': len(outer),
                'variants': list(perturb.VARIANTS), 'probe_launched': False}
    import torch
    import timm
    from src.model import dataset, lightning as knee, net

    if not torch.cuda.is_available():
        raise RuntimeError('The probe requires CUDA to match the reference precision')
    out = Path(out)
    if out.exists():
        raise FileExistsError('Probe output exists; use a fresh directory')
    out.mkdir(parents=True)
    index = dataset.StudyIndex(manifest, series, labels, sides, max_series=5, series_selection='protocol')
    studies = knee.StudyVolumeDataset(index, outer, paths, perturb.DEPTH, augment=False, resample=False)
    backbone = net.build_backbone(recipe['backbone'], pretrained=False)
    defaults = net.backbone_defaults(recipe['backbone'])
    model = net.KneeNet(backbone, backbone.num_features,
                        input_mean=defaults['input_mean'], input_std=defaults['input_std'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to('cuda').eval()
    saved = pd.read_parquet(reference / 'outer_predictions.parquet').set_index(ID)
    rows, started = [], time.monotonic()
    with torch.inference_mode():
        for i, uid in enumerate(outer):
            item = studies[i]
            if item[ID] != uid:
                raise ValueError('Dataset order differs from the outer study list')
            for name in perturb.VARIANTS:
                batch = dataset.collate_studies([perturb.apply(name, item, uid)])
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(batch['volumes'].cuda(), batch['meta'].cuda(), batch['mask'].cuda())
                scores = logits.float().sigmoid().cpu().numpy()[0]
                if name == 'identity':
                    check = perturb.reproduction(scores, saved.loc[uid, LABELS].to_numpy(float))
                    if not check['passed']:
                        raise RuntimeError(f'Identity pass does not reproduce the saved prediction '
                                           f'for {uid}: max |dp| {check["max_abs_dp"]:.4f}. Wrong '
                                           'weights, inputs or code; no summary written.')
                rows.append({ID: uid, 'variant': name, **dict(zip(LABELS, map(float, scores)))})
            done = i + 1
            if done in (1, 5) or done % 50 == 0 or done == len(outer):
                elapsed = time.monotonic() - started
                print(f'{done}/{len(outer)} studies | {elapsed / done:.2f} s/study | '
                      f'projected total {elapsed / done * len(outer) / 60:.1f} min', flush=True)
    predictions = pd.DataFrame(rows)
    predictions_path = out / 'perturbation_predictions.parquet'
    predictions.to_parquet(predictions_path, index=False)
    summary = perturb.summarise(predictions, saved.reset_index())
    write_json_new(out / 'summary.json', summary)
    code_root = Path(__file__).resolve().parents[1]
    receipt = {
        'probe': 'perturbation_v1', 'model_arm': contract['arm'], 'studies': len(outer),
        'limit': limit, 'variants': list(perturb.VARIANTS),
        'checkpoint_sha256': sha(checkpoint), 'reference_contract_sha256': sha(reference / 'run_contract.json'),
        'input_config_sha256': sha(input_config), 'predictions_sha256': sha(predictions_path),
        'summary_sha256': sha(out / 'summary.json'),
        'code_sha256': {name: sha(code_root / name) for name in
                        ['src/model/perturb.py', 'tools/perturbation_probe.py']},
        'runtime_versions': {'torch': torch.__version__, 'timm': timm.__version__,
                             'gpu': torch.cuda.get_device_name(), 'precision': 'fp16 autocast'},
        'reference_runtime_versions': contract['runtime_versions'],
        'reproduction': summary['reproduction'], 'elapsed_seconds': time.monotonic() - started,
        'labels_used': False, 'training_launched': False}
    write_json_new(out / 'run_contract.json', receipt)
    return {'status': 'probe_complete', 'studies': len(outer),
            'reproduction': summary['reproduction'],
            'sensitive': sorted({(r['variant'], r['finding']) for r in summary['rows'] if r['sensitive']})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int)
    a = parser.parse_args()
    if a.execute and a.out is None:
        parser.error('--execute requires --out')
    print(json.dumps(run(a.design, a.inputs, a.checkpoint, a.reference, a.out, a.execute, a.limit),
                     indent=2, default=str))
