"""Score one frozen model's outer studies under alternative series selections.

Inference only. Same interface and checks as tools/perturbation_probe.py: the
frozen design, inputs, checkpoint and reference predictions are verified, and
the reference policy (`protocol5`) must reproduce the saved predictions. Each
other policy re-scores only studies whose kept series differ; identical sets
reuse the reference prediction. Labels are not read here; scoring is separate.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import perturb, selection_policies as policies
from src.validation.proxy import ID, LABELS, sha, write_json_new
from tools.perturbation_probe import check_reference
from tools.train_nested_proxy_control import preflight

QUALITY = 'design/series_quality.parquet'


class Chosen:
    """Index view that hands `load_study` a fixed subset of a study's series."""
    def __init__(self, index, rows):
        self.index, self.rows = index, rows

    def series_for(self, study, rng=None):
        return self.rows

    def targets_for(self, study):
        return self.index.targets_for(study)

    def report_for(self, study):
        return self.index.report_for(study)


def run(design, input_config, checkpoint, reference, out, execute=False, limit=None, quality_path=None):
    design, input_config = Path(design), Path(input_config)
    checkpoint, reference = Path(checkpoint), Path(reference)
    spec, roles, labels, manifest, series, sides, paths, recipe = preflight(design, input_config)
    contract, outer = check_reference(spec, roles, recipe, checkpoint, reference)
    code_root = Path(__file__).resolve().parents[1]
    quality_path = Path(quality_path) if quality_path else code_root / QUALITY
    quality = pd.read_parquet(quality_path)
    outer_series = series[series[ID].isin(outer)].SeriesInstanceUID
    if not set(outer_series) <= set(quality.SeriesInstanceUID):
        raise ValueError('Header quality table does not cover every outer series')
    if limit is not None:
        outer = outer[:limit]
    if not execute:
        return {'status': 'preflight_passed', 'studies': len(outer),
                'policies': list(policies.POLICIES), 'probe_launched': False}
    import torch
    import timm
    from src.model import dataset, net

    if not torch.cuda.is_available():
        raise RuntimeError('The probe requires CUDA to match the reference precision')
    out = Path(out)
    if out.exists():
        raise FileExistsError('Probe output exists; use a fresh directory')
    out.mkdir(parents=True)
    index = dataset.StudyIndex(manifest, series, labels, sides, max_series=None, series_selection='protocol')
    backbone = net.build_backbone(recipe['backbone'], pretrained=False)
    defaults = net.backbone_defaults(recipe['backbone'])
    model = net.KneeNet(backbone, backbone.num_features,
                        input_mean=defaults['input_mean'], input_std=defaults['input_std'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to('cuda').eval()
    saved = pd.read_parquet(reference / 'outer_predictions.parquet').set_index(ID)
    shards = {path: np.load(path, mmap_mode='r') for path in paths}
    quality = quality.set_index('SeriesInstanceUID')

    def score(uid, rows):
        item = dataset.load_study(Chosen(index, rows), uid, shards)
        available = item['volumes'].shape[1]
        keep = np.linspace(0, available - 1, perturb.DEPTH).round().astype(int)
        item['volumes'] = item['volumes'][:, keep]
        batch = dataset.collate_studies([item])
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(batch['volumes'].cuda(), batch['meta'].cuda(), batch['mask'].cuda())
        return logits.float().sigmoid().cpu().numpy()[0]

    rows_out, kept_out, started, rescored = [], [], time.monotonic(), 0
    with torch.inference_mode():
        for i, uid in enumerate(outer):
            rows = index._by_study[uid].reset_index(drop=True)
            rows = rows.join(quality, on='SeriesInstanceUID')
            chosen = {p: tuple(policies.choose(p, rows)) for p in policies.POLICIES}
            base = score(uid, rows.iloc[list(chosen[policies.REFERENCE])].reset_index(drop=True))
            check = perturb.reproduction(base, saved.loc[uid, LABELS].to_numpy(float))
            if not check['passed']:
                raise RuntimeError(f'Reference selection does not reproduce the saved prediction for '
                                   f'{uid}: max |dp| {check["max_abs_dp"]:.4f}. No summary written.')
            for policy, keep in chosen.items():
                same = keep == chosen[policies.REFERENCE]
                scores = base if same else score(uid, rows.iloc[list(keep)].reset_index(drop=True))
                rescored += not same
                rows_out.append({ID: uid, 'policy': policy, 'changed': not same,
                                 **dict(zip(LABELS, map(float, scores)))})
                kept_out.append({ID: uid, 'policy': policy, 'n_kept': len(keep), 'n_available': len(rows),
                                 'kept': json.dumps(rows.SeriesInstanceUID.iloc[list(keep)].tolist())})
            done = i + 1
            if done in (1, 5) or done % 50 == 0 or done == len(outer):
                elapsed = time.monotonic() - started
                print(f'{done}/{len(outer)} studies | {rescored} re-scored | {elapsed / done:.2f} s/study | '
                      f'projected total {elapsed / done * len(outer) / 60:.1f} min', flush=True)
    predictions = pd.DataFrame(rows_out)
    base = predictions[predictions.policy == policies.REFERENCE].set_index(ID)
    reproduction = perturb.reproduction(base.loc[outer, LABELS].to_numpy(float),
                                        saved.loc[outer, LABELS].to_numpy(float))
    predictions_path = out / 'selection_predictions.parquet'
    predictions.to_parquet(predictions_path, index=False)
    pd.DataFrame(kept_out).to_parquet(out / 'selections.parquet', index=False)
    receipt = {
        'probe': 'series_selection_v1', 'model_arm': contract['arm'], 'studies': len(outer),
        'limit': limit, 'policies': list(policies.POLICIES), 'reference_policy': policies.REFERENCE,
        'rescored_study_policies': rescored,
        'checkpoint_sha256': sha(checkpoint), 'reference_contract_sha256': sha(reference / 'run_contract.json'),
        'input_config_sha256': sha(input_config), 'predictions_sha256': sha(predictions_path),
        'selections_sha256': sha(out / 'selections.parquet'),
        'code_sha256': {name: sha(code_root / name) for name in
                        ['src/model/selection_policies.py', 'tools/selection_probe.py']},
        'quality_table_sha256': sha(quality_path),
        'runtime_versions': {'torch': torch.__version__, 'timm': timm.__version__,
                             'gpu': torch.cuda.get_device_name(), 'precision': 'fp16 autocast'},
        'reproduction': reproduction, 'elapsed_seconds': time.monotonic() - started,
        'labels_used': False, 'training_launched': False}
    write_json_new(out / 'run_contract.json', receipt)
    return {'status': 'probe_complete', 'studies': len(outer), 'rescored': rescored,
            'reproduction': reproduction}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--quality', type=Path)
    a = parser.parse_args()
    if a.execute and a.out is None:
        parser.error('--execute requires --out')
    print(json.dumps(run(a.design, a.inputs, a.checkpoint, a.reference, a.out, a.execute, a.limit, a.quality),
                     indent=2, default=str))
