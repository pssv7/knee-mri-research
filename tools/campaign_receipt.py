"""Snapshot reproducible campaign inputs and pending execution gates; no launches."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.artifacts import sha256, load_metadata


def build(root=ROOT):
    root = Path(root)
    manifest_path = root / 'outputs/ensemble_16_24_protocol/ensemble.json'
    manifest = json.loads(manifest_path.read_text())
    files = ['docs/multiplane_campaign_plan.md', 'data/meta/train_labels.parquet',
             'data/meta/train.csv', 'data/meta/train_series.csv',
             'data/meta/compare_protocol_incumbent_all_folds/protocol_addition_oof.parquet',
             'outputs/ensemble_16_24_protocol/ensemble.json',
             'outputs/ensemble_16_24_protocol/reconstructed_folds.parquet',
             'outputs/ensemble_16_24_protocol/provenance.json']
    checkpoints = []
    for member in manifest['members']:
        path = manifest_path.parent / member['checkpoint']
        meta = load_metadata(path)
        checkpoints.append({'path': str(path.relative_to(root)), 'sha256': meta['sha256'],
                            'fold': meta['fold'], 'experiment': meta['experiment'],
                            'sidecar_sha256': sha256(str(path) + '.json'), 'weight': member['weight']})
    source_files = sorted([*root.glob('src/**/*.py'), *root.glob('tools/*.py')])
    receipt = {'state': 'prepared_not_gpu_validated',
               'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'source_sha256': {str(p.relative_to(root)): sha256(p) for p in source_files},
               'input_sha256': {name: sha256(root / name) for name in files},
               'incumbent_checkpoints': checkpoints,
               'legacy_exclusion': 'reconstructed, not independently proven',
               'screening': {'folds': [1, 3], 'standalone_delta': .005, 'maximum_finding_drop': .020},
               'allocation': {'recipe': '50/50 candidate/incumbent', 'pooled_delta': .003,
                              'bootstrap_draws': 1000, 'lower_bound_strictly_positive': True},
               'seed_rule': {'state': 'requires_acceptance_before_repeat_launch', 'seed': 1,
                             'standalone_delta_strictly_positive': True,
                             'blend_lower_bound_strictly_positive': True, 'include_in_submission': False},
               'deployment': {'state': 'requires_acceptance_before_dependent_launch',
                              'recipe': 'equal family weights', 'delta': .003, 'minimum_fold_delta': -.005,
                              'bootstrap_lower_bound_strictly_positive': True},
               'sequence': ['spatial_fold1', 'spatial_fold3_if_passed', 'resolution_fold1', 'resolution_fold3_if_passed'],
               'parallel_resolution': 'not authorized; use sequential order',
               'plane_adapters': 'deferred; excluded from deployment',
               'deadlines_utc': {'expansion_start_by': '2026-10-01', 'packaging_by': '2026-10-17',
                                 'submission': '2026-10-22T23:59:00Z'},
               'pending': ['verify all six original 384 receipts and payloads on Drive',
                           'representative A100 smokes and spatial fold-1 screen',
                           'fold-3 screens, incumbent blends and seed repeats after gates',
                           'promoted family completion',
                           'mixed-resolution inference only if resolution promoted',
                           'exact-manifest packaging and 100-study T4 benchmark',
                           'explicit approval of verified submission; scored result and final entries']}
    output = root / 'outputs/multiplane_campaign'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'input_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return output / 'input_receipt.json'


if __name__ == '__main__':
    print(build())
