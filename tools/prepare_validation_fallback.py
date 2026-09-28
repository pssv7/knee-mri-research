"""Freeze nested report-proxy folds and assess one bounded PF proposal rule.

No held-out report reserve is opened, no labels are repaired, and no job runs.
Outputs must be a new directory; historical artifacts remain immutable.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import fuse, mode_fusion, pf_mention_candidate
from src.validation.proxy import (ID, LABELS, sha, write_json_new, linked_components,
                                  exclude_components, group_folds, nested_roles,
                                  training_only_sharpen, validate_roles)


def prepare(root=ROOT, out=None):
    out = Path(out or root / 'outputs/validation_fallback_v1')
    if out.exists():
        raise FileExistsError('Use a new output directory; frozen records are never overwritten')
    audit = root / 'outputs/validation_reset_20260924'
    metadata = pd.read_csv(audit / 'study_metadata.csv')
    near = pd.read_csv(audit / 'near_report_pairs.csv')
    components = linked_components(metadata, near)
    source_paths = [audit / 'study_metadata.csv', audit / 'near_report_pairs.csv',
                    root / 'data/meta/train.csv', root / 'data/meta/train_labels.parquet']
    protected = set(metadata.loc[metadata.fold == -1, ID])
    reasons = {u: {'historical_gold'} for u in protected}
    for rel, reason in [
        ('outputs/label_quality/audit_v1/exposure.parquet', 'known_prior_review'),
        ('outputs/label_quality/audit_v1/sample.parquet', 'report_rule_design'),
        ('outputs/label_quality/audit_v1/extra_controls.parquet', 'report_rule_design'),
        ('outputs/label_quality/audit_v1/reserve.parquet', 'protected_text_reserve'),
        ('outputs/validation_reset_20260924/pilot_custodian_key.csv', 'future_image_pilot')]:
        p = root / rel
        source_paths.append(p)
        # Read only membership of the protected report reserve.
        frame = pd.read_csv(p, usecols=[ID]) if p.suffix == '.csv' else pd.read_parquet(p, columns=[ID])
        for u in frame[ID]:
            protected.add(u)
            reasons.setdefault(u, set()).add(reason)
    allowed = exclude_components(components, protected)
    blocked = components[~components[ID].isin(allowed[ID])].copy()
    blocked['reason'] = blocked[ID].map(lambda u: '|'.join(sorted(reasons.get(u, {'linked_to_protected'}))))
    labels = pd.read_parquet(root / 'data/meta/train_labels.parquet')
    frame = allowed.merge(labels, on=ID, validate='one_to_one').sort_values(ID).reset_index(drop=True)
    frame['outer_fold'] = group_folds(frame)

    reports = pd.read_csv(root / 'data/meta/train.csv').set_index(ID)
    ledger_path = root / 'outputs/label_quality/audit_v1/ledger_assembled.parquet'
    ledger = pd.read_parquet(ledger_path)
    reviewed = ledger[ledger.finding == 'PF OA']
    hits = []
    for row in reviewed.to_dict('records'):
        suggestion = pf_mention_candidate.propose(reports.at[row[ID], 'Report'], row['p1_old_target'])
        if suggestion is not None:
            hits.append({'case_id': row['p1_case_id'], 'report_state': row['p1_report_states'],
                         'review_error_class': row['p1_error_class'],
                         'agreed_explicit_miss': bool(row['agreed_target_correction'] and
                                                    row['p1_error_class'] == 'missed_mention')})
    valid = sum(x['agreed_explicit_miss'] for x in hits)
    candidate_status = {'rule_version': pf_mention_candidate.VERSION,
        'proposed_design_cases': len(hits), 'agreed_explicit_misses': valid,
        'design_precision': valid / len(hits) if hits else None,
        'status': 'closed_design_screen_failed' if not hits or valid / len(hits) < .90 else 'requires_fresh_report_review',
        'cases': hits, 'production_label_changes': 0, 'protected_reserve_opened': False,
        'image_ground_truth': False,
        'limitation': 'Existing same-person report reviews are design evidence, not independent confirmation. No image readers available.'}

    # Build raw per-study report supervision once; fit sharpening separately in each train role.
    cat_path, prob_path = [root / 'data/meta' / n for n in ['llm_labels.parquet', 'llm_probs.parquet']]
    source_paths.extend([cat_path, prob_path, ledger_path])
    runtime_paths = list((root / 'src/model').glob('*.py')) + [root / p for p in [
        'src/config.py', 'src/validation/proxy.py', 'src/labels/calibrate.py',
        'tools/train_nested_proxy_control.py', 'tools/evaluate_nested_proxy.py']]
    runtime_hashes = {str(p.relative_to(root)): sha(p) for p in sorted(runtime_paths)}
    cat, prob = pd.read_parquet(cat_path), pd.read_parquet(prob_path)
    ids = set(frame[ID])
    cat, prob = [x[x[ID].isin(ids)].copy() for x in [cat, prob]]
    lex = fuse.lexicon_frame(reports.loc[cat[ID], 'Report'], cat[ID])
    raw = fuse.fuse_scores(lex, mode_fusion.combine(cat, prob))
    ref = fuse.fuse_scores(lex, mode_fusion.from_categorical(cat))
    out.mkdir(parents=True)
    components.to_csv(out / 'components.csv', index=False)
    blocked.to_csv(out / 'exclusions.csv', index=False)
    frame[[ID, 'component', 'outer_fold']].to_csv(out / 'outer_folds.csv', index=False)
    write_json_new(out / 'pf_candidate_design_screen.json', candidate_status)
    designs = {f'outer{f}': nested_roles(frame, f) for f in range(5)}
    # A metadata-defined equipment stress set; not a new-site claim.
    scanner = metadata.set_index(ID).scanner_proxy
    target_scanner = frame[ID].map(scanner).value_counts().index[0]
    holdout_components = set(frame.loc[frame[ID].map(scanner) == target_scanner, 'component'])
    stress = frame.copy()
    stress['outer_fold'] = np.where(stress.component.isin(holdout_components), 0, 1)
    designs['scanner_stress'] = nested_roles(stress, 0)
    counts = {}
    for name, roles in designs.items():
        validate_roles(roles)
        table = training_only_sharpen(raw, ref, roles.loc[roles.role == 'train', ID])
        table = table.merge(roles, on=ID, validate='one_to_one').sort_values(ID)
        folder = out / name
        folder.mkdir()
        roles.to_csv(folder / 'roles.csv', index=False)
        table.to_parquet(folder / 'proxy_labels.parquet', index=False)
        support = []
        for role, group in table.groupby('role'):
            for label in LABELS:
                npos = int((group[label] >= .5).sum())
                support.append({'role': role, 'finding': label, 'positive': npos, 'negative': len(group) - npos})
        support = pd.DataFrame(support)
        support.to_csv(folder / 'class_counts.csv', index=False)
        if (support[['positive', 'negative']] <= 0).any().any():
            raise ValueError(f'{name}: a finding lacks both classes; no twelve-finding evaluation possible')
        counts[name] = {k: int(v) for k, v in roles.role.value_counts().items()}
        spec = {'schema_version': 1, 'endpoint': 'mined_report_label_macro_roc_auc',
            'runtime_code_sha256': runtime_hashes,
            'design': name, 'roles_sha256': sha(folder / 'roles.csv'),
            'proxy_labels_sha256': sha(folder / 'proxy_labels.parquet'),
            'model_count': 1, 'checkpoint_selection_role': 'selection',
            'initialization': 'original_pretrained', 'backbone': 'convnext_tiny.fb_in22k_ft_in1k',
            'cache_size': 224, 'slices': 16, 'max_series': 5, 'series_selection': 'protocol',
            'epochs': 12, 'seed': 0, 'effective_batch': 8, 'learning_rate': .0001,
            'schedule': 'onecycle', 'warmup_epochs': 3.6, 'candidate_status': candidate_status['status'],
            'protected_exclusions_sha256': sha(out / 'exclusions.csv'),
            'image_gain_claim_allowed': False, 'gold_monitoring': False,
            'scope': 'A new nested proxy protocol, not numerically comparable to legacy selected-fold AUC.'}
        write_json_new(folder / 'spec.json', spec)
    code_paths = [root / p for p in ['src/validation/proxy.py', 'src/labels/pf_mention_candidate.py',
                 'src/labels/repair.py', 'src/labels/lexicon.py', 'src/labels/fuse.py',
                 'src/labels/mode_fusion.py', 'tools/prepare_validation_fallback.py']]
    status = {'status': 'implemented_not_trained', 'expert_readers_available': False,
              'eligible_studies': len(frame), 'excluded_studies': len(blocked),
              'known_components': int(frame.component.nunique()), 'counts': counts,
              'scanner_stress_proxy': target_scanner,
              'patient_linkage': 'unavailable; known text links only',
              'pf_candidate': candidate_status['status'], 'training_jobs_launched': 0,
              'image_review_complete': False, 'locked_image_final_set_created': False,
              'runtime_code_sha256': runtime_hashes,
              'source_sha256': {str(p.relative_to(root)): sha(p) for p in source_paths + code_paths}}
    write_json_new(out / 'status.json', status)
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json_new(out / 'manifest.json', manifest)
    print(json.dumps({'output': str(out), **{k: v for k, v in status.items() if k != 'source_sha256'}}, indent=2))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    prepare(out=args.out)
