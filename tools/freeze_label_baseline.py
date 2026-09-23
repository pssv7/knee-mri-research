"""Freeze the inputs the label-quality work starts from; never alter them.

    .venv/bin/python tools/freeze_label_baseline.py --create
    .venv/bin/python tools/freeze_label_baseline.py --verify

`--create` writes `docs/label_quality/baseline_<version>.json` -- hashes,
counts and metadata only, safe to track -- and the per-study report hashes to
`outputs/label_quality/report_hashes_<version>.parquet`, which stays ignored.
It refuses to overwrite an existing version: a frozen baseline is replaced by
a new version, never edited.

`--verify` recomputes everything and fails if any input differs from the
frozen record. Code drift is listed but does not fail, because the tools are
expected to change while the inputs are not.

A pass is a reproduction check. It proves the inputs are the ones that were
frozen; it says nothing about whether the labels are right. Nothing here
prints report text -- only paths, counts and hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from tools.build_label_review import sha, verified_protocol_oof

LABELS_PATH = 'data/meta/train_labels.parquet'
REPORTS_PATH = 'data/meta/train.csv'
FOLDS_PATH = 'outputs/ensemble_16_24/reconstructed_folds.parquet'
GOLD_PATH = 'data/meta/gold_truth_58.parquet'
DUMPS = ['data/meta/llm_labels.parquet', 'data/meta/llm_probs.parquet',
         'data/meta/llm_labels_gold.parquet', 'data/meta/llm_probs_gold.parquet']
INCUMBENT_OOF = 'data/meta/compare_protocol_incumbent_all_folds/incumbent_oof.parquet'
RESOLUTION_OOF = 'outputs/resolution384_downloads/oof_all_folds.parquet'
RESOLUTION_RECEIPT = 'outputs/resolution384_downloads/import_receipt.json'
SOURCE_DOCS = ['docs/label_quality_plan.md', '.cross-plan/label-quality-solo/PLAN.final.md']
# Hashes build_label_review.py recorded on 2026-09-15, before this work began.
PRIOR_RECEIPT = 'outputs/label_review_v1/summary.json'
CODE = ['src/config.py', 'src/labels/fuse.py', 'src/labels/mode_fusion.py',
        'src/labels/lexicon.py', 'src/labels/llm_extract.py', 'src/labels/reviewed.py',
        'src/model/splits.py', 'tools/audit_labels.py', 'tools/build_label_review.py',
        'tools/compare_oof.py', 'tools/prepare_reviewed_labels.py',
        'tools/prepare_kaggle_training_pair.py', 'tools/freeze_label_baseline.py']


def table(root, rel):
    path = root / rel
    frame = pd.read_parquet(path) if rel.endswith('.parquet') else pd.read_csv(path)
    return frame, {'path': rel, 'sha256': sha(path), 'rows': len(frame),
                   'columns': list(map(str, frame.columns))}


def report_hashes(reports):
    """One sha256 per study, over the report text exactly as the corrections bind it."""
    hashes = pd.DataFrame({
        config.ID_COL: reports[config.ID_COL].astype(str),
        'report_sha256': [hashlib.sha256(str(text).encode()).hexdigest()
                          for text in reports['Report']]})
    return hashes.sort_values(config.ID_COL, kind='stable').reset_index(drop=True)


def digest_lines(lines):
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def git(root, *args):
    try:
        return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inventory(root=ROOT, resolver=verified_protocol_oof):
    """Everything --create freezes, plus the report-hash table kept out of the record."""
    labels, labels_facts = table(root, LABELS_PATH)
    expected = [config.ID_COL] + [c for name in config.LABELS for c in (name, name + '_weight')]
    labels_facts['label_order_ok'] = list(labels.columns) == expected

    reports, reports_facts = table(root, REPORTS_PATH)
    hashes = report_hashes(reports)
    reports_facts.pop('columns')  # carries only the id, the report and the gold columns
    reports_facts['report_hashes_sha256'] = digest_lines(
        f'{u}:{h}' for u, h in zip(hashes[config.ID_COL], hashes.report_sha256))

    folds, folds_facts = table(root, FOLDS_PATH)
    folds_facts['per_fold'] = {str(k): int(v) for k, v in folds.fold.value_counts().sort_index().items()}

    gold_uids = sorted(reports.loc[reports[config.LABELS[0]].notna(), config.ID_COL].astype(str))
    _, gold_facts = table(root, GOLD_PATH)
    gold = {'studies': len(gold_uids), 'uids_sha256': digest_lines(gold_uids),
            'in_training_labels': len(set(gold_uids) & set(labels[config.ID_COL].astype(str))),
            'reference': gold_facts}

    dumps = {}
    for rel in DUMPS:
        _, facts = table(root, rel)
        facts['columns'] = len(facts['columns'])
        dumps[rel] = facts

    paths, provenance, _, summary_path = resolver(root)
    summary = json.loads(summary_path.read_text())
    checkpoints = {str(e['fold']): e['checkpoint_sha256']
                   for e in summary['source_receipts'] if e['family'] == 'protocol'}
    resolution = json.loads((root / RESOLUTION_RECEIPT).read_text())
    predictions = {
        'protocol_oof': provenance,
        'protocol_checkpoints': checkpoints,
        'protocol_receipts': {'path': str(summary_path.relative_to(root)), 'sha256': sha(summary_path)},
        'incumbent_oof': {'path': INCUMBENT_OOF, 'sha256': sha(root / INCUMBENT_OOF)},
        'resolution384_oof': {'path': RESOLUTION_OOF, 'sha256': sha(root / RESOLUTION_OOF),
                              'note': 'first recorded here; no earlier receipt holds its hash'},
        'resolution384_checkpoints': {str(m['fold']): m['checkpoint_sha256'] for m in resolution['members']},
    }

    sources = {}
    for rel in SOURCE_DOCS:
        path = root / rel
        sources[rel] = {'sha256': sha(path), 'git_blob': git(root, 'rev-parse', f'HEAD:{rel}')}

    record = {
        'schema_version': 1,
        'scope': ('Reproduction record: a passing --verify proves the inputs are the frozen ones, '
                  'not that any label is correct.'),
        'inputs': {'labels': labels_facts, 'reports': reports_facts, 'folds': folds_facts,
                   'gold': gold, 'extractor_dumps': dumps, 'reference_predictions': predictions,
                   'source_documents': sources},
    }
    return record, hashes


def cross_check(root, record):
    """Compare against the hashes recorded before this work began."""
    prior = json.loads((root / PRIOR_RECEIPT).read_text())['input_sha256']
    now = {LABELS_PATH: record['inputs']['labels']['sha256'],
           REPORTS_PATH: record['inputs']['reports']['sha256'],
           INCUMBENT_OOF: record['inputs']['reference_predictions']['incumbent_oof']['sha256'],
           record['inputs']['reference_predictions']['protocol_receipts']['path']:
               record['inputs']['reference_predictions']['protocol_receipts']['sha256']}
    return {'receipt': PRIOR_RECEIPT,
            'matches': {path: prior.get(path) == value for path, value in sorted(now.items())}}


def code_state(root):
    dirty = set((git(root, 'status', '--porcelain', '--', *CODE) or '').splitlines())
    dirty = {line[3:] for line in dirty}
    return {'git_head': git(root, 'rev-parse', 'HEAD'),
            'files': {rel: {'sha256': sha(root / rel), 'uncommitted_changes': rel in dirty}
                      for rel in CODE if (root / rel).exists()}}


def dump(record):
    return json.dumps(record, indent=2, sort_keys=True) + '\n'


def paths_for(root, version):
    return (root / f'docs/label_quality/baseline_{version}.json',
            root / f'outputs/label_quality/report_hashes_{version}.parquet')


def create(root=ROOT, version='v1', resolver=verified_protocol_oof):
    record_path, hashes_path = paths_for(root, version)
    if record_path.exists() or hashes_path.exists():
        raise FileExistsError(f'{version} is frozen; freeze a new version instead of overwriting')
    record, hashes = inventory(root, resolver)
    record['cross_check'] = cross_check(root, record)
    if not all(record['cross_check']['matches'].values()):
        raise ValueError(f"inputs differ from {PRIOR_RECEIPT}: {record['cross_check']['matches']}")
    if not record['inputs']['labels']['label_order_ok'] or record['inputs']['gold']['in_training_labels']:
        raise ValueError('label columns out of order, or gold studies inside the training labels')
    record['code'] = code_state(root)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    hashes_path.parent.mkdir(parents=True, exist_ok=True)
    hashes.to_parquet(hashes_path, index=False)
    record['inputs']['reports']['report_hashes_file'] = {
        'path': str(hashes_path.relative_to(root)), 'rows': len(hashes)}
    record_path.write_text(dump(record))
    return record_path


def differences(frozen, current, prefix=''):
    """Dotted paths whose values differ; values themselves are never printed."""
    if isinstance(frozen, dict) and isinstance(current, dict):
        out = []
        for key in sorted(set(frozen) | set(current)):
            out += differences(frozen.get(key), current.get(key), f'{prefix}{key}.')
        return out
    return [] if frozen == current else [prefix.rstrip('.')]


def verify(root=ROOT, version='v1', resolver=verified_protocol_oof):
    record_path, hashes_path = paths_for(root, version)
    frozen = json.loads(record_path.read_text())
    current, hashes = inventory(root, resolver)
    current['inputs']['reports']['report_hashes_file'] = frozen['inputs']['reports'].get('report_hashes_file')
    changed = differences(frozen['inputs'], current['inputs'])
    stored = pd.read_parquet(hashes_path)
    if not stored.equals(hashes):
        changed.append(f'report hashes ({(stored.report_sha256 != hashes.report_sha256).sum()} studies)'
                       if len(stored) == len(hashes) else 'report hashes (row count)')
    was, now = frozen['code']['files'], code_state(root)['files']
    drift = sorted(rel for rel in set(was) | set(now)
                   if (was.get(rel) or {}).get('sha256') != (now.get(rel) or {}).get('sha256'))
    return changed, drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--version', default='v1')
    args = parser.parse_args(argv)
    if args.create:
        path = create(version=args.version)
        record = json.loads(path.read_text())
        print(f'frozen {path.relative_to(ROOT)}')
        print(f"  labels {record['inputs']['labels']['rows']} studies, reports "
              f"{record['inputs']['reports']['rows']}, gold {record['inputs']['gold']['studies']}, "
              f"protocol OOF folds {len(record['inputs']['reference_predictions']['protocol_oof'])}")
        print(f"  matches {PRIOR_RECEIPT}: {all(record['cross_check']['matches'].values())}")
        return 0
    changed, drift = verify(version=args.version)
    for key in changed:
        print(f'CHANGED  {key}')
    for rel in drift:
        print(f'code drift (not a failure)  {rel}')
    print('verify:', 'FAIL' if changed else 'pass (reproduction, not correctness)')
    return 1 if changed else 0


if __name__ == '__main__':
    sys.exit(main())
