"""Draw the priority report-audit sample and seal the step-3 manifest.

    .venv/bin/python tools/build_priority_audit.py

Order: verify the frozen baseline, inventory prior exposure, draw the
validation reserve, protect its studies, then draw controls and the enriched
strata (src/labels/audit_sampling.py).

Case-level files carry targets and model scores, so they go to ignored
`outputs/label_quality/audit_v1/`. The reviewer never opens them: review
packets (plan step 4) are built from them with those fields removed. The
tracked manifest `docs/label_quality/manifest_v1.json` holds settings, counts
and hashes only, and it is never overwritten. Nothing here prints report
text.
"""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import audit_sampling as sampling
from src.model import splits
from tools.audit_labels import detect_language
from tools.build_label_review import sha, verified_protocol_oof
from tools.compare_oof import read_predictions
from tools import freeze_label_baseline as baseline

ID = config.ID_COL
OUT = 'outputs/label_quality/audit_v1'
MANIFEST = 'docs/label_quality/manifest_v1.json'
QUEUE = 'outputs/label_review_v1/review_queue.jsonl'
INDEX = 'data/meta/resolution_index/index_train.parquet'
BOUND = {'baseline': 'docs/label_quality/baseline_v1.json',
         'rubric': 'docs/report_review_rubric.md',
         'decisions': 'docs/label_quality/decisions.md',
         'approved_plan': '.cross-plan/label-quality-solo/PLAN.final.md',
         'source_plan': 'docs/label_quality_plan.md'}


def exposure(root, gold_uids):
    """Every study whose report is known to have been shown to a reader.

    The label_review_v1 page rendered the full report for all 250 queued
    pairs, reviewed or not, so every queued study counts as exposed.
    Anything read ad hoc during development was never logged; that gap is
    recorded as a limitation rather than guessed at.
    """
    queue = pd.read_json(root / QUEUE, lines=True)
    rows = [{ID: uid, 'source': 'label_review_v1_queue'} for uid in sorted(set(queue[ID]))]
    rows += [{ID: uid, 'source': 'gold_58'} for uid in sorted(gold_uids)]
    prior_pairs = {(r[ID], r['finding']): r['review_id'] for r in queue.to_dict('records')
                   if r['finding'] in sampling.PRIORITY}
    scopes = queue[['review_id', ID, 'finding', 'review_scope']]
    return pd.DataFrame(rows), prior_pairs, scopes


def load_pairs(root):
    """Per priority finding, every study's target, weight, fold, protocol OOF and language."""
    labels = pd.read_parquet(root / baseline.LABELS_PATH)
    folds = pd.read_parquet(root / baseline.FOLDS_PATH)
    reports = pd.read_csv(root / baseline.REPORTS_PATH)
    gold_uids = splits.load_gold_uids(root / baseline.REPORTS_PATH)
    languages = pd.Series([detect_language(t) for t in reports['Report']], index=reports[ID])
    paths, _, _, _ = verified_protocol_oof(root)
    oof = read_predictions(paths, folds)
    pairs = {f: sampling.finding_pairs(labels, folds, oof, languages, f) for f in sampling.PRIORITY}
    return pairs, gold_uids


def build(root=ROOT, settings=sampling.Settings()):
    manifest_path, out = root / MANIFEST, root / OUT
    if manifest_path.exists():
        raise FileExistsError('manifest_v1 is sealed; a new sample is a new version')
    changed, _ = baseline.verify(root, 'v1')
    if changed:
        raise ValueError(f'inputs differ from the frozen baseline: {changed}')

    pairs, gold_uids = load_pairs(root)
    calls = pd.read_parquet(root / 'data/meta/llm_labels.parquet').set_index(ID)
    scanners = splits.scanner_key(pd.read_parquet(root / INDEX, columns=[
        ID, 'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength']))

    exposed, prior_pairs, scopes = exposure(root, gold_uids)
    reserve, reserve_short = sampling.draw_reserve(pairs, set(exposed[ID]), settings)
    reserved = set(reserve[ID])
    sample, sample_short, population = sampling.draw_audit(pairs, reserved, prior_pairs, settings)

    def describe(frame):
        frame = frame.copy()
        frame['scanner'] = frame[ID].map(scanners).fillna('unknown')
        frame['weight_bucket'] = frame.weight.round(2).astype(str)
        frame['extractor_call'] = [
            {1.0: 'yes', 0.0: 'no'}.get(calls.at[u, f], 'unclear_or_missing')
            for u, f in zip(frame[ID], frame.finding)]
        return frame
    reserve, sample = describe(reserve), describe(sample)

    # Pending v1 checks not already in the priority sample: the secondary workstream.
    in_sample = set(zip(sample[ID], sample.finding))
    not_sampled = pd.Series([(u, f) not in in_sample for u, f in zip(scopes[ID], scopes.finding)],
                            index=scopes.index)
    secondary = scopes[(scopes.review_scope != 'full_report') & not_sampled]

    out.mkdir(parents=True, exist_ok=True)
    files = {'reserve': reserve, 'sample': sample, 'exposure': exposed, 'secondary_queue': secondary}
    hashes = {}
    for name, frame in files.items():
        path = out / f'{name}.parquet'
        frame.reset_index(drop=True).to_parquet(path, index=False)
        hashes[f'{OUT}/{name}.parquet'] = {'sha256': sha(path), 'rows': len(frame)}

    counts = lambda f, cols: {' | '.join(map(str, k)): int(v) for k, v in f.groupby(cols).size().items()}
    manifest = {
        'schema_version': 1,
        'sealed': '2026-09-23',
        'settings': settings.to_dict(),
        'bound_documents': {k: {'path': v, 'sha256': sha(root / v)} for k, v in BOUND.items()},
        'exposure': {
            'studies': int(exposed[ID].nunique()),
            'by_source': counts(exposed, ['source']),
            'limitation': ('Only label_review_v1 and the gold 58 are recorded. Reports read ad hoc '
                           'during earlier development were not logged, so freshness is the '
                           'absence of a record, not a proof.')},
        'reserve': {'pairs': counts(reserve, ['finding', 'stratum']),
                    'studies_protected': len(reserved), 'shortfalls': reserve_short},
        'audit': {'population_pairs': population,
                  'pairs': counts(sample, ['finding', 'role', 'provenance']),
                  'controls_meeting_suspect_criteria': counts(
                      sample[sample.role == 'control'], ['finding', 'meets_false_positive', 'meets_missed_positive']),
                  'shortfalls': sample_short},
        'secondary_queue_pairs': len(secondary),
        'coverage': {'sample': sampling.coverage(sample, ['language', 'scanner', 'weight_bucket', 'extractor_call']),
                     'reserve': sampling.coverage(reserve.assign(role=reserve.stratum),
                                                  ['language', 'weight_bucket', 'extractor_call'])},
        'files': hashes,
        'blinding': ('The case files carry targets and protocol OOF. The reviewer works only '
                     'from packets built without them (plan step 4).'),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    return manifest


def main():
    manifest = build()
    print(f'sealed {MANIFEST}')
    print(f"  exposed studies {manifest['exposure']['studies']}; reserved studies "
          f"{manifest['reserve']['studies_protected']}")
    for finding in sampling.PRIORITY:
        print(f"  {finding}: population {manifest['audit']['population_pairs'][finding]} pairs")
    print('  pairs by finding | role | provenance:')
    for key, n in manifest['audit']['pairs'].items():
        print(f'    {key}: {n}')
    for item in manifest['reserve']['shortfalls'] + manifest['audit']['shortfalls']:
        print(f"  SHORTFALL {item['finding']} {item['stratum']}: wanted {item['wanted']}, available {item['available']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
