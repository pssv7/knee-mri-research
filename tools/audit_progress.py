"""Where the solo report audit stands, and the one-time control-interval trigger.

    .venv/bin/python tools/audit_progress.py status
    .venv/bin/python tools/audit_progress.py trigger
    .venv/bin/python tools/audit_progress.py draw-extra --finding "PF OA"

`status` lists each prepared batch: rendered, stage A and stage B ingested,
and when its cases become due for the 7-day re-review.

`trigger` applies decision D4. Once all 50 original controls of a finding
have a pass-1 stage B, it computes the Wilson interval of their incorrect-
target share, stores that first assessment, and never recomputes it.

`draw-extra` draws the single extra batch of 50 controls for a finding whose
stored assessment fired. Pairs already reviewed as enriched cases reuse that
review; the rest get a pass-1 batch through
`build_blinded_packets.py prepare --pass 1 --batch X1`.

Counts, dates and intervals only; no report text.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import audit_sampling as sampling
from src.labels import audit_stats, ledger
from tools import build_blinded_packets as bp
from tools import build_priority_audit as audit
from tools import freeze_label_baseline as baseline
from tools.build_label_review import sha

ID = config.ID_COL
TRIGGER = 'outputs/label_quality/audit_v1/trigger_v1.json'
EXTRA = 'outputs/label_quality/audit_v1/extra_controls.parquet'
EXTRA_RECORD = 'docs/label_quality/extra_controls_v1.json'


def status(root):
    keys = bp.all_keys(root)
    rows = []
    for (pass_no, batch), group in keys.groupby(['pass', 'batch'], sort=True):
        p = bp.paths(root, pass_no, batch)
        a = pd.read_parquet(p['stage_a']) if p['stage_a'].exists() else None
        b = pd.read_parquet(p['stage_b']) if p['stage_b'].exists() else None
        due = None
        if a is not None and pass_no == 1:
            due = (datetime.fromisoformat(a.ingested_at.iloc[0]) + ledger.REREVIEW_DELAY).date().isoformat()
        rows.append({'pass': int(pass_no), 'batch': batch, 'cases': len(group),
                     'rendered': (p['packet'] / 'packet.html').exists(),
                     'stage_a': a.ingested_at.iloc[0][:10] if a is not None else '',
                     'stage_b': b.ingested_at.iloc[0][:10] if b is not None else '',
                     're_review_due': due or ''})
    return pd.DataFrame(rows)


def control_results(root):
    """Pass-1 stage-B results of the original controls, by finding."""
    keys = bp.all_keys(root)
    if keys.empty:
        return pd.DataFrame(columns=[ID, 'finding', 'final_target_correct'])
    controls = keys[(keys['pass'] == 1) & (keys.role == 'control')]
    frames = []
    for batch in sorted(set(controls.batch)):
        p = bp.paths(root, 1, batch)
        if p['stage_b'].exists():
            frames.append(controls[controls.batch == batch].merge(
                pd.read_parquet(p['stage_b'])[['case_id', 'final_target_correct']], on='case_id'))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(controls.columns) + ['final_target_correct'])


def assess(root, now=None):
    """Decision D4: store each finding's first complete assessment; never recompute it."""
    now = now or datetime.now(timezone.utc)
    path = root / TRIGGER
    stored = json.loads(path.read_text()) if path.exists() else {}
    manifest = json.loads((root / audit.MANIFEST).read_text())
    results = control_results(root)
    for finding in sampling.PRIORITY:
        if finding in stored:
            continue
        wanted = int(manifest['audit']['pairs'].get(f'{finding} | control | random_draw', 0))
        done = results[results.finding == finding]
        if len(done) < wanted or not wanted:
            continue
        errors = sum(1 for v in done.final_target_correct if v is not None and not v)
        stored[finding] = {**audit_stats.trigger(errors, len(done)), 'assessed_at': now.isoformat()}
    if stored:
        path.write_text(json.dumps(stored, indent=2, sort_keys=True) + '\n')
    return stored


def draw_extra(root, finding, now=None, settings=sampling.Settings()):
    now = now or datetime.now(timezone.utc)
    assessment = assess(root, now).get(finding)
    if assessment is None:
        raise ValueError(f'{finding}: the interval is not assessed yet (all 50 controls need a pass-1 stage B)')
    if not assessment['fires']:
        raise ValueError(f"{finding}: interval width {assessment['width']:.3f} is within 0.20; no extra draw")
    extra_path = root / EXTRA
    existing = pd.read_parquet(extra_path) if extra_path.exists() else pd.DataFrame(columns=['finding'])
    if (existing.finding == finding).any():
        raise FileExistsError(f'{finding}: its one extra control batch is already drawn')
    changed, _ = baseline.verify(root, 'v1')
    if changed:
        raise ValueError(f'inputs differ from the frozen baseline: {changed}')
    manifest = json.loads((root / audit.MANIFEST).read_text())
    pairs, _ = audit.load_pairs(root)
    reserved = set(pd.read_parquet(root / audit.OUT / 'reserve.parquet')[ID])
    sample = pd.read_parquet(root / audit.OUT / 'sample.parquet')
    pop = pairs[finding]
    pop = pop[pop.fold.isin(settings.discovery_folds) & ~pop[ID].isin(reserved)].reset_index(drop=True)
    if len(pop) != manifest['audit']['population_pairs'][finding]:
        raise ValueError(f'{finding}: population no longer matches the sealed manifest')
    original = set(sample[(sample.finding == finding) & (sample.role == 'control')][ID])
    remaining = pop[~pop[ID].isin(original)].reset_index(drop=True)
    rng = sampling.rng_for(settings, 'extra_controls', finding)
    n = min(settings.controls, len(remaining))
    chosen = remaining.iloc[sorted(rng.choice(len(remaining), size=n, replace=False))].copy()
    reviewed = set(zip(sample[ID], sample.finding))
    chosen['role'] = 'extra_control'
    chosen['inclusion_probability'] = (len(original) + n) / len(pop)
    chosen['already_reviewed'] = [(u, finding) in reviewed for u in chosen[ID]]
    chosen['drawn_at'] = now.isoformat()
    combined = pd.concat([existing, chosen], ignore_index=True) if len(existing) else chosen
    combined.to_parquet(extra_path, index=False)
    record_path = root / EXTRA_RECORD
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record[finding] = {'drawn_at': now.isoformat(), 'pairs': n,
                       'already_reviewed_as_enriched': int(chosen.already_reviewed.sum()),
                       'trigger': {k: assessment[k] for k in ('errors', 'controls', 'width')},
                       'file_sha256_after_draw': sha(extra_path)}
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('action', choices=['status', 'trigger', 'draw-extra'])
    parser.add_argument('--finding', choices=sampling.PRIORITY)
    args = parser.parse_args(argv)
    if args.action == 'status':
        table = status(ROOT)
        print(table.to_string(index=False) if len(table) else 'no batches prepared')
    elif args.action == 'trigger':
        stored = assess(ROOT)
        for finding in sampling.PRIORITY:
            s = stored.get(finding)
            print(f'{finding}: ' + ('not assessable yet' if s is None else
                  f"{s['errors']}/{s['controls']} incorrect, 95% [{s['lower']:.3f}, {s['upper']:.3f}], "
                  f"width {s['width']:.3f} -> {'EXTRA BATCH DUE' if s['fires'] else 'no extra batch'}"))
    else:
        if not args.finding:
            parser.error('draw-extra needs --finding')
        chosen = draw_extra(ROOT, args.finding)
        print(f'{args.finding}: drew {len(chosen)} extra controls, {int(chosen.already_reviewed.sum())} '
              'already reviewed as enriched; prepare the rest with --pass 1 --batch X1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
