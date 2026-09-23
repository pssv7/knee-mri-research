"""The report-audit results and the go/no-go decision (plan step 6).

    .venv/bin/python tools/audit_report.py

Assembles every ingested review (both stages, both passes) into one ledger
and writes `docs/label_audit_v2.md`:

- control estimates with Wilson intervals, split into missed positives and
  false positives, on the original random controls (and, separately, the
  controls enlarged by decision D4 when that batch is complete);
- the enriched strata's counts, never pooled into a rate;
- error classes ranked by agreed corrections, missed mentions by language;
- assertion classes, and coverage of the reviewed pairs;
- agreement between the two passes: on the random controls, and separately
  on the selectively re-reviewed proposed changes;
- unresolved items, adaptive sampling, and the decision D5 go/no-go.

The assembled ledger, which carries evidence quotes, goes to ignored
outputs for the candidate step. The tracked report holds counts, rates and
pseudonymous case IDs only; it never quotes a report. It can be rerun as
reviews arrive; the version committed at the decision is the one that
counts.
"""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import audit_sampling as sampling
from src.labels import audit_stats, ledger
from tools import audit_progress as progress
from tools import build_blinded_packets as bp
from tools import build_priority_audit as audit

ID = config.ID_COL
DEADLINE = date(2026, 10, 10)
REPORT = 'docs/label_audit_v2.md'
ASSEMBLED = 'outputs/label_quality/audit_v1/ledger_assembled.parquet'
THRESHOLD = 5


def records(root):
    """One row per reviewed case: keys, stage A and stage B (where ingested)."""
    keys = bp.all_keys(root)
    rows = []
    for (pass_no, batch), group in keys.groupby(['pass', 'batch'], sort=True):
        p = bp.paths(root, pass_no, batch)
        if not p['stage_a'].exists():
            continue
        a = pd.read_parquet(p['stage_a']).drop(columns=['finding']).rename(
            columns={'ingested_at': 'a_at', 'reviewer': 'a_reviewer', 'source_sha256': 'a_source'})
        frame = group.merge(a, on='case_id', how='inner')
        if p['stage_b'].exists():
            b = pd.read_parquet(p['stage_b']).drop(columns=['finding']).rename(
                columns={'ingested_at': 'b_at', 'reviewer': 'b_reviewer', 'source_sha256': 'b_source'})
            frame = frame.merge(b, on='case_id', how='left')
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def direction(state, target):
    if pd.isna(target):
        return None
    if state == 'present' and target < ledger.POSITIVE:
        return 'missed_positive'
    if state in ('negated', 'present_below_threshold') and target >= ledger.POSITIVE:
        return 'false_positive'
    return None


def pair_table(recs):
    """Pass 1 and pass 2 side by side, one row per (study, finding)."""
    if recs.empty:
        return pd.DataFrame()
    recs = recs.copy()
    targets = recs['old_target'] if 'old_target' in recs else pd.Series([None] * len(recs), index=recs.index)
    recs['direction'] = [direction(s, t) for s, t in zip(recs.report_states, targets)]
    first = recs[recs['pass'] == 1].set_index([ID, 'finding'])
    second = recs[recs['pass'] == 2].set_index([ID, 'finding'])
    table = first.add_prefix('p1_').join(second.add_prefix('p2_'), how='left')
    agreed = []
    for row in table.itertuples():
        b1 = {'disposition': getattr(row, 'p1_disposition', None), 'proposed_target': getattr(row, 'p1_proposed_target', ''),
              'proposed_weight': getattr(row, 'p1_proposed_weight', ''), 'old_weight': getattr(row, 'p1_old_weight', None)}
        b2 = {'disposition': getattr(row, 'p2_disposition', None), 'proposed_target': getattr(row, 'p2_proposed_target', ''),
              'proposed_weight': getattr(row, 'p2_proposed_weight', ''), 'old_weight': getattr(row, 'p2_old_weight', None)}
        ok = isinstance(b1['disposition'], str) and isinstance(b2['disposition'], str) and ledger.agreed_change(b1, b2)
        agreed.append(ok)
    table['agreed_change'] = agreed
    table['agreed_target_correction'] = table.agreed_change & (
        table['p1_disposition'] == 'correct_target' if 'p1_disposition' in table else False)
    return table.reset_index()


def design(root):
    sample = pd.read_parquet(root / audit.OUT / 'sample.parquet')
    cols = [ID, 'finding', 'role', 'provenance', 'inclusion_probability', 'language', 'scanner',
            'weight_bucket', 'extractor_call']
    frame = sample[[c for c in cols if c in sample]].rename(columns={'role': 'design_role'})
    extra_path = root / progress.EXTRA
    extra = pd.read_parquet(extra_path) if extra_path.exists() else None
    return frame, extra


def rate(k, n):
    if not n:
        return None
    lo, hi = audit_stats.wilson(k, n)
    return {'k': int(k), 'n': int(n), 'rate': k / n, 'lower': lo, 'upper': hi}


def judged_wrong(value):
    """True only for a computed False; None (no judgement from text) is not an error."""
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and value == False  # noqa: E712


def control_block(table, planned):
    """Original random controls of one finding: rates, agreed corrections, decidability."""
    empty = {'planned': planned, 'with_stage_b': 0, 'incorrect': None, 'missed_positive': None,
             'false_positive': None, 'no_judgement_from_text': 0, 'proposed_changes_pass1': 0,
             'awaiting_pass2': 0, 'agreed_target_errors': 0, 'agreed_weight_changes': 0, 'decidable': False}
    if not len(table) or 'p1_disposition' not in table:
        return empty
    with_b = table[table.p1_disposition.notna()]
    n = len(with_b)
    if not n:
        return empty
    wrong = with_b[with_b.p1_final_target_correct.map(judged_wrong)]
    proposals = with_b[with_b.p1_disposition != 'keep']
    pending = proposals[proposals.p2_disposition.isna()] if 'p2_disposition' in proposals else proposals
    return {
        'planned': planned, 'with_stage_b': n,
        'incorrect': rate(len(wrong), n),
        'missed_positive': rate(int((wrong.p1_direction == 'missed_positive').sum()), n),
        'false_positive': rate(int((wrong.p1_direction == 'false_positive').sum()), n),
        'no_judgement_from_text': int(with_b.p1_final_target_correct.map(
            lambda v: v is None or (isinstance(v, float) and pd.isna(v))).sum()),
        'proposed_changes_pass1': len(proposals), 'awaiting_pass2': len(pending),
        'agreed_target_errors': int(with_b.agreed_target_correction.sum()),
        'agreed_weight_changes': int((with_b.agreed_change & (with_b.p1_disposition == 'correct_weight')).sum()),
        'decidable': bool(planned and n >= planned and len(pending) == 0),
    }


def summarise(root, now=None):
    now = now or datetime.now(timezone.utc)
    manifest = json.loads((root / audit.MANIFEST).read_text())
    recs = records(root)
    table = pair_table(recs)
    frame, extra = design(root)
    if len(table):
        table = table.merge(frame, on=[ID, 'finding'], how='left')
        # Extra controls (D4) live only in the review keys; carry their role across.
        table['design_role'] = table.design_role.fillna(table.p1_role)
    out = {'generated': now.isoformat(), 'reviewed_pairs': len(table),
           'controls': {}, 'enlarged_controls': {}, 'enriched': {}, 'classes': {}, 'missed_by_language': {},
           'assertions': {}, 'agreement': {}, 'unresolved': {}, 'coverage': {}}
    for finding in sampling.PRIORITY:
        planned = int(manifest['audit']['pairs'].get(f'{finding} | control | random_draw', 0))
        part = table[table.finding == finding] if len(table) else table
        controls = part[part.design_role == 'control'] if len(part) else part
        out['controls'][finding] = control_block(controls, planned)
        if extra is not None and (extra.finding == finding).any():
            ex = extra[extra.finding == finding]
            ex_rows = part.merge(ex[[ID]], on=ID) if len(part) else part
            both = pd.concat([controls, ex_rows])
            block = control_block(both, planned + len(ex))
            block['complete'] = block['with_stage_b'] >= planned + len(ex)
            out['enlarged_controls'][finding] = block
        for role in ('suspected_false_positive', 'suspected_missed_positive'):
            sub = part[part.design_role == role] if len(part) else part
            with_b = sub[sub['p1_disposition'].notna()] if len(sub) and 'p1_disposition' in sub else sub.iloc[0:0]
            out['enriched'][f'{finding} | {role}'] = {
                'reviewed': len(with_b),
                'incorrect': int(with_b.p1_final_target_correct.map(judged_wrong).sum()) if len(with_b) else 0,
                'agreed_target_errors': int(with_b.agreed_target_correction.sum()) if len(with_b) else 0,
                'from_label_review_v1': int((with_b.provenance == 'label_review_v1').sum()) if len(with_b) else 0}
    if len(table) and 'p1_error_class' in table:
        judged = table[table.p1_error_class.notna()]
        for (f, c), g in judged.groupby(['finding', 'p1_error_class']):
            if c != 'none':
                out['classes'][f'{f} | {c}'] = {'pass1': len(g), 'agreed_target_errors': int(g.agreed_target_correction.sum())}
        missed = judged[judged.p1_error_class == 'missed_mention']
        out['missed_by_language'] = {f'{f} | {l}': len(g) for (f, l), g in missed.groupby(['finding', 'language'])}
        out['unresolved'] = {
            'unresolved': int((judged.p1_error_class == 'unresolved').sum()),
            'report_image_disagreement': int((judged.p1_error_class == 'report_image_disagreement').sum()),
            'proposed_once_not_agreed': int(((judged.p1_disposition != 'keep') & judged.p2_disposition.notna()
                                             & ~judged.agreed_change).sum()) if 'p2_disposition' in judged else 0}
    if len(table):
        group = table.design_role.map(lambda r: 'control' if r == 'control' else 'enriched')
        out['assertions'] = {f'{f} | {g} | {s}': len(x) for (f, g, s), x in
                             table.assign(group=group).groupby(['finding', 'group', 'p1_report_states'])}
        for column in ('language', 'scanner', 'weight_bucket', 'extractor_call'):
            if column in table:
                out['coverage'][column] = {str(k): int(v) for k, v in table[column].value_counts().items()}
    if len(table) and 'p2_report_states' in table:
        twice = table[table.p2_report_states.notna()]
        ctrl = twice[twice.design_role == 'control']
        if len(ctrl):
            out['agreement']['controls'] = {
                'pairs': len(ctrl),
                'state_raw': float((ctrl.p1_report_states == ctrl.p2_report_states).mean()),
                'state_kappa': audit_stats.cohen_kappa(ctrl.p1_report_states, ctrl.p2_report_states)}
            if 'p2_disposition' in ctrl and ctrl.p2_disposition.notna().all():
                out['agreement']['controls'].update({
                    'disposition_raw': float((ctrl.p1_disposition == ctrl.p2_disposition).mean()),
                    'disposition_kappa': audit_stats.cohen_kappa(ctrl.p1_disposition, ctrl.p2_disposition)})
        changes = twice[(twice.design_role != 'control') & (twice.p1_disposition != 'keep')] \
            if 'p1_disposition' in twice else twice.iloc[0:0]
        if len(changes):
            out['agreement']['proposed_changes'] = {'pairs': len(changes),
                                                    'agreed': int(changes.agreed_change.sum())}
    trigger_path = root / progress.TRIGGER
    out['trigger'] = json.loads(trigger_path.read_text()) if trigger_path.exists() else {}
    out['extra_draws'] = json.loads((root / progress.EXTRA_RECORD).read_text()) \
        if (root / progress.EXTRA_RECORD).exists() else {}
    out['decision'] = audit_stats.go_no_go(
        {f: {'decidable': b.get('decidable', False), 'agreed_target_errors': b.get('agreed_target_errors', 0)}
         for f, b in out['controls'].items()}, THRESHOLD, deadline_passed=now.date() > DEADLINE)
    out['secondary_queue_pending'] = int(manifest.get('secondary_queue_pairs', 0))
    keys = bp.all_keys(root)
    out['delay_waived'] = int(keys[(keys['pass'] == 2)].get('delay_waived', pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) \
        if len(keys) else 0
    return out, table


def fmt(r):
    if not r:
        return '—'
    return f"{r['k']}/{r['n']} = {r['rate']:.1%} [{r['lower']:.1%}, {r['upper']:.1%}]"


def markdown(s):
    d = s['decision']
    lines = ['# Report audit v2: Lateral Meniscus, Contusion, PF OA', '',
             f"Generated {s['generated'][:16]} UTC by `tools/audit_report.py`. Counts, rates and intervals only; "
             'no report text. One reviewer throughout: agreement is consistency, not independence.', '',
             '## Decision (D5)', '',
             f"**{d['decision'].upper()}**" + (f" — {', '.join(d['findings'])}" if d['findings'] else '')
             + (' (pass 2 ran without the 7-day delay; see the D6 limitation below)' if s.get('delay_waived') else ''), '',
             f'Go needs {THRESHOLD}+ target errors agreed by both passes among a finding\'s original random '
             'controls. Weight-only changes do not count.', '',
             '## Random controls (population estimates)', '',
             '| Finding | Reviewed | Label judged incorrect | Missed positive | False positive | '
             'No judgement from text | Agreed target errors | Awaiting pass 2 | Decidable |', '|---|---|---|---|---|---|---|---|---|']
    for f, b in s['controls'].items():
        lines.append(f"| {f} | {b.get('with_stage_b', 0)}/{b['planned']} | {fmt(b.get('incorrect'))} | "
                     f"{fmt(b.get('missed_positive'))} | {fmt(b.get('false_positive'))} | "
                     f"{b.get('no_judgement_from_text', 0)} | {b['agreed_target_errors']} | "
                     f"{b.get('awaiting_pass2', 0)} | {'yes' if b['decidable'] else 'no'} |")
    lines += ['', '95% Wilson intervals. The first-pass "incorrect" rate counts every reading the rubric '
              'judges against the final label; only agreed target errors can become corrections.', '']
    if s['enlarged_controls']:
        lines += ['### Enlarged controls (D4)', '']
        for f, b in s['enlarged_controls'].items():
            lines.append(f"- {f}: {fmt(b.get('incorrect'))}; " + ('complete' if b['complete']
                         else 'incomplete, so the estimate is reported as provisional') + '.')
        lines.append('')
    lines += ['## Enriched strata (selected for looking wrong; never a rate)', '',
              '| Finding and stratum | Reviewed | Label judged incorrect | Agreed target errors | From label_review_v1 |',
              '|---|---|---|---|---|']
    for k, e in s['enriched'].items():
        lines.append(f"| {k} | {e['reviewed']} | {e['incorrect']} | {e['agreed_target_errors']} | {e['from_label_review_v1']} |")
    lines += ['', '## Error classes, ranked by agreed target errors', '']
    ranked = sorted(s['classes'].items(), key=lambda kv: (-kv[1]['agreed_target_errors'], -kv[1]['pass1']))
    lines += [f"- {k}: {v['agreed_target_errors']} agreed, {v['pass1']} in pass 1" for k, v in ranked] or ['- none yet']
    lines += ['', 'Missed mentions by language (pass 1): ' +
              (', '.join(f'{k}: {v}' for k, v in sorted(s['missed_by_language'].items())) or 'none yet'), '',
              '## Assertion classes (pass 1)', '']
    lines += [f'- {k}: {v}' for k, v in sorted(s['assertions'].items())] or ['- none yet']
    lines += ['', '## Agreement between passes', '']
    c = s['agreement'].get('controls')
    if c:
        kappa = lambda v: 'undefined' if v is None else f'{v:.2f}'
        lines.append(f"- Random controls ({c['pairs']} pairs): reading {c['state_raw']:.1%}, κ {kappa(c['state_kappa'])}"
                     + (f"; disposition {c['disposition_raw']:.1%}, κ {kappa(c['disposition_kappa'])}"
                        if 'disposition_raw' in c else '') + '.')
    else:
        lines.append('- Random controls: no second pass yet.')
    pc = s['agreement'].get('proposed_changes')
    lines.append(f"- Selectively re-reviewed proposed changes: {pc['agreed']}/{pc['pairs']} agreed." if pc
                 else '- Proposed changes: no second pass yet.')
    lines += ['', 'The two lines are reported apart on purpose: controls are a random sample, proposed '
              'changes were re-reviewed because the first pass flagged them.', '',
              '## Unresolved and not acted on', '']
    u = s['unresolved']
    lines.append(f"- unresolved {u.get('unresolved', 0)}, report–image disagreement "
                 f"{u.get('report_image_disagreement', 0)}, proposed in one pass but not agreed "
                 f"{u.get('proposed_once_not_agreed', 0)}." if u else '- none yet.')
    lines += ['', '## Adaptive sampling', '']
    if s['trigger']:
        for f, t in s['trigger'].items():
            lines.append(f"- {f}: first assessment {t['errors']}/{t['controls']}, width {t['width']:.3f} "
                         f"({'extra batch due' if t['fires'] else 'no extra batch'}).")
    else:
        lines.append('- Interval trigger (D4) not yet assessed.')
    for f, e in s['extra_draws'].items():
        lines.append(f"- {f}: {e['pairs']} extra controls drawn, {e['already_reviewed_as_enriched']} "
                     'reusing an enriched review. The go/no-go stays on the original controls.')
    if s.get('delay_waived'):
        lines += ['', f"**Limitation (D6):** the 7-day delay before pass 2 was waived for {s['delay_waived']} "
                  're-reviewed pairs. Their agreement partly reflects recall of the first reading, which '
                  'inflates agreed corrections and biases the decision toward go.']
    lines += ['', 'Limits: the enriched strata are selected by model disagreement and describe mechanisms, '
              'not prevalence. Prior exposure is known only from label_review_v1 and the gold 58. '
              f"Secondary workstream: {s['secondary_queue_pending']} pending v1 checks, reported separately "
              'when done.', '', '## Coverage of reviewed pairs', '']
    for column, counts in s['coverage'].items():
        lines.append(f'- {column}: ' + ', '.join(f'{k} {v}' for k, v in list(counts.items())[:12]))
    return '\n'.join(lines) + '\n'


def main():
    summary, table = summarise(ROOT)
    if len(table):
        (ROOT / ASSEMBLED).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(ROOT / ASSEMBLED, index=False)
    (ROOT / REPORT).write_text(markdown(summary), encoding='utf-8')
    d = summary['decision']
    print(f"wrote {REPORT}: {summary['reviewed_pairs']} reviewed pairs; decision {d['decision']}"
          + (f" ({', '.join(d['findings'])})" if d['findings'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
