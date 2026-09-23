"""Blinded review packets for the solo report audit (plan step 4).

    prepare --pass 1 --batch A     assign pseudonymous IDs, order, translation requests
    render  --pass 1 --batch A     the stage-A packet: report, translation, question
    reveal  --pass 1 --batch A     the stage-B packet, only after stage A is ingested
    prepare --pass 2 --batch R1    blind re-review of controls and proposed changes,
                                   only for pairs whose first reading is 7+ days old

Pass 1 has two batches. Batch A holds all 150 random controls mixed with 75
enriched cases, so the reviewer can't tell which cases are controls; batch B
holds the remaining enriched cases. Controls are therefore read first, which
starts their re-review clock early.

Packets are built from an explicit allow-list: a case ID that encodes nothing,
the finding asked about, the report and its translation. Targets, weights,
scores, extractor output, strata, roles and earlier decisions never reach a
stage-A packet. The key linking IDs to studies lives in `review/keys/`,
outside the packet directory. Nothing here prints report text.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import secrets
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import audit_sampling, ledger, translate

ID = config.ID_COL
REVIEW = 'outputs/label_quality/review'
SAMPLE = 'outputs/label_quality/audit_v1/sample.parquet'
BATCH_A_ENRICHED = 75
PACKET_FIELDS = ('case_id', 'finding', 'language', 'report', 'translation')


def paths(root, pass_no, batch):
    base = root / REVIEW
    name = f'pass{pass_no}_{batch}'
    return {'keys': base / 'keys' / f'{name}.parquet',
            'requests': base / 'translate' / f'{name}_requests.jsonl',
            'cache': base / 'translate' / 'cache.parquet',
            'packet': base / 'packets' / name,
            'stage_a': base / 'ledger' / f'{name}_stage_a.parquet',
            'stage_b': base / 'ledger' / f'{name}_stage_b.parquet'}


def reports(root):
    frame = pd.read_csv(root / 'data/meta/train.csv', usecols=[ID, 'Report'])
    return frame.set_index(ID)['Report'].astype(str)


def new_ids(n, taken=()):
    ids, taken = [], set(taken)
    while len(ids) < n:
        token = 'K' + secrets.token_hex(3).upper()
        if token not in taken:
            taken.add(token)
            ids.append(token)
    return ids


def all_keys(root):
    found = sorted((root / REVIEW / 'keys').glob('pass*_*.parquet'))
    return pd.concat([pd.read_parquet(p) for p in found], ignore_index=True) if found else pd.DataFrame()


def write_keys(root, pass_no, batch, pairs, now):
    p = paths(root, pass_no, batch)
    if p['keys'].exists():
        raise FileExistsError(f'{p["keys"].name} exists; a prepared batch is never redrawn')
    pairs = pairs[[ID, 'finding', 'role', 'language']].reset_index(drop=True)
    order = list(range(len(pairs)))
    secrets.SystemRandom().shuffle(order)
    keys = pairs.iloc[order].reset_index(drop=True)
    existing = all_keys(root)
    keys.insert(0, 'case_id', new_ids(len(keys), existing.case_id if len(existing) else ()))
    keys['pass'], keys['batch'], keys['prepared_at'] = pass_no, batch, now.isoformat()
    text = reports(root)
    keys['report_sha256'] = [translate.report_hash(text[u]) for u in keys[ID]]
    p['keys'].parent.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(p['keys'], index=False)
    cache = load_cache(p['cache'])
    requests = [{'case_id': c, 'language': lang, 'report': text[u]}
                for c, u, lang, h in zip(keys.case_id, keys[ID], keys.language, keys.report_sha256)
                if lang != 'en' and h not in cache]
    p['requests'].parent.mkdir(parents=True, exist_ok=True)
    with p['requests'].open('w', encoding='utf-8') as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return keys, len(requests)


def prepare_pass1(root, batch, now):
    sample = pd.read_parquet(root / SAMPLE)
    enriched = sample[sample.role != 'control']
    rng = audit_sampling.rng_for(audit_sampling.Settings(), 'batch_A', 'all')
    chosen = enriched.iloc[sorted(rng.choice(len(enriched), size=BATCH_A_ENRICHED, replace=False))]
    if batch == 'A':
        pairs = pd.concat([sample[sample.role == 'control'], chosen])
    elif batch == 'B':
        pairs = enriched.drop(index=chosen.index)
    else:
        raise ValueError('pass 1 has batches A and B')
    return write_keys(root, 1, batch, pairs, now)


def prepare_pass2(root, batch, now):
    """Controls and every proposed change whose first reading is at least 7 days old."""
    keys = all_keys(root)
    first = keys[keys['pass'] == 1]
    rows = []
    for (b,), group in first.groupby(['batch']):
        p = paths(root, 1, b)
        if not (p['stage_a'].exists() and p['stage_b'].exists()):
            continue
        a, bb = pd.read_parquet(p['stage_a']), pd.read_parquet(p['stage_b'])
        merged = group.merge(a[['case_id', 'ingested_at']], on='case_id').merge(
            bb[['case_id', 'disposition']], on='case_id')
        due = [ledger.rereview_due(datetime.fromisoformat(t), now) for t in merged.ingested_at]
        wanted = (merged.role == 'control') | (merged.disposition != 'keep')
        rows.append(merged[pd.Series(due, index=merged.index) & wanted])
    eligible = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[ID, 'finding'])
    done = set(zip(keys[keys['pass'] == 2][ID], keys[keys['pass'] == 2].finding)) if len(keys) else set()
    eligible = eligible[[(u, f) not in done for u, f in zip(eligible[ID], eligible.finding)]]
    if eligible.empty:
        raise ValueError('nothing is due for re-review yet (first readings must be 7 days old)')
    return write_keys(root, 2, batch, eligible, now)


def load_cache(path):
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return dict(zip(frame.report_sha256, frame.translation))


def add_translations(root, pass_no, batch, results_path):
    """Merge a notebook-22 result file into the translation cache, after checking it."""
    p = paths(root, pass_no, batch)
    requests = [json.loads(line) for line in p['requests'].read_text(encoding='utf-8').splitlines() if line]
    results = [json.loads(line) for line in Path(results_path).read_text(encoding='utf-8').splitlines() if line]
    problems = translate.check_results(requests, results)
    if problems:
        raise ValueError('translation file rejected:\n  ' + '\n  '.join(problems))
    by_case = {r['case_id']: r for r in results}
    frame = pd.DataFrame([{'report_sha256': translate.report_hash(q['report']),
                           'translation': translate.clean(by_case[q['case_id']]['translation']),
                           'model': by_case[q['case_id']].get('model', '')} for q in requests])
    if p['cache'].exists():
        frame = pd.concat([pd.read_parquet(p['cache']), frame]).drop_duplicates('report_sha256', keep='first')
    frame.to_parquet(p['cache'], index=False)
    return len(requests)


def packet_rows(root, pass_no, batch, allow_missing=False):
    """The only data a stage-A packet may carry: exactly PACKET_FIELDS."""
    p = paths(root, pass_no, batch)
    keys = pd.read_parquet(p['keys'])
    text, cache = reports(root), load_cache(p['cache'])
    rows = []
    for row in keys.to_dict('records'):
        translation = cache.get(row['report_sha256'], '')
        if row['language'] != 'en' and not translation and not allow_missing:
            raise ValueError(f"{row['case_id']}: no translation yet; run notebook 22 and --translations")
        rows.append({'case_id': row['case_id'], 'finding': row['finding'], 'language': row['language'],
                     'report': text[row[ID]], 'translation': translation})
    assert all(set(r) == set(PACKET_FIELDS) for r in rows)
    return rows


STYLE = ('<style>body{max-width:960px;margin:24px auto;font:15px/1.5 system-ui,sans-serif;padding:0 16px}'
         'article{border-top:2px solid #999;padding:14px 0}pre{white-space:pre-wrap;background:#f6f6f6;'
         'padding:10px}.tr{background:#eef4ff}.me{background:#fff7e0;padding:8px}'
         '@media (prefers-color-scheme:dark){body{background:#161616;color:#e6e6e6}pre{background:#222}'
         '.tr{background:#1d2633}.me{background:#332b12}}</style>')


def render(root, pass_no, batch, allow_missing=False):
    p = paths(root, pass_no, batch)
    rows = packet_rows(root, pass_no, batch, allow_missing)
    cards = []
    for i, r in enumerate(rows, 1):
        translation = (f'<p><b>Machine translation (aid only; quote the original)</b></p>'
                       f'<pre class="tr">{html.escape(r["translation"])}</pre>') if r['translation'] else ''
        cards.append(f'<article><h2>{i}. {r["case_id"]} — {html.escape(r["finding"])}</h2>'
                     f'<p>What does this report state about <b>{html.escape(r["finding"])}</b>? '
                     f'Apply docs/report_review_rubric.md sections 2–5.</p>'
                     f'<p><b>Original report ({html.escape(r["language"])})</b></p>'
                     f'<pre>{html.escape(r["report"])}</pre>{translation}</article>')
    p['packet'].mkdir(parents=True, exist_ok=True)
    (p['packet'] / 'packet.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>Review pass {pass_no} batch {batch}</title>{STYLE}'
        f'<h1>Stage A — pass {pass_no}, batch {batch} ({len(rows)} cases)</h1>'
        '<p>Fill stage_a.csv in this folder. Record what the report states; nothing else is shown '
        'on purpose.</p>' + ''.join(cards), encoding='utf-8')
    template = pd.DataFrame([{c: '' for c in ledger.STAGE_A} for _ in rows])
    template['case_id'] = [r['case_id'] for r in rows]
    template['finding'] = [r['finding'] for r in rows]
    template.to_csv(p['packet'] / 'stage_a.csv', index=False)
    return len(rows)


def reveal(root, pass_no, batch):
    """Stage B: the reviewer's own stage-A answer beside the target and extractor calls.

    Shown: final target, weight, the categorical call with its severity and
    evidence, and the probability-mode score. Never shown: model OOF scores,
    sampling role or stratum, or another pass's decisions.
    """
    p = paths(root, pass_no, batch)
    if not p['stage_a'].exists():
        raise ValueError('stage A has not been ingested; the reveal comes after the blind reading')
    keys = pd.read_parquet(p['keys'])
    a = pd.read_parquet(p['stage_a']).set_index('case_id')
    labels = pd.read_parquet(root / 'data/meta/train_labels.parquet').set_index(ID)
    calls = pd.read_parquet(root / 'data/meta/llm_labels.parquet').set_index(ID)
    probs = pd.read_parquet(root / 'data/meta/llm_probs.parquet').set_index(ID)
    cards = []
    for row in keys.to_dict('records'):
        u, f, c = row[ID], row['finding'], row['case_id']
        mine = ''.join(f'<div>{k}: {html.escape(str(a.at[c, k]))}</div>' for k in ledger.STAGE_A[2:])
        call = calls.at[u, f]
        call = 'yes' if call == 1 else 'no' if call == 0 else 'unclear'
        cards.append(
            f'<article><h2>{c} — {html.escape(f)}</h2><div class="me"><b>Your stage-A reading</b>{mine}</div>'
            f'<p>Final target <b>{labels.at[u, f]:.3f}</b> · weight {labels.at[u, f + "_weight"]:.3f} · '
            f'final target counted correct: <b>{ledger.final_target_correct(a.at[c, "report_states"], labels.at[u, f])}</b></p>'
            f'<p>Categorical extractor: {call} · severity {html.escape(str(calls.at[u, f + "_severity"]))} · '
            f'probability mode {probs.at[u, f]:.3f}</p>'
            f'<p>Extractor evidence:</p><pre>{html.escape(str(calls.at[u, f + "_evidence"]))}</pre></article>')
    (p['packet'] / 'reveal.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>Reveal pass {pass_no} batch {batch}</title>{STYLE}'
        f'<h1>Stage B — pass {pass_no}, batch {batch}</h1><p>Fill stage_b.csv. Rubric section 6 '
        'decides which dispositions are allowed.</p>' + ''.join(cards), encoding='utf-8')
    template = pd.DataFrame([{c: '' for c in ledger.STAGE_B} for _ in range(len(keys))])
    template['case_id'], template['finding'] = keys.case_id.values, keys.finding.values
    template.to_csv(p['packet'] / 'stage_b.csv', index=False)
    return len(keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('action', choices=['prepare', 'render', 'reveal'])
    parser.add_argument('--pass', dest='pass_no', type=int, choices=[1, 2], required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--translations', help='notebook-22 results to merge before rendering')
    parser.add_argument('--allow-missing-translation', action='store_true')
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    p = paths(ROOT, args.pass_no, args.batch)
    if args.action == 'prepare':
        keys, n = (prepare_pass1 if args.pass_no == 1 else prepare_pass2)(ROOT, args.batch, now)
        print(f'prepared {len(keys)} cases; {n} reports need translation -> {p["requests"].relative_to(ROOT)}')
    elif args.action == 'render':
        if args.translations:
            print(f'merged {add_translations(ROOT, args.pass_no, args.batch, args.translations)} translations')
        print(f'rendered {render(ROOT, args.pass_no, args.batch, args.allow_missing_translation)} cases -> '
              f'{p["packet"].relative_to(ROOT)}/packet.html and stage_a.csv')
    else:
        print(f'revealed {reveal(ROOT, args.pass_no, args.batch)} cases -> '
              f'{p["packet"].relative_to(ROOT)}/reveal.html and stage_b.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
