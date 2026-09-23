"""Ingest a filled stage-A or stage-B response sheet into the review ledger.

    .venv/bin/python tools/ingest_ledger.py --pass 1 --batch A --stage a PATH/stage_a.csv
    .venv/bin/python tools/ingest_ledger.py --pass 1 --batch A --stage b PATH/stage_b.csv

Every case in the batch must appear exactly once, and every row must pass
src/labels/ledger.py. A sheet with any error is rejected whole, so a
half-valid batch can't slip in. An ingested stage is never overwritten: the
ingestion time is what the 7-day re-review rule measures from, and the blind
reading must stay as it was first given. Errors name the case and the field,
never the quoted text.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.labels import ledger
from tools.build_blinded_packets import paths, reports

ID = config.ID_COL
REVIEWER = 'participant'


def read_sheet(path, columns):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f'sheet lacks columns {missing}')
    return frame[columns].apply(lambda s: s.str.strip())


def check_cases(frame, keys):
    problems = []
    dup = frame.case_id[frame.case_id.duplicated()]
    problems += [f'{c}: appears more than once' for c in sorted(set(dup))]
    problems += [f'{c}: missing from the sheet' for c in sorted(set(keys.case_id) - set(frame.case_id))]
    problems += [f'{c}: not in this batch' for c in sorted(set(frame.case_id) - set(keys.case_id))]
    wrong = frame.merge(keys[['case_id', 'finding']], on='case_id', suffixes=('', '_key'))
    problems += [f'{c}: finding altered' for c in wrong.case_id[wrong.finding != wrong.finding_key]]
    return problems


def ingest(root, pass_no, batch, stage, sheet, now=None):
    now = now or datetime.now(timezone.utc)
    p = paths(root, pass_no, batch)
    target = p['stage_a'] if stage == 'a' else p['stage_b']
    if target.exists():
        raise FileExistsError(f'{target.name} is already ingested and is never overwritten')
    keys = pd.read_parquet(p['keys'])
    frame = read_sheet(sheet, ledger.STAGE_A if stage == 'a' else ledger.STAGE_B)
    problems = check_cases(frame, keys)
    by_case = keys.set_index('case_id')
    if stage == 'a':
        text = reports(root)
        for row in frame.to_dict('records'):
            if row['case_id'] in by_case.index:
                uid = by_case.at[row['case_id'], ID]
                problems += [f"{row['case_id']}: {e}" for e in ledger.validate_stage_a(row, text[uid])]
    else:
        if not p['stage_a'].exists():
            raise ValueError('stage A must be ingested before stage B')
        a = pd.read_parquet(p['stage_a']).set_index('case_id')
        labels = pd.read_parquet(root / 'data/meta/train_labels.parquet').set_index(ID)
        correct, old_t, old_w = [], [], []
        for row in frame.to_dict('records'):
            c = row['case_id']
            if c not in by_case.index:
                correct.append(None); old_t.append(None); old_w.append(None)
                continue
            uid, f = by_case.at[c, ID], by_case.at[c, 'finding']
            t, w = float(labels.at[uid, f]), float(labels.at[uid, f + '_weight'])
            errors, ok = ledger.validate_stage_b(a.loc[c].to_dict(), row, t, w)
            problems += [f'{c}: {e}' for e in errors]
            correct.append(ok); old_t.append(t); old_w.append(w)
        frame['final_target_correct'], frame['old_target'], frame['old_weight'] = correct, old_t, old_w
    if problems:
        raise ValueError(f'{len(problems)} problem(s); nothing ingested:\n  ' + '\n  '.join(problems))
    frame['ingested_at'] = now.isoformat()
    frame['reviewer'] = REVIEWER
    frame['source_sha256'] = hashlib.sha256(Path(sheet).read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pass', dest='pass_no', type=int, choices=[1, 2], required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--stage', choices=['a', 'b'], required=True)
    parser.add_argument('sheet')
    args = parser.parse_args(argv)
    try:
        frame = ingest(ROOT, args.pass_no, args.batch, args.stage, args.sheet)
    except (ValueError, FileExistsError) as exc:
        print(exc)
        return 1
    print(f'ingested {len(frame)} rows (pass {args.pass_no}, batch {args.batch}, stage {args.stage})')
    if args.stage == 'b':
        print('  dispositions:', frame.disposition.value_counts().to_dict())
    return 0


if __name__ == '__main__':
    sys.exit(main())
