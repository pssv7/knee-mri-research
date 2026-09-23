"""Build a reproducible report-review queue; never alter training labels."""
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from tools.compare_oof import read_predictions


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_protocol_oof(root=ROOT):
    """Locate each protocol fold's OOF file and prove it is the recorded one.

    Receipts may name a path that has since moved; the staged copy is accepted
    only when its hash matches the receipt, its sidecar declares exactly the
    fold's validation studies with none in training, and its checkpoint hash
    is the recorded one.
    """
    summary_path = root / 'data/meta/compare_protocol_incumbent_all_folds/summary.json'
    summary = json.loads(summary_path.read_text())
    folded = pd.read_parquet(root / 'outputs/ensemble_16_24/reconstructed_folds.parquet')
    paths, provenance = [], []
    for entry in summary['source_receipts']:
        if entry['family'] != 'protocol':
            continue
        path = root / entry['oof']
        if not path.exists():
            fold = entry['fold']
            path = root / f'outputs/slice_overlay_staging/fold{fold}/baseline_fold{fold}/oof_fold{fold}.parquet'
        if sha(path) != entry['oof_sha256']:
            raise ValueError(f'Protocol OOF hash mismatch: {path}')
        sidecar = path.parent / f"convnext_fold{entry['fold']}.pt.json"
        meta = json.loads(sidecar.read_text())
        expected = set(folded.loc[folded.fold == entry['fold'], config.ID_COL])
        if set(meta['val_uids']) != expected or expected & set(meta['train_uids']):
            raise ValueError('Protocol training/validation exclusion failed')
        if meta['sha256'] != entry['checkpoint_sha256']:
            raise ValueError('Protocol checkpoint association changed')
        paths.append(path)
        provenance.append({'path': str(path.relative_to(root)), 'sha256': sha(path)})
    return paths, provenance, folded, summary_path


def build():
    out = ROOT / 'outputs/label_review_v1'
    out.mkdir(parents=True, exist_ok=True)
    paths, provenance, folded, summary_path = verified_protocol_oof()
    protocol = read_predictions(paths, folded).sort_index()
    legacy_path = ROOT / 'data/meta/compare_protocol_incumbent_all_folds/incumbent_oof.parquet'
    legacy = read_predictions([legacy_path], folded).sort_index()
    if not legacy.index.equals(protocol.index):
        raise ValueError('OOF populations differ')
    labels_path = ROOT / 'data/meta/train_labels.parquet'
    reports_path = ROOT / 'data/meta/train.csv'
    labels = pd.read_parquet(labels_path).set_index(config.ID_COL)
    reports = pd.read_csv(reports_path).set_index(config.ID_COL)
    categorical = pd.read_parquet(ROOT / 'data/meta/llm_labels.parquet').set_index(config.ID_COL)
    probability = pd.read_parquet(ROOT / 'data/meta/llm_probs.parquet').set_index(config.ID_COL)
    rows = []
    # Development only. Neither held-out fold 1 nor 3 is opened for report review.
    eligible = protocol.index[protocol.fold.isin([0, 2, 4])]
    for uid in eligible:
        for finding in config.LABELS:
            target = float(labels.at[uid, finding])
            a, b = float(legacy.at[uid, finding]), float(protocol.at[uid, finding])
            direction = 'possible_missed_positive' if target < .5 else 'possible_false_positive'
            strength = min(a, b) if target < .5 else 1 - max(a, b)
            rows.append({'StudyInstanceUID': uid, 'fold': int(protocol.at[uid, 'fold']),
                'finding': finding, 'target': target,
                'weight': float(labels.at[uid, finding + '_weight']),
                'legacy_blend': a, 'protocol': b, 'direction': direction,
                'opposition_strength': strength,
                'categorical_call': None if pd.isna(categorical.at[uid, finding]) else float(categorical.at[uid, finding]),
                'categorical_evidence': str(categorical.at[uid, finding + '_evidence']),
                'probability_target': float(probability.at[uid, finding]),
                'report': str(reports.at[uid, 'Report'])})
    pairs = pd.DataFrame(rows)
    # Threshold fixed before reviewing text; require BOTH available sources to oppose.
    candidates = pairs[pairs.opposition_strength >= .7].copy()
    if len(candidates) < 200:
        raise ValueError(f'Only {len(candidates)} strong disagreements; do not silently loosen threshold')
    selected = candidates.sort_values('opposition_strength', ascending=False, kind='stable').head(200).copy()
    selected['sample'] = 'disagreement'
    # Random control comes from the remaining pair population, not just agreements.
    controls = pairs.drop(index=selected.index).sample(n=50, random_state=20260915).copy()
    controls['sample'] = 'random_control'
    packet = pd.concat([selected, controls], ignore_index=True)
    packet.insert(0, 'review_id', [f'R{i:03}' for i in range(1, len(packet)+1)])
    notes_path = ROOT / 'tools/label_review_v1_notes.json'
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    packet['review_status'] = [notes.get(i, ['evidence_screened_full_report_pending', ''])[0] for i in packet.review_id]
    packet['review_note'] = [notes.get(i, ['', ''])[1] for i in packet.review_id]
    packet['review_scope'] = ['full_report' if i in notes else 'categorical_evidence_only' for i in packet.review_id]
    packet.to_json(out / 'review_queue.jsonl', orient='records', lines=True, force_ascii=False)
    stats = {'population_studies': len(eligible), 'population_pairs': len(pairs),
        'strong_disagreements': len(candidates), 'review_pairs': len(packet),
        'discovery_folds': [0, 2, 4], 'reserved_folds': [1, 3],
        'selected_by_finding': selected.finding.value_counts().to_dict(),
        'selected_by_direction': selected.direction.value_counts().to_dict(),
        'selected_by_weight': selected.weight.value_counts().to_dict(),
        'review_status_counts': packet.review_status.value_counts().to_dict(),
        'protocol_provenance': provenance,
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [labels_path, reports_path, legacy_path, summary_path]},
        'limitation': 'Legacy blend has recorded OOF provenance but missing individual source files; independent legacy training exclusion cannot be re-proven. Protocol exclusion checked against sidecars. Correlated predictions are review signals, not ground truth. Discovery folds are not clean future validation folds.'}
    (out / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    cards = []
    for row in packet.to_dict('records'):
        cards.append('<article><h2>' + html.escape(f"{row['review_id']} — {row['finding']} ({row['sample']})") + '</h2><p>'
            + html.escape(f"Fold {row['fold']} | target {row['target']:.3f} | weight {row['weight']:.3f} | legacy {row['legacy_blend']:.3f} | protocol {row['protocol']:.3f}")
            + '</p><p>Review: ' + html.escape(row['review_status'] + ' — ' + row['review_note'])
            + '</p><p>Extractor evidence: ' + html.escape(row['categorical_evidence']) + '</p><pre>' + html.escape(row['report']) + '</pre></article>')
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Label review</title><style>body{max-width:950px;margin:30px auto;font:16px sans-serif}pre{white-space:pre-wrap}article{border-top:1px solid #aaa;padding:20px 0}</style><h1>Label review — development folds 0, 2, 4</h1><p>Review suggestions only. No labels changed. Report agreement does not establish image truth.</p>' + ''.join(cards))
    print(json.dumps({k:v for k,v in stats.items() if k not in ['protocol_provenance','input_sha256']}, indent=2))


if __name__ == '__main__':
    build()
