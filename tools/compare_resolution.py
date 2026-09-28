"""Validate independent resolution runs and measure their matched held-out AUC."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.resolution import sha256
from tools.compare_oof import read_predictions, compare


def validate_contracts(contracts):
    if len(contracts) != 2 or [c['cache_size'] for c in contracts] != [224, 384]:
        raise ValueError('expected 224/384 contracts in that order')
    values = []
    for contract in contracts:
        c = dict(contract)
        size = c.pop('cache_size')
        spec = dict(c.pop('experiment'))
        if spec.pop('name') != f'resolution{size}' or spec['series_selection'] != 'protocol':
            raise ValueError('resolution experiment identity or selection mismatch')
        if (c['epochs'] != 12 or c['effective_batch'] != 8 or not c['gradient_checkpointing']
                or c['cache_slices'] != 24 or not c['initial_weights_sha256']
                or not c['labels_sha256']):
            raise ValueError('training contract differs from resolution screen')
        values.append((c, spec))
    if values[0] != values[1]:
        raise ValueError('settings or initial weights differ beyond resolution')


def run(control, candidate, labels_path, output):
    paths = [Path(control), Path(candidate)]
    contracts = [json.loads((p / 'run_contract.json').read_text()) for p in paths]
    validate_contracts(contracts)
    fold = contracts[0]['fold']
    for p in paths:
        receipt = json.loads((p.parent / 'resolution_run.json').read_text())
        if receipt['state'] != 'complete' or receipt['smoke']:
            raise ValueError('requires completed full training arms')
    metadata = []
    for p, contract in zip(paths, contracts):
        checkpoint = p / f'convnext_fold{fold}.pt'
        meta = json.loads(Path(str(checkpoint) + '.json').read_text())
        if meta['sha256'] != sha256(checkpoint) or meta['cache_size'] != contract['cache_size']:
            raise ValueError('checkpoint provenance mismatch')
        if meta['experiment'] != contract['experiment']:
            raise ValueError('experiment metadata differs')
        metadata.append(meta)
    for key in ('train_uids', 'val_uids', 'fold', 'labels'):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f'unmatched {key}')
    if set(metadata[0]['train_uids']) & set(metadata[0]['val_uids']):
        raise ValueError('training/validation overlap')
    if sha256(labels_path) != contracts[0]['labels_sha256']:
        raise ValueError('label matrix changed')
    folded = [pd.read_parquet(p / 'folds.parquet') for p in paths]
    pd.testing.assert_frame_equal(*folded)
    predictions = [read_predictions([p / f'oof_fold{fold}.parquet'], f) for p, f in zip(paths, folded)]
    if any(set(p.index) != set(metadata[0]['val_uids']) for p in predictions):
        raise ValueError('incomplete held-out predictions')
    table, summary = compare(pd.read_parquet(labels_path), *predictions)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError('undefined AUC')
    delta = summary['macro']['candidate'] - summary['macro']['baseline']
    summary['screening_gate'] = {'passed': delta >= .005 and bool((table.candidate_delta >= -.02).all()),
                                 'required_macro_delta': .005, 'maximum_finding_drop': .02,
                                 'note': 'Standalone allocation gate; requires replication and runtime review.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'per_finding.csv', index_label='finding')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'candidate', 'labels', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.control, args.candidate, args.labels, args.out)
