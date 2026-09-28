"""Strict fixed-recipe evaluation, with checkpoint and OOF provenance.

Provenance JSON contains `incumbent` and `candidates` lists of family objects:
{name, weight, runs: [{fold, checkpoint, oof, contract, receipt}]}.
Paths are relative to the JSON. Legacy runs may omit contract/receipt only
when `legacy_reconstructed: true` is explicitly declared on the run.
Candidate runs always require completed receipts and contracts.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.model.artifacts import sha256
from tools.compare_oof import compare, read_predictions


def fixed_mix(frames, weights):
    if not frames or len(frames) != len(weights):
        raise ValueError("One positive weight per family is required")
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid recipe weights")
    frames = [frame.sort_index() for frame in frames]
    first = frames[0]
    for frame in frames:
        if (not frame.index.is_unique or not first.index.equals(frame.index)
                or not first['fold'].equals(frame['fold'])):
            raise ValueError("Recipes require exactly matching unique held-out IDs and folds")
        values = frame[config.LABELS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid probabilities")
    result = first.copy()
    result[config.LABELS] = sum(w * f[config.LABELS] for w, f in zip(weights / weights.sum(), frames))
    return result


def validated_family(family, root, folded, folds, candidate, labels_hash=None):
    runs = family['runs']
    if len(runs) != len(folds) or sorted(r['fold'] for r in runs) != sorted(folds):
        raise ValueError("Each family requires exact requested fold coverage")
    frames, sources = [], []
    for run in runs:
        fold = run['fold']
        checkpoint, oof = [root / run[k] for k in ('checkpoint', 'oof')]
        sidecar = Path(str(checkpoint) + '.json')
        meta = json.loads(sidecar.read_text())
        if (meta['sha256'] != sha256(checkpoint) or meta['fold'] != fold
                or meta['labels'] != config.LABELS):
            raise ValueError("Checkpoint hash, fold or labels differ")
        val = set(folded.loc[folded.fold == fold, config.ID_COL])
        train = set(folded.loc[folded.fold != fold, config.ID_COL])
        if (len(meta['val_uids']) != len(val) or set(meta['val_uids']) != val
                or len(meta['train_uids']) != len(train) or set(meta['train_uids']) != train
                or set(meta['train_uids']) & val):
            raise ValueError("Checkpoint train/validation membership differs")
        source = {"family": family['name'], "fold": fold,
                  "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": meta['sha256'],
                  "sidecar_sha256": sha256(sidecar), "oof": str(oof.resolve()), "oof_sha256": sha256(oof)}
        if candidate or not run.get('legacy_reconstructed', False):
            contract_path, receipt_path = [root / run[k] for k in ('contract', 'receipt')]
            contract = json.loads(contract_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            if (receipt['state'] != 'complete' or receipt.get('smoke', False)
                    or contract['fold'] != fold or contract['experiment'] != meta['experiment']):
                raise ValueError("Incomplete run or mismatched contract")
            if candidate:
                if contract['seed'] != 0 or contract['epochs'] != 12 or not contract['initial_weights_sha256']:
                    raise ValueError('Primary candidate must use seed 0 and the complete schedule')
                if labels_hash is not None and contract.get('labels_sha256') != labels_hash:
                    raise ValueError('Candidate label hash differs')
                saved = pd.read_parquet(contract_path.parent / 'folds.parquet')
                pd.testing.assert_series_equal(saved.set_index(config.ID_COL)['fold'].sort_index(),
                                               folded.set_index(config.ID_COL)['fold'].sort_index())
            source.update(contract_sha256=sha256(contract_path), receipt_sha256=sha256(receipt_path),
                          training_seed=contract['seed'])
        else:
            source['limitation'] = 'Legacy training exclusion reconstructed, not independently proven'
        frame = read_predictions([oof], folded)
        if set(frame.index) != val:
            raise ValueError("Incomplete or extra OOF studies; intersections are forbidden")
        frames.append(frame)
        sources.append(source)
    return pd.concat(frames).sort_index(), sources


def recipe_result(labels, baseline, candidate, weights, draws):
    table, summary = compare(labels, baseline, candidate, draws=draws, seed=0)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Undefined per-finding AUC")
    # compare() also evaluates a secondary blend; it is deliberately omitted.
    per_fold = {}
    for fold in sorted(baseline.fold.unique()):
        a, b = [f.loc[f.fold == fold] for f in (baseline, candidate)]
        t, _ = compare(labels, a, b, draws=1)
        per_fold[str(int(fold))] = {"baseline": float(t.baseline.mean()),
                                  "recipe": float(t.candidate.mean()),
                                  "delta": float(t.candidate_delta.mean()),
                                  "per_finding": t[['baseline', 'candidate', 'candidate_delta']].to_dict('index')}
    delta = summary['macro']['candidate'] - summary['macro']['baseline']
    ci = summary['paired_macro_delta_95ci']['candidate']
    return {"weights": weights, "baseline_macro": summary['macro']['baseline'],
            "macro": summary['macro']['candidate'], "delta": delta,
            "paired_delta_95ci": ci, "bootstrap_draws": draws,
            "per_fold": per_fold,
            "per_finding": table[['baseline', 'candidate', 'candidate_delta']].to_dict('index'),
            "allocation_gate_passed": delta >= .003 and ci[0] > 0,
            "deployment_gate_passed": delta >= .003 and ci[0] > 0
                and all(v['delta'] >= -.005 for v in per_fold.values())}


def run(incumbent_oof, candidate_oofs, folds_path, labels_path, provenance_path,
        requested_folds, output, draws=1000):
    if draws != 1000:
        raise ValueError("Campaign decisions require the predeclared 1,000 bootstrap draws")
    folds = list(requested_folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
        raise ValueError("Choose unique folds 0 through 4")
    folded = pd.read_parquet(folds_path)
    if not folded[config.ID_COL].is_unique or len(folded) != 4349 or set(folded.fold) != set(range(5)):
        raise ValueError("Requires the complete 4,349-study reconstructed split")
    labels = pd.read_parquet(labels_path)
    if not labels[config.ID_COL].is_unique:
        raise ValueError("Duplicate labels")
    provenance_path = Path(provenance_path)
    spec = json.loads(provenance_path.read_text())
    incumbent_families, candidates, receipts = [], [], []
    names = [f['name'] for f in spec['incumbent'] + spec['candidates']]
    if len(set(names)) != len(names) or len(candidate_oofs) != len(spec['candidates']):
        raise ValueError("Unique families and one explicit OOF per candidate are required")
    for kind, target in (('incumbent', incumbent_families), ('candidates', candidates)):
        for family in spec[kind]:
            frame, source = validated_family(family, provenance_path.parent, folded, folds, kind == 'candidates', sha256(labels_path))
            target.append(frame)
            receipts.extend(source)
    incumbent_weights = [float(f['weight']) for f in spec['incumbent']]
    if len(set(incumbent_weights)) != 1:
        raise ValueError("Campaign incumbent must retain equal family weights")
    baseline = fixed_mix(incumbent_families, incumbent_weights)
    expected = set(folded.loc[folded.fold.isin(folds), config.ID_COL])
    for paths, reference in [(incumbent_oof, baseline), *zip(candidate_oofs, candidates)]:
        frame = read_predictions([paths], folded)
        if set(frame.index) != expected:
            raise ValueError("Explicit OOF must cover exactly requested folds")
        if not np.allclose(frame.sort_index()[config.LABELS], reference[config.LABELS], atol=1e-7, rtol=0):
            raise ValueError("Explicit OOF differs from checkpoint-bound family recipe")
    n = len(incumbent_families)
    results = {}
    for family, candidate in zip(spec['candidates'], candidates):
        name = family['name']
        results[name] = {
            'standalone': recipe_result(labels, baseline, candidate, {name: 1.0}, draws),
            'allocation_50_50': recipe_result(labels, baseline, fixed_mix([baseline, candidate], [1, 1]),
                                               {'incumbent': .5, name: .5}, draws),
            'deployment_equal_family': recipe_result(labels, baseline, fixed_mix([baseline, candidate], [n, 1]),
                                                     {'incumbent': n / (n + 1), name: 1 / (n + 1)}, draws)}
    if len(candidates) > 1:
        names = [f['name'] for f in spec['candidates']]
        total = n + len(candidates)
        results['combined_deployment_equal_family'] = recipe_result(
            labels, baseline, fixed_mix([baseline] + candidates, [n] + [1] * len(candidates)),
            {'incumbent': n / total, **{name: 1 / total for name in names}}, draws)
    report = {'folds': sorted(folds), 'studies': len(expected), 'recipes': results, 'sources': receipts,
              'source_hashes': {str(p): sha256(p) for p in [incumbent_oof, *candidate_oofs, folds_path, labels_path, provenance_path]},
              'limitation': 'Mined-label fixed-model bootstrap; legacy exclusion is reconstructed. Seed variability is separate.',
              'leaderboard_context': 'Historical OOF-to-LB ratios imply roughly +0.08–0.12 OOF blend gain to close +0.041; not a forecast or gate.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('incumbent', 'splits', 'labels', 'provenance', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate', nargs='+', required=True)
    parser.add_argument('--folds', nargs='+', type=int, required=True)
    args = parser.parse_args()
    run(args.incumbent, args.candidate, args.splits, args.labels, args.provenance, args.folds, args.out)
