"""Compare frozen singleton predictions on outer observations exactly once per output.

Refuses legacy contracts that selected checkpoints on the evaluation fold.
This command emits a report-label diagnostic and never a model promotion.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.proxy import (ID, LABELS, sha, write_json_new,
    validate_prediction_provenance, paired_proxy_evaluation)


def evaluate(design, control_dir, candidate_dir, out, draws=2000):
    design, out = Path(design), Path(out)
    if out.exists():
        raise FileExistsError('Evaluation result already exists; do not overwrite readouts')
    spec = json.loads((design / 'spec.json').read_text())
    if sha(design / 'roles.csv') != spec['roles_sha256'] or sha(design / 'proxy_labels.parquet') != spec['proxy_labels_sha256']:
        raise ValueError('Frozen split or labels changed')
    roles = pd.read_csv(design / 'roles.csv')
    outer = roles[roles.role == 'outer'].sort_values(ID)
    table = pd.read_parquet(design / 'proxy_labels.parquet')
    if table[ID].duplicated().any() or set(table[ID]) != set(roles[ID]):
        raise ValueError('Invalid label membership')
    values = table[LABELS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid proxy labels; missing values must not become negatives')
    labels = table.set_index(ID).loc[outer[ID], LABELS]
    predictions, contracts, sources = [], [], {}
    for arm, directory in [('control', control_dir), ('candidate', candidate_dir)]:
        directory = Path(directory)
        contract_path, pred_path = directory / 'run_contract.json', directory / 'outer_predictions.parquet'
        contract = json.loads(contract_path.read_text())
        if contract.get('roles_sha256') != spec['roles_sha256'] or contract.get('proxy_labels_sha256') != spec['proxy_labels_sha256']:
            raise ValueError('Run is not bound to the frozen design')
        if contract.get('outer_predictions_sha256') != sha(pred_path):
            raise ValueError('Prediction hash mismatch')
        checkpoint = directory / 'selected.pt'
        if contract.get('checkpoint_sha256') != sha(checkpoint):
            raise ValueError('Selected checkpoint hash mismatch')
        pred = pd.read_parquet(pred_path)
        validate_prediction_provenance(roles, contract, pred)
        predictions.append(pred.set_index(ID).loc[outer[ID], LABELS].to_numpy(float))
        contracts.append(contract)
        sources[arm] = {'contract_sha256': sha(contract_path), 'prediction_sha256': sha(pred_path)}
    for key in ['seed', 'training_recipe', 'initial_weights_sha256', 'runtime_versions', 'input_config_sha256']:
        if contracts[0].get(key) is None or contracts[0].get(key) != contracts[1].get(key):
            raise ValueError(f'Matched-control mismatch: {key}')
    policies = [c.get('sampling_policy', 'legacy') for c in contracts]
    if policies != ['legacy', 'legacy']:
        if policies != ['legacy', 'distinct_raw_17_23_v1']:
            raise ValueError('Invalid isolated sampler comparison')
        for key in ['raw_counts_sha256', 'runtime_code_sha256']:
            if not contracts[0].get(key) or contracts[0][key] != contracts[1].get(key):
                raise ValueError(f'Matched sampler provenance mismatch: {key}')
    report = paired_proxy_evaluation((labels.to_numpy() >= .5).astype(float), *predictions,
                                     outer.component.to_numpy(), draws=draws)
    report['sources'] = sources
    report['sampling_policies'] = policies
    report['roles_sha256'] = spec['roles_sha256']
    write_json_new(out, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['design', 'control', 'candidate', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--draws', type=int, default=2000)
    a = p.parse_args()
    r = evaluate(a.design, a.control, a.candidate, a.out, a.draws)
    print(json.dumps({k: r[k] for k in ['control_macro', 'candidate_macro', 'macro_delta', 'paired_sampling_95ci', 'promotion']}, indent=2))
