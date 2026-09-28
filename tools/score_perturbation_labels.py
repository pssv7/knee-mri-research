"""Write labels for every perturbation variant and score the probe against them.

Labels come from the frozen outer0 report-mined labels (>= 0.5), carried
through each transform by `perturb_labels.variant_labels`; the mirror is written both
ways. Reads the probe's predictions; no GPU, no new image readings.

    python tools/score_perturbation_labels.py
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import perturb_labels
from src.validation.proxy import ID, LABELS, sha

DESIGN = ROOT / 'outputs/distinct_sampling_pair_v1/control'
RESULTS = ROOT / 'outputs/perturbation_probe_tiny_outer0_v1_results/full'


def outer_truth(design=DESIGN):
    spec = json.loads((design / 'spec.json').read_text())
    for name, key in [('roles.csv', 'roles_sha256'), ('proxy_labels.parquet', 'proxy_labels_sha256')]:
        if sha(design / name) != spec[key]:
            raise ValueError(f'Frozen design changed: {name}')
    roles = pd.read_csv(design / 'roles.csv')
    outer = roles[roles.role == 'outer'].set_index(ID).sort_index()
    labels = pd.read_parquet(design / 'proxy_labels.parquet').set_index(ID)
    truth = (labels.loc[outer.index, LABELS] >= .5).astype(int)
    return truth, outer.component


def main(results=RESULTS, design=DESIGN, out=None):
    results = Path(results)
    contract = json.loads((results / 'run_contract.json').read_text())
    if sha(results / 'perturbation_predictions.parquet') != contract['predictions_sha256']:
        raise ValueError('Probe predictions changed since the run')
    predictions = pd.read_parquet(results / 'perturbation_predictions.parquet')
    truth, components = outer_truth(design)
    written = []
    for arm, name, rule in perturb_labels.labelled_arms():
        frame = perturb_labels.variant_labels(truth, name, rule).reset_index()
        written.append(frame.assign(arm=arm, variant=name, label_rule=rule or perturb_labels.LABEL_RULES[name]))
    out = Path(out) if out else results.parent / 'labelled'
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(written).to_parquet(out / 'variant_labels.parquet', index=False)
    report = perturb_labels.score_against_labels(predictions, truth, components)
    report['predictions_sha256'] = contract['predictions_sha256']
    (out / 'label_scores.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--out', type=Path)
    a = parser.parse_args()
    r = main(a.results, out=a.out)
    print(f"identity macro {r['identity_macro']:.4f} over {r['studies']} studies / {r['components']} components")
    for arm, v in r['arms'].items():
        lo, hi = v['macro_delta_95ci']
        print(f"{arm:18s} macro {v['macro']:.4f}  delta {v['macro_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
