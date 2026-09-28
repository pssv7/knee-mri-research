"""Score the series-selection check against the frozen outer0 mined labels.

Each policy is compared with `protocol5` on all 691 outer0 studies with the
project's paired component bootstrap. Studies a policy leaves unchanged keep
identical scores, so the paired delta is the effect of re-selecting the
changed studies on whole-set AUC. Exploratory: nothing is adopted from it.

    python tools/score_selection_probe.py
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import selection_policies as policies
from src.validation.proxy import ID, LABELS, paired_proxy_evaluation, sha
from tools.score_perturbation_labels import outer_truth

RESULTS = ROOT / 'outputs/selection_probe_tiny_outer0_v1_results/full'


def main(results=RESULTS, draws=2000):
    results = Path(results)
    contract = json.loads((results / 'run_contract.json').read_text())
    if sha(results / 'selection_predictions.parquet') != contract['predictions_sha256']:
        raise ValueError('Selection predictions changed since the run')
    if not contract['reproduction']['passed']:
        raise ValueError('Reference selection did not reproduce the saved predictions')
    predictions = pd.read_parquet(results / 'selection_predictions.parquet')
    truth, components = outer_truth()
    studies = list(truth.index)
    def wide(policy):
        rows = predictions[predictions.policy == policy].set_index(ID)
        if set(rows.index) != set(studies):
            raise ValueError(f'{policy}: outer studies incomplete')
        return rows.loc[studies, LABELS].to_numpy(float)
    reference = wide(policies.REFERENCE)
    report = {'studies': len(studies), 'reference_policy': policies.REFERENCE,
              'predictions_sha256': contract['predictions_sha256'], 'policies': {}}
    for policy in policies.POLICIES:
        if policy == policies.REFERENCE:
            continue
        evaluation = paired_proxy_evaluation(truth.to_numpy(float), reference, wide(policy),
                                             components.loc[studies].to_numpy(), draws=draws)
        changed = predictions[(predictions.policy == policy) & predictions.changed][ID].nunique()
        report['policies'][policy] = {'changed_studies': int(changed), **evaluation}
    (results.parent / 'selection_scores.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    a = parser.parse_args()
    r = main(a.results)
    for policy, v in r['policies'].items():
        lo, hi = v['paired_sampling_95ci']
        per = {k: round(x['candidate'] - x['control'], 3) for k, x in v['per_finding'].items()}
        print(f"{policy:10s} changed {v['changed_studies']:3d}  macro {v['candidate_macro']:.4f}  "
              f"delta {v['macro_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print('           ', per)
