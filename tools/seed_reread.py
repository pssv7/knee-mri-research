"""Re-read earlier screen deltas against seed-to-seed variation of the frozen recipe.

Declared before the seed results arrived: seed SD per finding over seeds {0,1,2}
(standard and explicit-label AUC); a screen delta is "within seed variation" when
|delta| <= 2 x SD. Three seeds make the SD rough; the table says so.

    python tools/seed_reread.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.silence_mask import MASKED, explicit_truth
from src.validation.proxy import ID, LABELS, strict_aucs

DESIGN = ROOT / 'outputs/distinct_sampling_pair_v1/control'
SEEDS = {'seed0': ROOT / 'outputs/kaggle_distinct_training_pair_v1_results/control',
         'seed1': ROOT / 'outputs/seed_repeat_v1_results/seed1',
         'seed2': ROOT / 'outputs/seed_repeat_v1_results/seed2'}


def outer_frames():
    roles = pd.read_csv(DESIGN / 'roles.csv')
    labels = pd.read_parquet(DESIGN / 'proxy_labels.parquet')
    outer = roles[roles.role == 'outer'].sort_values(ID)[ID]
    full = (labels.set_index(ID).loc[outer, LABELS] >= .5).to_numpy(float)
    return outer, full, explicit_truth(labels, outer).to_numpy(float)


def seed_table():
    outer, full, explicit = outer_frames()
    rows = {}
    for name, directory in SEEDS.items():
        p = pd.read_parquet(directory / 'outer_predictions.parquet').set_index(ID).loc[outer, LABELS].to_numpy(float)
        rows[name] = {'full': strict_aucs(full, p)[0], 'explicit': strict_aucs(explicit, p)[0]}
    table = {}
    for kind in ('full', 'explicit'):
        values = np.array([rows[n][kind] for n in SEEDS])
        table[kind] = {'by_seed': {n: dict(zip(LABELS, map(float, rows[n][kind]))) for n in SEEDS},
                       'sd': dict(zip(LABELS, map(float, values.std(axis=0, ddof=1)))),
                       'macro_by_seed': dict(zip(SEEDS, map(float, values.mean(axis=1)))),
                       'masked_mean_sd': float(values[:, [LABELS.index(f) for f in MASKED]].mean(axis=1).std(ddof=1))}
    table['macro_sd'] = float(np.std(list(table['full']['macro_by_seed'].values()), ddof=1))
    return table


def screen_deltas():
    mirror = json.loads((ROOT / 'outputs/mirror_screen_v1_results/outer_proxy_evaluation.json').read_text())
    silence = json.loads((ROOT / 'outputs/silence_screen_v1_results/explicit_evaluation.json').read_text())
    selection = json.loads((ROOT / 'outputs/selection_probe_tiny_outer0_v1_results/selection_scores.json').read_text())
    out = {'mirror (full)': {f: v['candidate'] - v['control'] for f, v in mirror['per_finding'].items()},
           'silence (full)': {f: v['full_delta'] for f, v in silence['per_finding'].items()},
           'silence (explicit)': {f: v['explicit_delta'] for f, v in silence['per_finding'].items()}}
    for policy in ('quality5', 'all'):
        out[f'selection {policy} (full)'] = {f: v['candidate'] - v['control']
                                            for f, v in selection['policies'][policy]['per_finding'].items()}
    macros = {'mirror (full)': mirror['macro_delta'],
              'silence explicit mean of masked six': silence['decision']['explicit_mean_delta'],
              'selection quality5 (full)': selection['policies']['quality5']['macro_delta'],
              'selection all (full)': selection['policies']['all']['macro_delta']}
    return out, macros


def main():
    table = seed_table()
    deltas, macros = screen_deltas()
    report = {'seed_table': table, 'rereads': {}, 'macro_rereads': {}}
    for screen, per in deltas.items():
        kind = 'explicit' if 'explicit' in screen else 'full'
        report['rereads'][screen] = {f: {'delta': d, 'two_sd': 2 * table[kind]['sd'][f],
                                         'outside_seed_variation': abs(d) > 2 * table[kind]['sd'][f]}
                                     for f, d in per.items()}
    for screen, d in macros.items():
        sd = table['explicit']['masked_mean_sd'] if 'explicit' in screen else table['macro_sd']
        report['macro_rereads'][screen] = {'delta': d, 'two_sd': 2 * sd, 'outside_seed_variation': abs(d) > 2 * sd}
    out = ROOT / 'outputs/seed_repeat_v1_results/seed_reread.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    r = main()
    t = r['seed_table']
    print('macro by seed:', {k: round(v, 4) for k, v in t['full']['macro_by_seed'].items()}, '| SD', round(t['macro_sd'], 4))
    print('seed SD (full):', {k: round(v, 3) for k, v in t['full']['sd'].items()})
    print('seed SD (explicit):', {k: round(v, 3) for k, v in t['explicit']['sd'].items()})
    print('\nmacro-level:')
    for s, v in r['macro_rereads'].items():
        print(f"  {s:38s} {v['delta']:+.4f} vs 2SD {v['two_sd']:.4f} -> {'OUTSIDE' if v['outside_seed_variation'] else 'within'}")
    print('\nper-finding deltas outside 2 x seed SD:')
    for s, per in r['rereads'].items():
        hits = {f: round(v['delta'], 3) for f, v in per.items() if v['outside_seed_variation']}
        print(f'  {s:28s} {hits if hits else "none"}')
