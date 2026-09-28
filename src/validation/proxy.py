"""Nested proxy evaluation with explicit identity and selection exclusions.

These tools cannot establish image accuracy from report-derived labels.
Known text links are grouped conservatively; patient linkage remains unknown.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.labels.calibrate import _auc

ID = config.ID_COL
LABELS = config.LABELS
ROLES = {'train', 'selection', 'outer'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json_new(path, value):
    """Never overwrite an evidence record, even on an otherwise harmless rerun."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def linked_components(metadata, near_pairs, patient_groups=None):
    """Union known links, including transitive links to protected studies.

    Exact text can be templated across patients, so this is conservative
    quarantine/grouping, not a declaration of common patient identity.
    """
    if metadata[ID].duplicated().any() or metadata[ID].isna().any():
        raise ValueError('Study IDs must be complete and unique')
    parent = {str(u): str(u) for u in metadata[ID]}

    def root(u):
        if u not in parent:
            raise ValueError('Link references an unknown study')
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def join(a, b):
        a, b = root(str(a)), root(str(b))
        parent[max(a, b)] = min(a, b)

    for _, g in metadata.groupby('normalized_report_hash'):
        if not str(g.normalized_report_hash.iloc[0]).strip():
            continue
        ids = list(g[ID].astype(str))
        for u in ids[1:]:
            join(ids[0], u)
    for row in near_pairs.itertuples():
        join(row.study_a, row.study_b)
    if patient_groups is not None:
        if patient_groups[ID].duplicated().any() or patient_groups.patient_group.isna().any():
            raise ValueError('Patient linkage must be unique and nonmissing')
        for _, g in patient_groups.groupby('patient_group'):
            ids = list(g[ID].astype(str))
            for u in ids:
                root(u)
            for u in ids[1:]:
                join(ids[0], u)
    return pd.DataFrame({ID: list(parent), 'component': [root(u) for u in parent]})


def exclude_components(components, protected_ids):
    protected_ids = set(protected_ids)
    if not protected_ids <= set(components[ID]):
        raise ValueError('Unknown protected study')
    bad = set(components.loc[components[ID].isin(protected_ids), 'component'])
    return components[~components.component.isin(bad)].copy()


def group_folds(frame, n_folds=5, seed=20260924):
    """Balance component sizes within coarse positive-count strata.

    Uses proxies only for stratification, never model scores. Deterministic
    under row reordering. Complete linked components are never split.
    """
    if frame[ID].duplicated().any() or frame.component.isna().any():
        raise ValueError('Invalid component frame')
    if frame.component.nunique() < n_folds or n_folds < 2:
        raise ValueError('Too few components for folds')
    counts = (frame[LABELS].to_numpy(float) >= .5).sum(axis=1).clip(max=4)
    work = frame[[ID, 'component']].copy()
    work['stratum'] = counts
    groups = work.groupby('component').agg(n=(ID, 'size'), stratum=('stratum', 'max')).reset_index()
    rng = np.random.default_rng(seed)
    groups = groups.sort_values('component')
    groups['tie'] = rng.random(len(groups))
    totals = np.zeros(n_folds, int)
    assignment = {}
    for _, g in groups.groupby('stratum', sort=True):
        strata = np.zeros(n_folds, int)
        for row in g.sort_values(['n', 'tie'], ascending=[False, True]).itertuples():
            f = min(range(n_folds), key=lambda f: (strata[f], totals[f], f))
            assignment[row.component] = f
            strata[f] += row.n
            totals[f] += row.n
    return frame[ID].map(frame.set_index(ID).component).map(assignment).astype(int)


def nested_roles(frame, outer_fold, seed=20260924):
    if outer_fold not in set(frame.outer_fold):
        raise ValueError('Unknown outer fold')
    roles = frame[[ID, 'component', 'outer_fold']].copy()
    roles['role'] = 'outer'
    inside = frame[frame.outer_fold != outer_fold].copy()
    inner = group_folds(inside, seed=seed + outer_fold + 1)
    roles.loc[inside.index, 'role'] = np.where(inner.to_numpy() == 0, 'selection', 'train')
    validate_roles(roles)
    return roles


def validate_roles(roles):
    if roles[ID].duplicated().any() or roles[ID].isna().any():
        raise ValueError('Duplicate or missing split IDs')
    if set(roles.role) != ROLES or roles.component.isna().any():
        raise ValueError('Need nonempty train, selection and outer roles')
    if (roles.groupby('component').role.nunique() > 1).any():
        raise ValueError('Linked component crosses split roles')


def training_only_sharpen(raw, reference, train_ids):
    """Fit the historical quantile-sharpening rule on training studies alone.

    Held-out queries use the training empirical CDF, with midpoint ties.
    Training results reproduce the original tied-rank formula on that subset.
    """
    if raw[ID].duplicated().any() or reference[ID].duplicated().any():
        raise ValueError('Duplicate label IDs')
    train_ids = list(train_ids)
    if len(set(train_ids)) != len(train_ids) or len(train_ids) < 2:
        raise ValueError('At least two unique training IDs required')
    a, b = raw.set_index(ID), reference.set_index(ID)
    if set(a.index) != set(b.index) or not set(train_ids) <= set(a.index):
        raise ValueError('Label or training membership mismatch')
    out = raw.copy()
    for name in LABELS:
        source = np.sort(a.loc[train_ids, name].to_numpy(float))
        marginal = np.sort(b.loc[train_ids, name].to_numpy(float))
        query = raw[name].to_numpy(float)
        if not all(np.isfinite(x).all() for x in [source, marginal, query]):
            raise ValueError('Nonfinite label values')
        ranks = (np.searchsorted(source, query, side='left') +
                 np.searchsorted(source, query, side='right') - 1) / (2 * (len(source) - 1))
        ranks = np.clip(ranks, 0, 1)
        out[name] = (np.quantile(marginal, ranks) + .02 * ranks) / 1.02
    return out


def validate_prediction_provenance(roles, contract, predictions):
    validate_roles(roles)
    for role, key in [('train', 'train_uids'), ('selection', 'selection_uids'), ('outer', 'outer_uids')]:
        values = contract.get(key, [])
        if len(values) != len(set(values)) or set(values) != set(roles.loc[roles.role == role, ID]):
            raise ValueError(f'{key} does not match the frozen split')
    if contract.get('model_count') != 1:
        raise ValueError('Evaluation requires exactly one deployed checkpoint')
    if contract.get('checkpoint_selection_role') != 'selection':
        raise ValueError('Checkpoint selection must use the inner selection set')
    if contract.get('initialization') != 'original_pretrained':
        raise ValueError('Historical knee checkpoints are not a clean initialization')
    if predictions[ID].duplicated().any() or set(predictions[ID]) != set(contract['outer_uids']):
        raise ValueError('Predictions must cover exactly the outer set')
    values = predictions[LABELS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Predictions must be finite probabilities')


def strict_aucs(truth, scores):
    truth, scores = np.asarray(truth, float), np.asarray(scores, float)
    if truth.shape != scores.shape or truth.ndim != 2 or truth.shape[1] != len(LABELS):
        raise ValueError('Expected aligned twelve-finding matrices')
    if not np.isfinite(scores).all() or np.isinf(truth).any() or not np.isin(truth[np.isfinite(truth)], [0, 1]).all():
        raise ValueError('Invalid scores or truth')
    values = []
    for j in range(len(LABELS)):
        mask = np.isfinite(truth[:, j])
        values.append(_auc(truth[mask, j], scores[mask, j]))
    values = np.asarray(values, float)
    macro = float(values.mean()) if np.isfinite(values).all() else None
    return values, macro


def paired_proxy_evaluation(truth, control, candidate, components, draws=2000, seed=20260924):
    """Fixed-prediction component bootstrap; never a training-variance estimate."""
    truth, control, candidate = [np.asarray(x, float) for x in (truth, control, candidate)]
    if draws < 100 or len(components) != len(truth) or pd.isna(components).any():
        raise ValueError('Need >=100 draws and complete aligned components')
    aa, ma = strict_aucs(truth, control)
    bb, mb = strict_aucs(truth, candidate)
    components = np.asarray(components)
    units = [np.flatnonzero(components == g) for g in sorted(set(components))]
    rng = np.random.default_rng(seed)
    boot = []
    if ma is not None and mb is not None:
        for _ in range(draws):
            ix = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
            a, am = strict_aucs(truth[ix], control[ix])
            b, bm = strict_aucs(truth[ix], candidate[ix])
            if am is not None and bm is not None:
                boot.append(b - a)
    ci = None
    per_ci = [None] * len(LABELS)
    if len(boot) >= .95 * draws:
        boot = np.asarray(boot)
        ci = np.quantile(boot.mean(axis=1), [.025, .975]).tolist()
        per_ci = np.quantile(boot, [.025, .975], axis=0).T.tolist()
    return {
        'endpoint': 'mined_report_label_macro_roc_auc', 'model_count_per_arm': 1,
        'control_macro': ma, 'candidate_macro': mb,
        'macro_delta': mb - ma if ma is not None and mb is not None else None,
        'paired_sampling_95ci': ci, 'valid_bootstrap_draws': len(boot), 'requested_draws': draws,
        'per_finding': {name: {'control': float(a) if np.isfinite(a) else None,
                               'candidate': float(b) if np.isfinite(b) else None,
                               'positive': int(np.nansum(truth[:, j])),
                               'negative': int(np.sum(truth[:, j] == 0)),
                               'missing': int(np.isnan(truth[:, j]).sum()),
                               'paired_sampling_95ci': per_ci[j]}
                        for j, (name, a, b) in enumerate(zip(LABELS, aa, bb))},
        'image_gain_demonstrated': False, 'promotion': 'not_supported_by_proxy_labels',
        'limitations': ['Patient identities unavailable unless separately supplied; components are known links only.',
                       'Bootstrap conditions on fitted models; no seed, selection-history or systematic label-bias uncertainty.',
                       'Per-finding intervals are descriptive, not simultaneous noninferiority certification.']}
