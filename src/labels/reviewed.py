"""Apply explicit reviewed corrections without changing validation or split IDs."""
import hashlib

import numpy as np


def apply_reviewed(labels, folded, corrections, heldout_fold, reports=None):
    if heldout_fold not in (1, 3):
        raise ValueError('Only reserved evaluation folds 1 and 3 are supported')
    key = 'StudyInstanceUID'
    if labels[key].duplicated().any() or folded[key].duplicated().any():
        raise ValueError('Duplicate study IDs')
    folds = folded.set_index(key)['fold']
    original = labels.set_index(key)
    if set(original.index) != set(folds.index):
        raise ValueError('Labels and fixed folds differ')
    result = original.copy(deep=True)
    seen = set()
    for correction in corrections:
        uid, finding = correction[key], correction['finding']
        if (uid, finding) in seen:
            raise ValueError('Duplicate correction')
        seen.add((uid, finding))
        if uid not in folds.index or folds.at[uid] not in (0, 2, 4):
            raise ValueError('Correction would touch a reserved or unknown study')
        if finding not in result or finding + '_weight' not in result:
            raise ValueError('Unknown finding')
        if not np.isclose(original.at[uid, finding], correction['old_target'], atol=1e-9, rtol=0):
            raise ValueError('Original target changed')
        if correction['new_target'] != 0.0 or not correction['evidence'].strip():
            raise ValueError('This candidate only supports reviewed, evidenced exclusions')
        if reports is not None:
            text = str(reports.loc[uid, 'Report'])
            if hashlib.sha256(text.encode()).hexdigest() != correction['report_sha256']:
                raise ValueError('Reviewed report changed')
            if correction['evidence'] not in text:
                raise ValueError('Evidence quote absent from report')
        result.at[uid, finding] = 0.0
    heldout = folds.index[folds == heldout_fold]
    if not result.loc[heldout].equals(original.loc[heldout]):
        raise ValueError('Validation labels changed')
    return result.reset_index()[labels.columns]
