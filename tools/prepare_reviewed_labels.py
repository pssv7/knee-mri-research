"""Build a separate reviewed-label candidate and stage a 224 protocol pair."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.labels.reviewed import apply_reviewed
from tools.prepare_kaggle_training_pair import training_source

# Exact excerpts checked against full reports. Unsupported/ambiguous positives
# are deliberately absent: absence of report evidence alone is not a negative.
# The excerpts are report text, so they live in an ignored data file rather
# than in version control; the hash binds this code to exactly those eleven.
EVIDENCE_PATH = ROOT / 'data/meta/reviewed_labels_v1_evidence.json'
EVIDENCE_SHA256 = '57075bed8c54b2b0c25abd6b8547905bda7fdee23c2906a645c1a2690557e59b'


def load_evidence(path=EVIDENCE_PATH):
    if hashlib.sha256(path.read_bytes()).hexdigest() != EVIDENCE_SHA256:
        raise ValueError('Reviewed evidence file changed')
    return json.loads(path.read_text(encoding='utf-8'))


def stage_source(notebook, label_hash, corrections, arm, fold=1):
    source = training_source(notebook, 'baseline', label_hash, fold=fold, match_slice=True)
    source = source.replace(f'/kaggle/working/baseline_fold{fold}',
                            f'/kaggle/working/reviewed_{arm}_fold{fold}')
    tree = ast.parse(source)
    changed = 0
    correction_hash = hashlib.sha256(json.dumps(corrections, sort_keys=True).encode()).hexdigest()
    for pos in range(len(tree.body)-1, -1, -1):
        node = tree.body[pos]
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'index' for t in node.targets):
            hook = f'''import hashlib
import json
import numpy as np
{inspect.getsource(apply_reviewed)}
_review_corrections = {corrections!r}
_fixed_split = folded[["StudyInstanceUID", "fold"]].copy()
_unchanged_validation = folded.loc[folded.fold == {fold}].copy(deep=True)
if {arm == 'candidate'!r}:
    folded = apply_reviewed(folded, _fixed_split, _review_corrections, {fold})
pd.testing.assert_frame_equal(_fixed_split, folded[["StudyInstanceUID", "fold"]])
pd.testing.assert_frame_equal(_unchanged_validation, folded.loc[folded.fold == {fold}])
pathlib.Path("label_review_contract.json").write_text(json.dumps({{"arm": {arm!r}, "corrections_sha256": {correction_hash!r}, "original_labels_sha256": {label_hash!r}, "correction_count": {len(corrections) if arm == 'candidate' else 0}, "validation_labels_changed": False, "folds_changed": False}}, indent=2))
'''
            tree.body[pos:pos] = ast.parse(hook).body
            changed += 1
    if changed != 1:
        raise ValueError('StudyIndex construction changed; no safe insertion point')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def build():
    root = ROOT / 'outputs/reviewed_labels_v1'
    root.mkdir(parents=True, exist_ok=True)
    queue = [json.loads(line) for line in (ROOT / 'outputs/label_review_v1/review_queue.jsonl').read_text().splitlines()]
    labels_path = ROOT / 'data/meta/train_labels.parquet'
    labels = pd.read_parquet(labels_path)
    folded = pd.read_parquet(ROOT / 'outputs/ensemble_16_24/reconstructed_folds.parquet')
    reports = pd.read_csv(ROOT / 'data/meta/train.csv').set_index('StudyInstanceUID')
    evidence_by_id = load_evidence()
    corrections = []
    for row in queue:
        if row['review_id'] not in evidence_by_id:
            continue
        evidence = evidence_by_id[row['review_id']]
        text = str(reports.at[row['StudyInstanceUID'], 'Report'])
        if evidence not in text or text != row['report']:
            raise ValueError('Review source changed')
        corrections.append({'review_id': row['review_id'], 'StudyInstanceUID': row['StudyInstanceUID'],
            'finding': row['finding'], 'old_target': row['target'], 'new_target': 0.,
            'evidence': evidence, 'rationale': row['review_note'],
            'report_sha256': hashlib.sha256(text.encode()).hexdigest()})
    if len(corrections) != len(evidence_by_id):
        raise ValueError('Missing reviewed correction')
    candidate = apply_reviewed(labels, folded, corrections, 1, reports)
    candidate.to_parquet(root / 'train_labels_reviewed_v1.parquet', index=False)
    (root / 'corrections.json').write_text(json.dumps(corrections, ensure_ascii=False, indent=2)+'\n')
    notebook = json.loads((ROOT / 'notebooks/08_finetune.ipynb').read_text())
    label_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    for arm in ('control', 'candidate'):
        source = stage_source(notebook, label_hash, corrections, arm)
        compile(source, arm, 'exec')
        (root / f'{arm}_fold1.py').write_text(source)
    receipt = {'state': 'staged_not_trained', 'corrections': len(corrections),
        'original_labels_sha256': label_hash,
        'candidate_labels_sha256': hashlib.sha256((root/'train_labels_reviewed_v1.parquet').read_bytes()).hexdigest(),
        'reserved_folds': [1,3], 'recipe': '224px protocol ConvNeXt V1, 16 slices, five series, 12 epochs, effective batch eight',
        'scope': 'Eleven report-grounded corrections, not a corpus-wide automatic relabeler. Original weights and validation labels retained. Fixed original split computed before changes.'}
    (root / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    build()
