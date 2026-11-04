import json

import pandas as pd
import pytest

from src.validation.proxy import ID, LABELS
from tools import summarize_fresh_extraction as summary


def test_fresh_summary_keeps_rejections_in_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    keydir = tmp_path / 'outputs/report_label_evaluation_v1/custodian'
    keydir.mkdir(parents=True)
    pd.DataFrame([{'case_id': c, ID: c, 'cohort': 'control', 'language_heuristic': 'en'} for c in ['a', 'b']]).to_json(keydir / 'fresh_key.jsonl', orient='records', lines=True)
    pd.DataFrame([{'case_id': c, 'finding': f, 'old_categorical_call': None,
                   'old_fused_target': .1} for c in ['a', 'b'] for f in LABELS]).to_json(keydir / 'fresh_old_extractor.jsonl', orient='records', lines=True)
    manifest = {'cases': ['a', 'b'], 'config': {'model': 'synthetic'}}
    records = {'a': {'case_id': 'a', 'status': 'accepted', 'charged_usd_upper': .02,
                     'response': {'id': 'response-a', 'usage': {'input_tokens': 10, 'output_tokens': 20}},
                     'assertions': {f: {'report_states': 'present'} for f in LABELS}},
               'b': {'case_id': 'b', 'status': 'rejected', 'charged_usd_upper': .03,
                     'response': {'id': 'response-b', 'usage': {'input_tokens': 10, 'output_tokens': 20}},
                     'validation_error': 'synthetic invalid quote'}}
    monkeypatch.setattr(summary.fresh, 'load', lambda *a: (manifest, {}, records))
    monkeypatch.setattr(summary.fresh.base, 'carried_cost', lambda *a: .5)
    run = tmp_path / 'run'; run.mkdir(); (run / 'manifest.json').write_text('{}')
    result = summary.summarize(run, tmp_path / 'result')
    counts = result['all_cohorts_descriptive_only']
    assert counts['slots'] == 24 and counts['accepted_slots'] == counts['unavailable_slots'] == 12
    assert counts['new_present_old_silent'] == 12
    assert counts['definite_call_agreement_fraction'] is None
    assert result['shared_accounted_usd_upper'] == pytest.approx(.55)
    assert not result['opus_review_of_this_fresh_cohort']
    assert not result['independent_accuracy_established']
    del records['b']
    with pytest.raises(ValueError, match='sealed'):
        summary.summarize(run, tmp_path / 'incomplete')
    assert not (tmp_path / 'incomplete').exists()
