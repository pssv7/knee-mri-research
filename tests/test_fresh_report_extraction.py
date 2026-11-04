import json

import pytest

from src.labels.report_candidate import request_body, parse_response
from src.labels.openai_extract import request_hash
from src.validation.proxy import sha
from tools import fresh_report_extraction as fresh
from test_claude_label_pilot import response
from test_report_evaluation import packet


@pytest.fixture
def prepared_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh.base, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "verify_bundle", lambda *a: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-never-sent")
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "manifest.json").write_text('{}')
    p, _ = packet()
    ppath = tmp_path / "packet.json"; ppath.write_text(json.dumps(p))
    checks = tmp_path / fresh.CLARIFICATIONS; checks.parent.mkdir(); checks.write_text('Fixed checks')
    old = tmp_path / "old_cost.json"; old.write_text(json.dumps({"charged_usd_upper": .5}))
    adjudication = tmp_path / "adjudication.json"; adjudication.write_text('{}')
    settings = {"max_cost_usd": 10, "input_reservation_usd_per_million": 4,
                "input_usd_per_million": 2, "output_usd_per_million": 10, "cache_write_upper_usd_per_million": 4}
    manifest = {"config": settings, "prior_cost_records": [{"path": old.name, "sha256": sha(old), "charged_usd_upper": .5}],
                "source_sha256": {fresh.CLARIFICATIONS: sha(checks)},
                "packet": {"path": ppath.name, "sha256": sha(ppath)},
                "adjudication": {"path": adjudication.name, "sha256": sha(adjudication)},
                "bundle": {"path": bundle.name, "manifest_sha256": sha(bundle/'manifest.json')}, "cases": []}
    for case in p['cases']:
        body = request_body(case['report'], p['rubric'], checks.read_text())
        manifest['cases'].append({"case_id":case['case_id'],"report_sha256":case['report_sha256'],
                                  "request_sha256":request_hash(body),"reserved_usd":fresh.base.reserve_cost(body,settings)})
    out = tmp_path / "run"; out.mkdir(); (out/'manifest.json').write_text(json.dumps(manifest))
    return out


def test_candidate_rejects_hedged_definite_state_without_rewriting():
    value = response()
    payload = json.loads(value['content'][0]['text'])
    item = next(r for r in payload['findings'] if r['finding']=='ACL')
    item.update(report_states='present', certainty='probable')
    value['content'][0]['text'] = json.dumps(payload)
    with pytest.raises(ValueError, match='inconsistency'):
        parse_response(value,'The ACL is intact.')


def test_shared_budget_blinding_and_completed_resume(prepared_fresh):
    calls = []
    def send(body,key):
        assert not any(term in json.dumps(body) for term in ['case_id','old_fused_target','StudyInstanceUID'])
        calls.append(body)
        return response(body['messages'][0]['content'][0]['text'])
    assert fresh.run(prepared_fresh,.1,send)['status']=='shared_budget_limit' and not calls
    result=fresh.run(prepared_fresh,10,send)
    assert result['reports_completed']==2 and result['accepted']==2
    assert result['shared_accounted_usd_upper']==pytest.approx(.548)
    fresh.run(prepared_fresh,10,lambda *a:pytest.fail('No paid retry'))
    assert len(calls)==2


def test_invalid_quotes_are_charged_and_excluded(prepared_fresh):
    result=fresh.run(prepared_fresh,10,lambda *a:response('Fabricated quote'))
    assert result['rejected']==2 and result['accepted']==0
    assert result['shared_accounted_usd_upper']==pytest.approx(.548)
    assert fresh.run(prepared_fresh,10,lambda *a:pytest.fail('No retry'))['rejected']==2


def test_transport_failure_reserves_cost_and_blocks_retry(prepared_fresh):
    def fail(*args): raise RuntimeError('simulated timeout')
    with pytest.raises(RuntimeError,match='timeout'): fresh.run(prepared_fresh,10,fail)
    r=json.loads(next((prepared_fresh/'attempts').glob('*.json')).read_text())
    assert r['charged_usd_upper']==r['reserved_usd']
    with pytest.raises(RuntimeError,match='reconciliation'):
        fresh.run(prepared_fresh,10,lambda *a:pytest.fail('No retry'))


def test_changed_prompt_and_excess_budget_block(prepared_fresh):
    with pytest.raises(ValueError,match='cap'):
        fresh.run(prepared_fresh,10.01,lambda *a:pytest.fail('No request'))
    (fresh.ROOT/fresh.CLARIFICATIONS).write_text('Changed after freeze')
    with pytest.raises(ValueError,match='source'):
        fresh.run(prepared_fresh,10,lambda *a:pytest.fail('No request'))
