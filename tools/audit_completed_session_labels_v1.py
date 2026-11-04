"""Revalidate completed report inventory; preserve annotations and prior receipts."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.validation.proxy import sha,write_json_new,ID
from tools.validate_report_evaluation import annotation_status
from tools import fresh_report_extraction as fresh

def main():
    out=ROOT/'outputs/completed_session_labels_audit_v1'
    if out.exists():raise FileExistsError(out)
    roots=[ROOT/'outputs'/s for s in ['chatgpt_manual_labels_v1','in_session_packets_v1','in_session_packets_remaining_v1']]
    inventory=[c['report_sha256'] for p in roots[-1].glob('batch_*/packet.json') for c in json.loads(p.read_text())['cases']]
    assert len(inventory)==len(set(inventory))==1978
    seen=[];batches=[];counts={};states=Counter();uids=set()
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text())
        for name,h in manifest['output_sha256'].items():assert sha(root/name)==h,name
        assert sha(ROOT/'tools/prepare_chatgpt_label_batches.py')==manifest['source_sha256']
        assert sha(ROOT/'docs/report_extraction_clarifications_v1.md')==manifest['clarifications_sha256']
        mapping={(r['batch'],r['case_id']):r for r in json.loads((root/'private_mapping.json').read_text())}
        count=0
        for f in sorted(root.glob('batch_*/annotations.session.json')):
            p=json.loads((f.parent/'packet.json').read_text());a=json.loads(f.read_text())
            status=annotation_status(p,a)
            assert status['status']=='complete_unverified_ai_annotations',str(f)
            assert all(r['certainty']=='definite' for r in a['rows'] if r['report_states'] in {'present','negated'})
            prov=json.loads((f.parent/'session_provenance.json').read_text())
            assert prov['annotations_sha256']==sha(f)
            assert prov['packet_file_sha256']==sha(f.parent/'packet.json')
            if 'decision_script' in prov:assert sha(prov['decision_script'])==prov['decision_script_sha256']
            v=json.loads((f.parent/'validation.session.json').read_text())
            assert v['status']=='validated_ai_annotations_not_ground_truth'
            assert v['response_file_sha256']==sha(f) and v['packet_file_sha256']==sha(f.parent/'packet.json')
            for c in p['cases']:
                seen.append(c['report_sha256']);count+=1
                mapped=mapping[(int(f.parent.name[-3:]),c['case_id'])]
                assert mapped['report_sha256']==c['report_sha256']
                for uid in mapped['training_uids']:
                    assert uid not in uids;uids.add(uid)
            states.update(r['report_states'] for r in a['rows'])
            batches.append({'path':str(f.parent.relative_to(ROOT)),'annotation_sha256':sha(f),'validation_sha256':sha(f.parent/'validation.session.json'),'reports':len(p['cases'])})
        counts[root.name]=count
    assert len(seen)==len(set(seen))==1978 and set(seen)==set(inventory)
    _,_,records=fresh.load(ROOT/'outputs/fresh_report_extraction_v1')
    key=pd.read_json(ROOT/'outputs/report_label_evaluation_v1/custodian/fresh_key.jsonl',lines=True).set_index('case_id')
    accepted={key.loc[r['case_id'],ID] for r in records.values() if r['status']=='accepted'}
    rejected_hashes={r['report_sha256'] for r in records.values() if r['status']!='accepted'}
    from src.labels.evaluation_sampling import digest
    roles=pd.read_csv(ROOT/'outputs/distinct_sampling_pair_v1/control/roles.csv')
    train=set(roles.loc[roles.role=='train',ID]);reports=pd.read_csv(ROOT/'data/meta/train.csv').set_index(ID)
    uncovered=sorted(train-uids-accepted)
    accepted_hashes={r['report_sha256'] for r in records.values() if r['status']=='accepted'}
    reasons=Counter('prior_rejected_report' if digest(reports.loc[u,'Report']) in rejected_hashes else 'duplicate_of_prior_accepted_report_not_propagated' if digest(reports.loc[u,'Report']) in accepted_hashes else 'unexplained' for u in uncovered)
    assert len(uncovered)==13 and reasons=={'prior_rejected_report':12,'duplicate_of_prior_accepted_report_not_propagated':1}
    production=sha(ROOT/'data/meta/train_labels.parquet')
    assert production=='e54e46531aec6cd16131d70aa283c990542dfa17daa890ac54da663cf74569b0'
    latest=ROOT/'outputs/in_session_packets_remaining_v1/checkpoint_through_396.json'
    cp=json.loads(latest.read_text());assert cp['remaining_unique_reports']==0 and cp['last_completed_case']=='B396R03'
    for row in cp['batches']:assert sha(roots[-1]/f"batch_{row['batch']:03d}/validation.session.json")==row['validation_sha256']
    result=dict(status='full_inventory_structurally_verified',reports=1978,finding_rows=sum(states.values()),batches=len(batches),counts_by_root=counts,duplicates=0,missing=0,states=dict(states),session_training_studies=len(uids),accepted_prior_studies=len(accepted),uncovered_training_studies=len(uncovered),uncovered_reason='12 studies have prior rejected extractions; one is a duplicate of a prior accepted report not propagated by the frozen builder; all reports exist and original targets remain',production_labels_sha256=production,checkpoint_sha256=sha(latest),audit_source_sha256=sha(__file__),semantic_accuracy_established=False,batch_receipts=batches)
    out.mkdir();write_json_new(out/'receipt.json',result)
    write_json_new(out/'uncovered_training_studies.json',{'study_ids':uncovered,'reason':result['uncovered_reason']})
    print(json.dumps({k:v for k,v in result.items() if k!='batch_receipts'},indent=2))
if __name__=='__main__':main()
