"""Assemble validated session-authored labels into a separate training candidate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.labels.openai_extract import FIELDS
from src.labels.evaluation_sampling import digest
from src.labels.extraction_repair import repair
from src.validation.proxy import ID,sha,write_json_new
from tools import fresh_report_extraction as fresh
from tools.label_repair_experiment import verify_plan
from tools.label_repair_pair import validate_changes
from tools.validate_report_evaluation import annotation_status


def build(session_roots,out):
    out=Path(out)
    if out.exists(): raise FileExistsError('Preserve previous candidate versions')
    experiment=ROOT/'outputs/extraction_repair_experiment_v1'
    plan=verify_plan(experiment)
    fm,_,records=fresh.load(ROOT/plan['extraction'])
    if len(records)!=len(fm['cases']):raise ValueError('Original extraction is incomplete')
    key=pd.read_json(ROOT/'outputs/report_label_evaluation_v1/custodian/fresh_key.jsonl',lines=True).set_index('case_id')
    assertions={key.loc[r['case_id'],ID]:r['assertions'] for r in records.values() if r['status']=='accepted'}
    base=ROOT/plan['baseline']
    roles=pd.read_csv(base/'control/roles.csv'); control=pd.read_parquet(base/'control/proxy_labels.parquet')
    reports=pd.read_csv(ROOT/'data/meta/train.csv').set_index(ID)
    train=set(roles.loc[roles.role=='train',ID])
    session_ids=[];source_hashes={};session_reports=0;seen_report_hashes=set()
    for directory in session_roots:
        directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text())
        source_hashes[str(directory/'manifest.json')]=sha(directory/'manifest.json')
        for name,expected in manifest['output_sha256'].items():
            if sha(directory/name)!=expected:raise ValueError('Frozen packet changed: '+name)
        mapping={(r['batch'],r['case_id']):r for r in json.loads((directory/'private_mapping.json').read_text())}
        files=sorted(directory.glob('batch_*/annotations.session.json'))
        if not files:raise ValueError('No session annotation files in '+str(directory))
        for file in files:
            folder=file.parent;number=int(folder.name.split('_')[-1])
            packet=json.loads((folder/'packet.json').read_text());ann=json.loads(file.read_text())
            state=annotation_status(packet,ann)
            if state['status']!='complete_unverified_ai_annotations':raise ValueError('Invalid/incomplete batch: '+str(file))
            provenance=json.loads((folder/'session_provenance.json').read_text())
            if provenance['annotations_sha256']!=sha(file):raise ValueError('Session annotation provenance changed')
            for name in ['packet.json','annotations.session.json','session_provenance.json']:
                source_hashes[str(folder/name)]=sha(folder/name)
            for case in packet['cases']:
                row=mapping[(number,case['case_id'])]
                if case['report_sha256']!=row['report_sha256'] or case['report_sha256'] in seen_report_hashes:
                    raise ValueError('Duplicate session report or changed mapping')
                seen_report_hashes.add(case['report_sha256'])
                values={r['finding']:{f:r[f] for f in FIELDS} for r in ann['rows'] if r['case_id']==case['case_id']}
                if any(a['report_states'] in {'present','negated'} and a['certainty']!='definite' for a in values.values()):
                    raise ValueError('Definite-state certainty conflict')
                for uid in row['training_uids']:
                    if uid not in train or uid in assertions:raise ValueError('Nontraining or duplicate study')
                    if digest(reports.loc[uid,'Report'])!=case['report_sha256']:raise ValueError('Original report changed')
                    assertions[uid]=values;session_ids.append(uid)
                session_reports+=1
    cat=pd.read_parquet(ROOT/'data/meta/llm_labels.parquet');prob=pd.read_parquet(ROOT/'data/meta/llm_probs.parquet')
    candidate,changes,keeps=repair(control,roles,cat,prob,reports,assertions)
    checks=validate_changes(control,candidate,roles,changes)
    out.mkdir(parents=True)
    candidate.to_parquet(out/'candidate_proxy_labels.parquet',index=False)
    for name,rows in [('changes.jsonl',changes),('kept.jsonl',keeps)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    session_set=set(session_ids)
    remaining_unique=1978-session_reports
    result={'status':'partial_training_relabel_candidate','session_reports':session_reports,
            'session_finding_rows':session_reports*12,'session_training_studies':len(session_ids),
            'previous_sonnet_accepted_reports':sum(r['status']=='accepted' for r in records.values()),
            'total_annotated_training_studies':len(assertions),'training_studies_total':len(train),
            'remaining_unique_reports':remaining_unique,
            'changed_targets_total':len(changes),'changed_studies_total':checks['changed_studies'],
            'session_changed_targets':sum(r[ID] in session_set for r in changes),
            'session_changed_studies':len({r[ID] for r in changes if r[ID] in session_set}),
            'checks':checks,'candidate_sha256':sha(out/'candidate_proxy_labels.parquet'),
            'changes_sha256':sha(out/'changes.jsonl'),'experiment_plan_sha256':sha(experiment/'plan.json'),
            'source_hashes':source_hashes,'builder_sha256':sha(__file__),
            'new_api_calls':0,'new_browser_requests':0,'production_labels_changed':False,
            'training_launched':False,'independent_accuracy_established':False}
    write_json_new(out/'receipt.json',result)
    return {k:v for k,v in result.items() if k!='source_hashes'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-root',dest='session_roots',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
