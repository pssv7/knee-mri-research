"""Read existing artifacts; write a separate validation audit and blank review packet.

No training, relabeling, network access, or changes to historical artifacts.
Run from the repository root with .venv/bin/python tools/audit_validation_reset.py.
The pilot is DEVELOPMENT only. This does not create a pristine locked test set.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.labels import calibrate, fuse
from src.model.splits import make_folds
from tools.audit_labels import detect_language

OUT = Path('outputs/validation_reset_20260924')
ID = config.ID_COL
L = config.LABELS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def aucs(y, p):
    return [float(calibrate._auc(y[:, i], p[:, i])) for i in range(len(L))]


def hm_halfwidth(a, pos, neg):
    q1, q2 = a / (2-a), 2*a*a / (1+a)
    return 1.96 * math.sqrt((a*(1-a)+(pos-1)*(q1-a*a)+(neg-1)*(q2-a*a))/(pos*neg))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv('data/meta/train.csv')
    labels = pd.read_parquet('data/meta/train_labels.parquet')
    gold = train[train[L[0]].notna()].copy()
    folds = make_folds(labels)[[ID, 'fold']]
    series = pd.read_csv('data/meta/train_series.csv')
    test = pd.read_csv('data/meta/test.csv')
    test_series = pd.read_csv('data/meta/test_series.csv')
    index_path = 'data/meta/resolution_index/index_train.parquet'
    ix = pd.read_parquet(index_path)
    study = train[[ID]].merge(folds, how='left', on=ID)
    study['fold'] = study['fold'].fillna(-1).astype(int)
    study['language_heuristic'] = train.Report.map(detect_language)
    fields = ['Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength']
    keys = ix[fields].fillna('unknown').astype(str).agg('|'.join, axis=1)
    ix['scanner_proxy'] = keys
    scanner = ix.groupby(ID).scanner_proxy.first()
    study['scanner_proxy'] = study[ID].map(scanner)
    study['vendor'] = study[ID].map(ix.groupby(ID).Manufacturer.first())
    study['report_hash'] = train.Report.fillna('').map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    study['normalized_report_hash'] = train.Report.fillna('').map(
        lambda s: hashlib.sha256(' '.join(unicodedata.normalize('NFKC', s).casefold().split()).encode()).hexdigest())
    study['series_count'] = study[ID].map(series.groupby(ID).size())
    summary = {'scope': 'Study IDs and available metadata only; no patient or pixel identity proof.',
               'counts': {'train': len(train), 'mined': len(labels), 'gold': len(gold),
                          'series': len(series), 'instances': len(ix)},
               'missing_linkage_columns': [c for c in ['PatientID','StudyDate','AccessionNumber','InstitutionName',
                  'StationName','DeviceSerialNumber','PatientBirthDate'] if c not in ix],
               'gold_label_missing': int(gold[L].isna().sum().sum()),
               'partial_labels_outside_gold': int(train.loc[train[L[0]].isna(),L].notna().sum().sum()),
               'gold_in_mined': len(set(gold[ID]) & set(labels[ID])),
               'duplicate_train_uids': int(train[ID].duplicated().sum()),
               'duplicate_series_uids': int(series.SeriesInstanceUID.duplicated().sum()),
               'duplicate_sop_uids': int(ix.SOPInstanceUID.duplicated().sum()),
               'sop_uids_across_studies': int((ix.groupby('SOPInstanceUID')[ID].nunique()>1).sum()),
               'series_uids_across_studies': int((ix.groupby('SeriesInstanceUID')[ID].nunique()>1).sum()),
               'visible_test_study_overlap': len(set(train[ID]) & set(test[ID])),
               'visible_test_series_overlap': len(set(series.SeriesInstanceUID) & set(test_series.SeriesInstanceUID)),
               'studies_multiple_scanner_proxies': int((ix.groupby(ID).scanner_proxy.nunique()>1).sum()),
               'scanner_proxies': int(study.scanner_proxy.nunique())}
    for name in ['report_hash','normalized_report_hash']:
        grouped = study.groupby(name).filter(lambda d: len(d)>1)
        grouped.to_csv(OUT / f'{name}_duplicates.csv', index=False)
        summary[name] = {'groups': int(grouped[name].nunique()), 'studies': len(grouped),
                         'cross_fold_groups': int((grouped.groupby(name).fold.nunique()>1).sum()),
                         'gold_mined_groups': sum(bool((g.fold==-1).any() and (g.fold>=0).any())
                                                  for _,g in grouped.groupby(name))}
    # Near-identical text is a linkage lead, not evidence of duplicate patients.
    from difflib import SequenceMatcher
    texts = [' '.join(unicodedata.normalize('NFKC', s).casefold().split()) for s in train.Report.fillna('')]
    token_sets = [set(s.split()) for s in texts]
    near = []
    for i in range(len(texts)):
        if len(texts[i]) < 100: continue
        for j in range(i):
            if len(texts[j]) < 100 or texts[i] == texts[j]: continue
            if min(len(texts[i]),len(texts[j])) / max(len(texts[i]),len(texts[j])) < .97: continue
            a,b = token_sets[i],token_sets[j]
            if len(a&b)/max(1,len(a|b)) < .97: continue
            ratio = SequenceMatcher(None,texts[i],texts[j],autojunk=False).ratio()
            if ratio >= .98:
                near.append({'study_a':train.iloc[i][ID], 'study_b':train.iloc[j][ID], 'text_similarity':ratio,
                             'cross_fold':bool(study.iloc[i].fold != study.iloc[j].fold)})
    pd.DataFrame(near,columns=['study_a','study_b','text_similarity','cross_fold']).to_csv(OUT/'near_report_pairs.csv',index=False)
    summary['near_report_search'] = {'pairs':len(near),'cross_fold':sum(x['cross_fold'] for x in near),
        'method':'NFKC/casefold/whitespace, length ratio >= .97, token Jaccard >= .97, SequenceMatcher >= .98; incomplete heuristic'}
    for col in ['language_heuristic','scanner_proxy','vendor']:
        pd.crosstab(study[col],study.fold).to_csv(OUT/f'{col}_by_fold.csv')
    summary['scanner_overlap_by_fold'] = {}
    summary['gold_scanner_proxies'] = int(study.loc[study.fold==-1,'scanner_proxy'].nunique())
    for f in range(5):
        tr = set(study.loc[(study.fold>=0)&(study.fold!=f),'scanner_proxy'])
        va = study[study.fold==f]
        summary['scanner_overlap_by_fold'][f] = {'validation_studies':len(va),
            'seen_scanner_studies':int(va.scanner_proxy.isin(tr).sum())}
    study.to_csv(OUT/'study_metadata.csv',index=False)
    pd.crosstab(series.Anatomical_Plane,[series.Fluid_Sensitive,series.Fat_Suppression]).to_csv(OUT/'protocol_counts.csv')
    # Rebuild, but never write, the historical labels.
    rebuilt, fused_gold, _ = fuse.build()
    assert list(rebuilt.columns) == list(labels.columns)
    assert rebuilt[ID].equals(labels[ID])
    summary['label_reproduction'] = {'numeric_cells':int(labels.drop(columns=ID).size),
        'maximum_difference':float(np.abs(rebuilt.drop(columns=ID).to_numpy()-labels.drop(columns=ID).to_numpy()).max())}
    yy=gold[L].to_numpy(int); pp=fused_gold[L].to_numpy(float)
    positives=[]
    for i,l in enumerate(L):
        y=yy[:,i]; p=pp[:,i]>=.5
        positives.append({'finding':l,'gold_positive':int(y.sum()),'gold_negative':int((1-y).sum()),
                          'mined_positive':int((labels[l]>=.5).sum()),
                          'gold_fused_auc':calibrate._auc(y,pp[:,i]),
                          'gold_fused_false_negative':int(((y==1)&~p).sum()),
                          'gold_fused_false_positive':int(((y==0)&p).sum())})
    pd.DataFrame(positives).to_csv(OUT/'label_diagnostics.csv',index=False)
    summary['gold_copies_match_train']={}
    for p in ['data/meta/gold_truth_58.csv','data/meta/gold_truth_58.parquet']:
        frame=pd.read_csv(p) if p.endswith('.csv') else pd.read_parquet(p)
        summary['gold_copies_match_train'][p]=bool(np.array_equal(frame.set_index(ID).loc[gold[ID],L].to_numpy(), yy))
    # Verify individual protocol references and packaged selected weights, without loading a network.
    frames=[]; checks=[]
    for family in ['protocol224','resolution384']:
        family_frames=[]
        for f in range(5):
            if family=='protocol224':
                pred=Path(f'outputs/kaggle_protocol_ensemble_upload/reference_protocol_fold{f}.parquet')
                weight=Path(f'outputs/ensemble_16_24_protocol/protocol_fold{f}.pt')
            else:
                pred=Path(f'outputs/resolution384_downloads/fold{f}/full/resolution384_fold{f}/oof_fold{f}.parquet')
                weight=Path(f'outputs/ensemble_resolution384/convnext_fold{f}.pt')
            side=json.loads(Path(str(weight)+'.json').read_text())
            df=pd.read_parquet(pred);df['fold']=f
            tr,va=set(side['train_uids']),set(side['val_uids'])
            expected=set(folds.loc[folds.fold==f,ID])
            check={'family':family,'fold':f,'predictions':str(pred),'weight':str(weight),
                   'sha256':sha(weight),'hash_matches':sha(weight)==side['sha256'],
                   'train_val_overlap':len(tr&va),'gold_train_overlap':len(tr&set(gold[ID])),
                   'val_matches_expected':va==expected,'prediction_ids_match':set(df[ID])==va,
                   'train_matches_expected':tr==set(labels[ID])-expected}
            assert check['hash_matches'] and check['val_matches_expected'] and check['prediction_ids_match'] and check['train_matches_expected']
            assert not check['train_val_overlap'] and not check['gold_train_overlap']
            checks.append(check);family_frames.append(df)
        frame=pd.concat(family_frames).set_index(ID).loc[labels[ID]]
        p=frame[L].to_numpy(float); y=(labels[L].to_numpy()>=.5).astype(int)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        scores=aucs(y,p)
        summary[family]={'macro_pooled':float(np.mean(scores)), 'per_finding':dict(zip(L,scores)),
            'macro_within_fold_mean':float(np.mean([np.mean(aucs(y[frame.fold.to_numpy()==f],p[frame.fold.to_numpy()==f])) for f in range(5)]))}
        frame.reset_index().to_parquet(OUT/f'{family}_oof.parquet',index=False)
        frames.append(frame)
    (OUT/'checkpoint_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    from tools.build_label_review import verified_protocol_oof
    paths, _, _, _ = verified_protocol_oof()
    recorded = pd.concat([pd.read_parquet(p) for p in paths]).set_index(ID).loc[labels[ID],L]
    assert np.array_equal(recorded.to_numpy(), frames[0][L].to_numpy())
    summary['protocol_predictions_match_historical_receipts'] = True
    meta = study.set_index(ID)
    subgroup = []
    for dim in ['language_heuristic','vendor']:
        for value,g in meta.loc[labels[ID]].groupby(dim):
            uu=g.index
            for l in L:
                yy=(labels.set_index(ID).loc[uu,l].to_numpy()>=.5).astype(int)
                subgroup.append({'dimension':dim,'stratum':value,'finding':l,'n':len(uu),
                    'positive':int(yy.sum()),'negative':int((1-yy).sum()),
                    'protocol_proxy_auc':calibrate._auc(yy,frames[0].loc[uu,l].to_numpy())})
    pd.DataFrame(subgroup).to_csv(OUT/'subgroup_proxy_diagnostics.csv',index=False)
    # Existing protected text reserve remains unopened and excluded from this pilot.
    exposure=pd.read_parquet('outputs/label_quality/audit_v1/exposure.parquet')
    sample=pd.read_parquet('outputs/label_quality/audit_v1/sample.parquet')
    extra=pd.read_parquet('outputs/label_quality/audit_v1/extra_controls.parquet')
    reserve=pd.read_parquet('outputs/label_quality/audit_v1/reserve.parquet')
    excluded=set(gold[ID])|set(exposure[ID])|set(sample[ID])|set(extra[ID])|set(reserve[ID])
    # Exclude exact/near text-linked cases too, pending identity review.
    excluded |= set(study.loc[study.normalized_report_hash.duplicated(False),ID])
    for r in near: excluded|={r['study_a'],r['study_b']}
    pool=study.loc[~study[ID].isin(excluded)].copy()
    # Fixed pilot size derives from workflow/budget estimation, not an AUC claim.
    pilot=pool.sample(n=120,random_state=20260924).copy()
    pilot.insert(0,'case_id',[f'DEV{i:04d}' for i in range(1,len(pilot)+1)])
    pilot['sampling_role']='development_feasibility_SRS'
    pilot['inclusion_probability']=120/len(pool)
    pilot.to_csv(OUT/'pilot_custodian_key.csv',index=False)
    pilot[['case_id',ID]].merge(series,on=ID).to_csv(OUT/'pilot_series_custodian_manifest.csv',index=False)
    rows=[]
    for cid in pilot.case_id:
        for finding in L:
            rows.append({'case_id':cid,'finding':finding,'label':'','assessable':'','certainty':'',
                         'severity':'','anatomical_location':'','series_reference':'','image_reference':'',
                         'reason_uncertain':'','reader_id':'','review_started_utc':'','review_finished_utc':'',
                         'rubric_version':'','viewer_series_complete':'','notes':''})
    for seed,reader in enumerate(['reader_a','reader_b']):
        order=pilot.case_id.sample(frac=1,random_state=20260924+seed).tolist()
        form=pd.DataFrame(rows).set_index('case_id').loc[order].reset_index()
        form.to_csv(OUT/f'pilot_{reader}_blank.csv',index=False)
    pd.DataFrame(columns=['case_id','finding','reader_a_label','reader_b_label','adjudicated_label',
                         'adjudicator_id','rationale','reference_version','timestamp_utc']).to_csv(OUT/'adjudication_blank.csv',index=False)
    summary['pilot']={'role':'development only; already exposed to historical training/selection',
        'eligible_studies':len(pool),'sample':120,'seed':20260924,'excluded_studies':len(excluded),
        'protected_existing_report_reserve_studies':int(reserve[ID].nunique()),
        'images_packaged':False,'locked_final_set_created':False}
    assert not (set(pilot[ID])&excluded)
    assert len(set(pilot[ID]))==120
    planning=[]
    for n in [58,120,500,1000,2000]:
        for prevalence in [.05,.10,.20,.50]:
            pos=max(1,round(n*prevalence));neg=n-pos
            planning.append({'n':n,'prevalence_assumption':prevalence,'positive':pos,'negative':neg,
                             'auc_assumption':.85,'approx_95ci_halfwidth':hm_halfwidth(.85,pos,neg)})
    pd.DataFrame(planning).to_csv(OUT/'precision_scenarios.csv',index=False)
    inputs=['data/meta/train.csv','data/meta/train_labels.parquet','data/meta/train_series.csv',index_path,
            'docs/label_quality/manifest_v1.json','docs/label_quality/baseline_v1.json']
    summary['input_sha256']={p:sha(p) for p in inputs}
    summary['script_sha256']=sha(__file__)
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    products={str(p):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(products,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
