"""Stage both frozen preflights and bounded cache-loader smoke, CPU only."""
import base64,io,json,zipfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.validation.proxy import sha
BODY=r'''
import base64,io,json,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
work=Path('/kaggle/working/distinct_smoke');work.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(work)
sys.path.insert(0,str(work))
from src.validation.proxy import sha,write_json_new
from tools.train_nested_proxy_control import preflight
from src.model.dataset import StudyIndex,load_study,mirror_volume
from src.data.sampling import sample_cached_volume,cached_indices
root=Path('/kaggle/input');started=time.monotonic()
receipt={'training_launched':False,'gpu_requested':False,'status':'started'}
try:
    def one(pattern):
        paths=list(root.rglob(pattern))
        if len(paths)!=1:raise ValueError(f'{pattern}: found {len(paths)}')
        return paths[0]
    previous=one('preflight_receipt.json').parent/'nested_proxy_preflight'
    pair=work/'outputs/distinct_sampling_pair_v1'
    prior_design=previous/'outputs/validation_fallback_v2/outer0'
    for arm in ['control','candidate']:
        for name in ['roles.csv','proxy_labels.parquet','class_counts.csv']:
            shutil.copy2(prior_design/name,pair/arm/name)
    shutil.copy2(previous/'outputs/validation_fallback_v2/exclusions.csv',pair/'exclusions.csv')
    inputs=json.loads((previous/'inputs.json').read_text())
    for entry in [inputs['series_meta'],inputs['side_predictions']]+[e for s in inputs['shards'] for e in s.values()]:
        entry['path']=str(one(Path(entry['path']).name))
    inp=work/'inputs.json';write_json_new(inp,inputs)
    # Run both exact production preflights, including full-array SHA-256 checks.
    data={}
    for arm in ['control','candidate']:
        print('Preflight',arm,flush=True)
        data[arm]=preflight(pair/arm,inp)
    spec,roles,labels,manifest,series,sides,paths,recipe=data['candidate']
    receipt['preflights']={'control':'passed','candidate':'passed'}
    index=StudyIndex(manifest,series,labels,sides,max_series=5,series_selection='protocol')
    selected=pd.concat([index.series_for(s) for s in index.studies],ignore_index=True)
    # Fixed seed and acquisition-only strata; no predictions or labels used.
    affected=selected[selected.n_slices_raw.between(17,23)]
    other=selected[~selected.n_slices_raw.between(17,23)]
    chosen=pd.concat([affected.sample(n=min(48,len(affected)),random_state=20260924),
                      other.sample(n=min(48,len(other)),random_state=20260924)])
    # Ensure each affected raw count is checked, plus left-sagittal cases.
    extra=affected.groupby('n_slices_raw',group_keys=False).head(1)
    left=affected[(affected.side_predicted=='L') & (affected.Anatomical_Plane=='Sagittal')].head(8)
    chosen=pd.concat([chosen,extra,left]).drop_duplicates('SeriesInstanceUID')
    shards={p:np.load(p,mmap_mode='r') for p in paths}
    records=[]
    for r in chosen.to_dict('records'):
        raw=shards[r['cache_path']][int(r['position'])]
        n=int(r['n_slices_raw']);reverse=r['side_predicted']=='L' and r['Anatomical_Plane']=='Sagittal'
        v=mirror_volume(raw,r['side_predicted'],r['Anatomical_Plane'])
        old=sample_cached_volume(v,n);new=sample_cached_volume(v,n,duplicate_free=True,depth_reversed=reverse)
        changed=not np.array_equal(old,new)
        unique=len({x.tobytes() for x in new})
        if 17<=n<=23:
            assert unique==16
            assert np.array_equal(new[0],v[0]) and np.array_equal(new[-1],v[-1])
            unmirrored=sample_cached_volume(raw,n,duplicate_free=True)
            expected=mirror_volume(unmirrored,r['side_predicted'],r['Anatomical_Plane'])
            assert np.array_equal(new,expected)
        else:assert not changed
        assert new.shape==old.shape and new.dtype==old.dtype
        records.append({'SeriesInstanceUID':r['SeriesInstanceUID'],'n_slices_raw':n,'side':r['side_predicted'],
          'plane':r['Anatomical_Plane'],'affected':17<=n<=23,'changed':changed,'unique_candidate':unique})
    # Exercise the actual study assembly on a bounded subset of these studies.
    study_ids=sorted(chosen.StudyInstanceUID.unique())[:16]
    checked=0
    for uid in study_ids:
        output=load_study(index,uid,shards,duplicate_free=True)
        for j,(_,r) in enumerate(index.series_for(uid).iterrows()):
            raw=shards[r.cache_path][int(r.position)]
            v=mirror_volume(raw,r.side_predicted,r.Anatomical_Plane)
            expected=sample_cached_volume(v,int(r.n_slices_raw),duplicate_free=True,
                depth_reversed=r.side_predicted=='L' and r.Anatomical_Plane=='Sagittal')
            assert np.array_equal(output['volumes'][j],expected);checked+=1
    pd.DataFrame(records).to_csv('/kaggle/working/smoke_series.csv',index=False)
    receipt.update(status='passed',eligible_studies=len(index.studies),eligible_series=len(manifest),
        sampled_series=len(records),affected_sampled=sum(r['affected'] for r in records),
        unaffected_sampled=sum(not r['affected'] for r in records),
        left_sagittal_affected_sampled=sum(r['affected'] and r['side']=='L' and r['plane']=='Sagittal' for r in records),
        study_loader_studies=len(study_ids),study_loader_series=checked,
        input_config_sha256=sha(inp),source_manifest_sha256=sha(work/'package_files.json'),
        raw_dicom_decode_tested=False,model_forward_tested=False,
        limitations='Acquisition-enriched technical sample, not accuracy evidence; no raw-DICOM replay, model fitting or scoring.')
except Exception as exc:
    receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json_new('/kaggle/working/smoke_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
'''

def main():
    out=ROOT/'outputs/kaggle_distinct_sampling_smoke_v1';out.mkdir(exist_ok=False)
    pair=ROOT/'outputs/distinct_sampling_pair_v1'
    spec=json.loads((pair/'control/spec.json').read_text())
    paths=set(spec['runtime_code_sha256'])|{'src/__init__.py','src/data/__init__.py','src/labels/__init__.py','src/validation/__init__.py',
        'outputs/distinct_sampling_pair_v1/raw_counts.parquet','outputs/distinct_sampling_pair_v1/control/spec.json','outputs/distinct_sampling_pair_v1/candidate/spec.json'}
    for name,h in spec['runtime_code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Frozen source changed: {name}')
    files={n:sha(ROOT/n) for n in sorted(paths)}
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
        for n in sorted(paths):z.write(ROOT/n,n)
        z.writestr('package_files.json',json.dumps(files,indent=2))
    code='PAYLOAD = '+repr(base64.b64encode(payload.getvalue()).decode())+'\n'+BODY
    compile(code,'smoke.py','exec');(out/'smoke.py').write_text(code)
    metadata=json.loads((ROOT/'outputs/kaggle_nested_proxy_preflight_v1/kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-distinct-sampling-smoke-v1',title='Knee Distinct Sampling Smoke V1',code_file='smoke.py')
    metadata['kernel_sources'].append('pssv27/knee-nested-proxy-preflight-v1')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'package_manifest.json').write_text(json.dumps({'files':files,'script_sha256':sha(out/'smoke.py')},indent=2))
    print(out, len(code),'bytes')
if __name__=='__main__':main()
