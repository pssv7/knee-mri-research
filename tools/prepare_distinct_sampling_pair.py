"""Freeze one matched sampler screen; no training or evaluation is launched."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.validation.proxy import sha,write_json_new
from src.data.sampling import POLICY


def main():
    old=ROOT/'outputs/validation_fallback_v2'
    for name,digest in json.loads((old/'manifest.json').read_text()).items():
        if sha(old/name)!=digest:raise ValueError(f'Frozen artifact changed: {name}')
    out=ROOT/'outputs/distinct_sampling_pair_v1';out.mkdir(exist_ok=False)
    shutil.copy2(old/'exclusions.csv',out/'exclusions.csv')
    roles=pd.read_csv(old/'outer0/roles.csv')
    index_path=ROOT/'data/meta/resolution_index/index_train.parquet'
    index=pd.read_parquet(index_path,columns=['StudyInstanceUID','SeriesInstanceUID'])
    counts=index[index.StudyInstanceUID.isin(roles.StudyInstanceUID)].groupby(['StudyInstanceUID','SeriesInstanceUID']).size().rename('raw_slices').reset_index()
    counts.to_parquet(out/'raw_counts.parquet',index=False)
    spec=json.loads((old/'outer0/spec.json').read_text())
    paths=set(spec['runtime_code_sha256']) | {'src/data/sampling.py','src/data/cache.py','src/data/dicom_io.py'}
    hashes={p:sha(ROOT/p) for p in sorted(paths)}
    for arm,policy in [('control','legacy'),('candidate',POLICY)]:
        dest=out/arm;dest.mkdir()
        for name in ['roles.csv','proxy_labels.parquet','class_counts.csv']:shutil.copy2(old/'outer0'/name,dest/name)
        new={**spec,'schema_version':2,'design':f'outer0_sampler_{arm}',
             'runtime_code_sha256':hashes,'sampling_policy':policy,
             'raw_counts_sha256':sha(out/'raw_counts.parquet'),
             'source_header_index_sha256':sha(index_path),
             'candidate_status':'technical_gate_passed_predictive_effect_unknown'}
        write_json_new(dest/'spec.json',new)
    write_json_new(out/'protocol.json',{
        'status':'prepared_not_trained','intervention':POLICY,'screen_fold':0,'seed':0,
        'training_jobs_launched':0,'image_gain_claim_allowed':False,
        'checkpoint_selection':'inner selection only; one checkpoint per arm',
        'primary_endpoint':'twelve-finding report-proxy macro ROC AUC',
        'exploratory_continuation_gate':{'macro_delta_min':.005,'per_finding_delta_min':-.02},
        'stop_rule':'Failed integrity/parity checks, failed or inconclusive bounded screen: stop, no blend or repeated tuning.',
        'confirmation':'Freeze after screen; unopened outer fold plus paired seeds for proxy robustness; new independent image labels for clinical claims.',
        'limitations':'Historical development used eligible data. No expert readers. CPU parity is not GPU training validation.',
        'inference_path':'build_model_series_volume for both arms: full24 normalization then mirror and sample',
        'source_design_manifest_sha256':sha(old/'manifest.json')})
    write_json_new(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(out)

if __name__=='__main__':main()
