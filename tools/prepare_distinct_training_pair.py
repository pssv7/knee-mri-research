"""Stage exactly one frozen matched pair, using one T4 per arm."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BODY=r'''
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','lightning==2.6.5','timm==1.0.26'],check=True)
import hashlib,json,os,shutil,time
from pathlib import Path
import torch,timm,lightning.pytorch as pl
out=Path('/kaggle/working');started=time.monotonic()
host=Path('/kaggle/input/notebooks/pssv27/knee-distinct-sampling-smoke-v1')
work=out/'pair_source';shutil.copytree(host/'distinct_smoke',work)
sys.path.insert(0,str(work))
from src.validation.proxy import sha,write_json_new
from tools.evaluate_nested_proxy import evaluate
receipt={'status':'starting','full_training_authorized':True,'arms':{},'additional_runs_allowed':False,
         'competition_submission':False,'ensemble_created':False,'image_gain_claim_allowed':False}
children={};logs={}
try:
    assert json.loads((host/'smoke_receipt.json').read_text())['status']=='passed'
    files=json.loads((work/'package_files.json').read_text())
    for name,h in files.items():assert sha(work/name)==h,name
    pair=work/'outputs/distinct_sampling_pair_v1'
    control=json.loads((pair/'control/spec.json').read_text())
    candidate=json.loads((pair/'candidate/spec.json').read_text())
    assert {k for k in control if control[k]!=candidate[k]}=={'design','sampling_policy'}
    for arm in ['control','candidate']:
        spec=json.loads((pair/arm/'spec.json').read_text())
        assert sha(pair/arm/'roles.csv')==spec['roles_sha256']
        assert sha(pair/arm/'proxy_labels.parquet')==spec['proxy_labels_sha256']
        for name,h in spec['runtime_code_sha256'].items():assert sha(work/name)==h,name
    assert torch.__version__=='2.10.0+cu128',f'GPU probe runtime changed: {torch.__version__}'
    assert timm.__version__=='1.0.26' and pl.__version__=='2.6.5'
    if torch.cuda.device_count()!=2:raise ValueError('Expected two T4s, one per matched arm')
    if any(torch.cuda.get_device_name(i)!='Tesla T4' for i in range(2)):raise ValueError('GPU differs from verified probe')
    # Original pretraining cache warm-up, not a knee checkpoint or trained arm.
    from src.model import net
    initial=net.build_backbone(control['backbone'],pretrained=True)
    del initial
    input_path=work/'inputs.json'
    inputs=json.loads(input_path.read_text())
    for entry in [inputs['series_meta'],inputs['side_predictions']]+[e for s in inputs['shards'] for e in s.values()]:
        p=Path(entry['path'])
        if not p.is_file() or p.stat().st_size!=entry['bytes']:raise ValueError(f'Verified input missing/changed: {p}')
    # Each unmodified runner repeats full SHA-256 and role/count provenance checks.
    for device,arm in enumerate(['control','candidate']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(device);env['PYTHONHASHSEED']='0'
        logs[arm]=(out/f'{arm}_execution.log').open('w')
        command=[sys.executable,str(work/'tools/train_nested_proxy_control.py'),
                 '--design',str(pair/arm),'--inputs',str(input_path),'--out',str(out/arm),'--execute']
        children[arm]=subprocess.Popen(command,cwd=work,env=env,stdout=logs[arm],stderr=subprocess.STDOUT)
        receipt['arms'][arm]={'physical_device':device,'pid':children[arm].pid,'epochs':12,'seed':0,'model_count':1}
    write_json_new(out/'launch_receipt.json',{**receipt,'status':'running','input_config_sha256':sha(input_path),
        'matched_pretraining_source':'original timm convnext_tiny.fb_in22k_ft_in1k','source_manifest_sha256':sha(work/'package_files.json')})
    while True:
        codes={a:p.poll() for a,p in children.items()}
        print('Pair process status',json.dumps(codes),flush=True)
        failed={a:c for a,c in codes.items() if c is not None and c!=0}
        if failed:
            for p in children.values():
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'Arm failed; stopped pair without retries: {failed}')
        if all(c==0 for c in codes.values()):break
        time.sleep(60)
    for f in logs.values():f.close()
    report=evaluate(pair/'control',out/'control',out/'candidate',out/'outer_proxy_evaluation.json',draws=2000)
    delta=report['macro_delta'];per=report['per_finding'];ci=report['paired_sampling_95ci']
    point_gate=delta is not None and delta>=.005 and all(v['candidate'] is not None and v['control'] is not None and v['candidate']-v['control']>=-.02 for v in per.values())
    decision={'point_screen_passed':point_gate,'paired_sampling_lower_bound_positive':ci is not None and ci[0]>0,
        'eligible_for_separate_confirmation':point_gate and ci is not None and ci[0]>0,
        'image_gain_demonstrated':False,'automatic_followup_training':False,'ensemble_rescue':False,
        'rule':'Frozen +0.005 macro / -0.02 per-finding point screen; inconclusive sampling evidence does not advance. New image labels required for image-performance claims.'}
    write_json_new(out/'screen_decision.json',decision)
    receipt.update(status='completed',control_macro=report['control_macro'],candidate_macro=report['candidate_macro'],
        macro_delta=delta,paired_sampling_95ci=ci,screen_decision=decision)
except Exception as exc:
    receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    for p in children.values():
        if p.poll() is None:p.terminate()
    raise
finally:
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json_new(out/'pair_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
'''
def main():
    out=ROOT/'outputs/kaggle_distinct_training_pair_v1';out.mkdir(exist_ok=False)
    compile(BODY,'train_pair.py','exec');(out/'train_pair.py').write_text(BODY)
    metadata=json.loads((ROOT/'outputs/kaggle_distinct_gpu_probe_v3/kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-distinct-sampling-pair-v1',title='Knee Distinct Sampling Pair V1',code_file='train_pair.py')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out)
if __name__=='__main__':main()
