"""Prepare a bounded GPU execution check, not an experiment campaign."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BODY=r'''
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','lightning==2.6.5','timm==1.0.26'],check=True)
import gc,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
import lightning.pytorch as pl
from torch.utils.data import DataLoader
root=Path('/kaggle/input');out=Path('/kaggle/working')
host=root/'notebooks/pssv27/knee-distinct-sampling-smoke-v1'
source=host/'distinct_smoke'
if not (source/'package_files.json').is_file():
    raise ValueError('Verified smoke source mount is unavailable')
print('Verified source mount located directly',flush=True)
work=out/'probe_source';shutil.copytree(source,work)
sys.path.insert(0,str(work))
from src.validation.proxy import sha,write_json_new
from src.model import dataset,lightning as knee,net
receipt={'status':'started','full_training_launched':False,'optimizer_steps_per_arm':2,
         'microbatches_per_arm':16,'validation_or_outer_scored':False,'arms':[]}
started=time.monotonic()
try:
    previous=json.loads((host/'smoke_receipt.json').read_text())
    assert previous['status']=='passed'
    files=json.loads((work/'package_files.json').read_text())
    for name,digest in files.items():assert sha(work/name)==digest,name
    pair=work/'outputs/distinct_sampling_pair_v1'
    spec=json.loads((pair/'control/spec.json').read_text())
    for name,digest in spec['runtime_code_sha256'].items():assert sha(work/name)==digest,name
    inputs=json.loads((work/'inputs.json').read_text())
    entries=[inputs['series_meta'],inputs['side_predictions']]+[e for s in inputs['shards'] for e in s.values()]
    for entry in entries:
        path=Path(entry['path'])
        print('Checking metadata/size',path.name,flush=True)
        if not path.is_file():raise ValueError(f'Verified input path unavailable: {path}')
        assert path.stat().st_size==entry['bytes']
        if path.suffix!='.npy':assert sha(path)==entry['sha256']
    manifest=pd.concat([pd.read_parquet(s['manifest']['path']).assign(cache_path=s['array']['path']) for s in inputs['shards']],ignore_index=True)
    roles=pd.read_csv(pair/'control/roles.csv');labels=pd.read_parquet(pair/'control/proxy_labels.parquet')
    meta=pd.read_csv(inputs['series_meta']['path']);sides=pd.read_parquet(inputs['side_predictions']['path'])
    manifest=manifest[manifest.StudyInstanceUID.isin(roles.StudyInstanceUID)]
    index=dataset.StudyIndex(manifest,meta,labels,sides,max_series=5,series_selection='protocol')
    train=sorted(roles.loc[roles.role=='train','StudyInstanceUID'])
    eligible=[]
    for uid in train:
        rows=index.series_for(uid)
        if len(rows)==5:eligible.append((uid,bool(rows.n_slices_raw.between(17,23).any())))
    chosen=[u for u,a in eligible if a][:8]+[u for u,a in eligible if not a][:8]
    assert len(chosen)==16 and set(chosen)<=set(train)
    write_json_new(out/'probe_sample.json',{'train_studies':chosen,'selection':'first eight five-series affected and eight five-series unaffected train studies, UID order; technical enrichment'})
    assert torch.cuda.is_available()
    receipt.update(gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),used_devices=1,
        torch_version=torch.__version__,timm_version=timm.__version__,lightning_version=pl.__version__,
        pretrained_initialization=True,precision='16-mixed',gradient_checkpointing=False)
    paths=[s['array']['path'] for s in inputs['shards']]
    initial_hash=None
    class Timing(pl.Callback):
        def __init__(self):self.times=[];self.losses=[];self.t=None
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            torch.cuda.synchronize();self.t=time.monotonic()
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            torch.cuda.synchronize();self.times.append(time.monotonic()-self.t)
            loss=outputs['loss'].detach().float()
            if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
            self.losses.append(float(loss))
        def on_before_optimizer_step(self,trainer,module,optimizer):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite or absent gradients')
    for arm,enabled in [('control',False),('candidate',True)]:
        print('GPU probe arm',arm,flush=True)
        pl.seed_everything(0,workers=True)
        loader=DataLoader(knee.StudyVolumeDataset(index,chosen,paths,16,augment=True,resample=False,duplicate_free=enabled),
            batch_size=1,shuffle=False,num_workers=2,persistent_workers=True,pin_memory=True,collate_fn=dataset.collate_studies)
        backbone=net.build_backbone(spec['backbone'],pretrained=True)
        total=knee.optimiser_steps(len(train),1,1,8,12)
        module=knee.KneeLitModule(backbone,backbone.num_features,total_steps=total,lr=.0001,
            warmup_steps=int(round(total/12*3.6)),**net.backbone_defaults(spec['backbone']))
        digest=hashlib.sha256()
        for name,t in sorted(module.state_dict().items()):digest.update(name.encode());digest.update(t.detach().cpu().contiguous().numpy().tobytes())
        h=digest.hexdigest()
        if initial_hash is None:initial_hash=h
        assert h==initial_hash
        callback=Timing()
        trainer=pl.Trainer(max_steps=2,max_epochs=1,limit_train_batches=16,limit_val_batches=0,num_sanity_val_steps=0,
            accelerator='gpu',devices=1,precision='16-mixed',accumulate_grad_batches=8,callbacks=[callback],
            logger=False,enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        trainer.fit(module,train_dataloaders=loader)
        torch.cuda.synchronize();wall=time.monotonic()-start
        assert trainer.global_step==2 and len(callback.times)==16
        if not all(torch.isfinite(p).all() for p in module.parameters()):raise ValueError('Nonfinite updated parameters')
        steady=float(np.mean(callback.times[1:]))
        receipt['arms'].append({'arm':arm,'initial_weights_sha256':h,'optimizer_steps':trainer.global_step,
           'microbatches':len(callback.times),'finite_losses_gradients_parameters':True,
           'batch_seconds':callback.times,'mean_batch_seconds_excluding_first':steady,
           'fit_wall_seconds':wall,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
           'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
           'twelve_epoch_compute_hours_at_five_series':steady*len(train)*12/3600,
           'twelve_epoch_scaled_fit_wall_hours':wall/16*len(train)*12/3600})
        write_json_new(out/f'{arm}_probe.json',receipt['arms'][-1])
        del trainer,module,backbone,loader,callback;gc.collect();torch.cuda.empty_cache()
    receipt.update(status='passed',matched_initialization=True,
        limitations='Two transient optimizer steps per arm on enriched training-only maximum-series sample. No checkpoint retained, AUC, validation or outer predictions. Timing extrapolations exclude full validation/checkpointing and are not a duration guarantee. Arrays size-checked against prior full SHA-256 receipt; small metadata rehashed.')
except Exception as exc:
    receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
finally:
    receipt['elapsed_seconds']=time.monotonic()-started
    write_json_new(out/'gpu_probe_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
'''
def main():
    out=ROOT/'outputs/kaggle_distinct_gpu_probe_v3';out.mkdir(exist_ok=False)
    compile(BODY,'probe.py','exec');(out/'probe.py').write_text(BODY)
    metadata=json.loads((ROOT/'outputs/kaggle_distinct_sampling_smoke_v1/kernel-metadata.json').read_text())
    metadata.update(id='pssv27/knee-distinct-gpu-probe-v1',title='Knee Distinct GPU Probe V1',code_file='probe.py',
                    enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4')
    metadata['kernel_sources']=[f'pssv27/knee-cache-224-01-shard-{i}' for i in (1,2,3)]+['pssv27/knee-side-classifier','pssv27/knee-distinct-sampling-smoke-v1']
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2));print(out)
if __name__=='__main__':main()
