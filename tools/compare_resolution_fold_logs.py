"""Compare 384 selected folds with saved 224 incumbent OOF and training logs."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src import config

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/resolution384_comparison'

def main():
 OUT.mkdir(exist_ok=True)
 labels=pd.read_parquet(ROOT/'data/meta/train_labels.parquet').set_index(config.ID_COL)
 gold=pd.read_parquet(ROOT/'data/meta/gold_truth_58.parquet').set_index(config.ID_COL)
 def read(path): return pd.read_parquet(path).set_index(config.ID_COL).sort_index()
 def auc(frame,truth=labels):
  y=truth.loc[frame.index,config.LABELS].to_numpy()>=.5
  return np.array([roc_auc_score(y[:,i],frame[n]) for i,n in enumerate(config.LABELS)])
 old=read(ROOT/'data/meta/compare_protocol_incumbent_all_folds/protocol_addition_oof.parquet')
 new=read(ROOT/'outputs/resolution384_downloads/oof_all_folds.parquet')
 assert old.index.equals(new.index) and old.fold.equals(new.fold)
 old_dirs={f:ROOT/f'outputs/kaggle_protocol_fold{f}_comparison_download/protocol_fold{f}' for f in (0,2,4)}
 old_dirs[1]=ROOT/'outputs/kaggle_reviewed_labels_fold1/v3_final/reviewed_control_fold1'
 old_dirs[3]=ROOT/'outputs/kaggle_protocol_fold3_report/protocol_fold3'
 rows=[]; logs=[]; protocols=[]; newgold=[]; oldgold=[]
 for f in range(5):
  nd=ROOT/f'outputs/resolution384_downloads/fold{f}/full/resolution384_fold{f}'
  od=old_dirs[f]
  expected=json.loads((ROOT/f'outputs/ensemble_16_24_protocol/protocol_fold{f}.pt.json').read_text())
  if f == 3:
   receipts=json.loads((ROOT/'data/meta/compare_protocol_incumbent_all_folds/summary.json').read_text())['source_receipts']
   receipt=next(r for r in receipts if r['family']=='protocol' and r['fold']==f)
   assert expected['sha256']==receipt['checkpoint_sha256']
   assert hashlib.sha256((od/f'oof_fold{f}.parquet').read_bytes()).hexdigest()==receipt['oof_sha256']
  else:
   actual=json.loads((od/f'convnext_fold{f}.pt.json').read_text())
   assert expected['sha256']==actual['sha256']
  proto=read(od/f'oof_fold{f}.parquet'); part=new[new.fold==f]
  assert proto.index.equals(part.index)
  proto['fold']=f;protocols.append(proto)
  rows.append({'fold':f,'studies':len(part),'previous_15model_core':auc(old[old.fold==f]).mean(),'previous_protocol224':auc(proto).mean(),'new384':auc(part).mean()})
  for family,directory,collection in [('224_protocol',od,oldgold),('384',nd,newgold)]:
   metrics=pd.read_csv(directory/f'lit_fold{f}/metrics/metrics.csv')
   v=metrics.dropna(subset=['val_macro']); best=v.loc[v.val_macro.idxmax()];last=v.iloc[-1]
   gp=read(directory/f'gold_predictions_fold{f}.parquet');collection.append(gp)
   logs.append({'fold':f,'family':family,'epochs':int(v.epoch.max())+1,'selected_epoch':int(best.epoch)+1,'best_val_auc':float(best.val_macro),'final_val_auc':float(last.val_macro),'drop_after_best':float(last.val_macro-best.val_macro),'selected_gold_auc':float(auc(gp,gold).mean()),'final_training_loss':float(metrics.train_loss_epoch.dropna().iloc[-1])})
 folds=pd.DataFrame(rows);folds['delta_vs_core']=folds.new384-folds.previous_15model_core;folds['delta_vs_protocol']=folds.new384-folds.previous_protocol224
 proto=pd.concat(protocols).sort_index()
 findings=pd.DataFrame({'finding':config.LABELS,'previous_15model_core':auc(old),'previous_protocol224':auc(proto),'new384':auc(new)})
 findings['delta_vs_core']=findings.new384-findings.previous_15model_core
 findings['delta_vs_protocol']=findings.new384-findings.previous_protocol224
 overlay=[]
 for f in (1,3):
  tr=read(ROOT/f'outputs/slice_overlay_staging/fold{f}/slice_transformer_fold{f}/oof_fold{f}.parquet')
  core=old[old.fold==f].copy();assert core.index.equals(tr.index)
  core[config.LABELS]=(core[config.LABELS]*3+tr[config.LABELS])/4
  overlay.append({'fold':f,'latest18_available_same_fold_proxy':auc(core).mean(),'new384':auc(new[new.fold==f]).mean()})
 def avg(frames):
  assert all(frames[0].index.equals(x.index) for x in frames)
  result=frames[0].copy();result[config.LABELS]=sum(x[config.LABELS] for x in frames)/len(frames);return result
 summary={'pooled_oof':{k:float(findings[k].mean()) for k in ['previous_15model_core','previous_protocol224','new384']},'gold58_five_fold_averages':{'previous_protocol224':float(auc(avg(oldgold),gold).mean()),'new384':float(auc(avg(newgold),gold).mean())},'latest18_available_folds':overlay,'limitations':['The 15-member core itself scored 0.914. Latest 18-member overlay also scored 0.914; transformer fold-2 OOF is absent locally, so a complete latest-overlay OOF comparison is unavailable.','OOF uses only same-fold members; averaging all trained folds on training studies would leak. This proxy does not reproduce full test-time ensembling.','Legacy training split membership was reconstructed rather than independently proved.','Gold58 is a small diagnostic set, not used to select models.','Resolution and previous protocol runs differ in hardware/runtime; not a resolution-only causal experiment.']}
 folds.to_csv(OUT/'fold_comparison.csv',index=False);findings.to_csv(OUT/'per_finding.csv',index=False);pd.DataFrame(logs).to_csv(OUT/'training_logs.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(folds.round(6).to_string(index=False));print(pd.DataFrame(logs).round(6).to_string(index=False));print(findings.round(6).to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
