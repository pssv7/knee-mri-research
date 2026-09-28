"""Bounded technical audit, restricted to unprotected fallback studies."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import config
from src.model.dataset import select_series, PLANE_MIRROR_AXIS
from src.data.cache import submission_indices, mirror_axis
from src.validation.proxy import sha, strict_aucs
ID=config.ID_COL
OUT=ROOT/'outputs/baseline_input_audit_v1'

def main():
    OUT.mkdir(exist_ok=False)
    roles=pd.read_csv(ROOT/'outputs/validation_fallback_v2/outer0/roles.csv')
    allowed=set(roles[ID])
    series=pd.read_csv(ROOT/'data/meta/train_series.csv')
    series=series[series[ID].isin(allowed)].sort_values('SeriesInstanceUID').copy()
    series['selected']=False
    study=[]
    cols=['Anatomical_Plane','Fluid_Sensitive','Fat_Suppression']
    for uid,g in series.groupby(ID):
        chosen=g.iloc[select_series(g,5,'protocol')]
        series.loc[chosen.index,'selected']=True
        study.append({ID:uid,'series_count':len(g),'omitted_series':len(g)-len(chosen),
          'lost_planes':len(set(g.Anatomical_Plane)-set(chosen.Anatomical_Plane)),
          'lost_protocols':len(g[cols].drop_duplicates())-len(chosen[cols].drop_duplicates())})
    ix=pd.read_parquet(ROOT/'data/meta/resolution_index/index_train.parquet')
    ix=ix[ix[ID].isin(allowed)]
    rows=[]
    for uid,g in ix.groupby('SeriesInstanceUID',sort=True):
        o=np.stack(g.ImageOrientationPatient).astype(float)
        p=np.stack(g.ImagePositionPatient).astype(float)
        normal=np.cross(o[0,:3],o[0,3:]);normal/=np.linalg.norm(normal)
        positions=np.sort(p@normal)
        selected=np.array(submission_indices(len(g),16))
        spacing=np.array(g.PixelSpacing.iloc[0],float)
        height=float(g.Rows.iloc[0])*spacing[0];width=float(g.Columns.iloc[0])*spacing[1]
        axes=[mirror_axis(v) for v in o]
        sides=set(str(v).strip().upper() for c in ['ImageLaterality','Laterality'] for v in g[c] if pd.notna(v)) & {'L','R'}
        rows.append({'SeriesInstanceUID':uid,'raw_slices':len(g),'unique_input_slices':len(set(selected)),
            'raw_coverage_fraction':len(set(selected))/len(g),'max_input_gap_mm':float(np.diff(positions[selected]).max()),
            'geometry_mirror_axis':axes[0],'axis_changes':len(set(axes))>1,
            'orientation_variation':float(np.max(np.abs(o-o[0]))),
            'physical_aspect_ratio':width/height,'matrix_aspect_ratio':float(g.Columns.iloc[0])/float(g.Rows.iloc[0]),
            'width_lr_sign':int(np.sign(o[0,0])),'normal_lr_sign':int(np.sign(normal[0])),
            'tag_side':next(iter(sides)) if len(sides)==1 else '', 'tag_conflict':len(sides)>1})
    series=series.merge(pd.DataFrame(rows),on='SeriesInstanceUID',validate='one_to_one')
    series['loader_mirror_axis']=series.Anatomical_Plane.map(PLANE_MIRROR_AXIS)
    series['mirror_axis_mismatch']=series.loader_mirror_axis!=series.geometry_mirror_axis
    series['aspect_distortion_over10pct']=np.abs(np.log(series.physical_aspect_ratio))>np.log(1.1)
    series.to_parquet(OUT/'series_technical.parquet',index=False)
    study=pd.DataFrame(study)
    chosen=series[series.selected]
    for col in ['mirror_axis_mismatch','aspect_distortion_over10pct','tag_conflict']:
        study[col]=study[ID].map(chosen.groupby(ID)[col].any())
    study['repeated_input_slices']=study[ID].map(chosen.assign(repeats=chosen.unique_input_slices<16).groupby(ID).repeats.any())
    meta=pd.read_csv(ROOT/'outputs/validation_reset_20260924/study_metadata.csv')
    study=study.merge(meta[[ID,'scanner_proxy','language_heuristic']],on=ID,validate='one_to_one')
    pred=pd.read_parquet(ROOT/'outputs/validation_reset_20260924/protocol224_oof.parquet').set_index(ID).loc[study[ID],config.LABELS]
    y=pd.read_parquet(ROOT/'data/meta/train_labels.parquet').set_index(ID).loc[study[ID],config.LABELS].to_numpy()>=.5
    study['mean_proxy_brier']=np.mean((pred.to_numpy()-y)**2,axis=1)
    study.to_csv(OUT/'study_diagnostics.csv',index=False)
    groups=[]
    for key in ['scanner_proxy','language_heuristic','mirror_axis_mismatch','aspect_distortion_over10pct','repeated_input_slices','lost_protocols']:
        for value,g in study.groupby(key):
            indices=g.index.to_numpy(); per,macro=strict_aucs(y[indices],pred.to_numpy()[indices])
            groups.append({'group':key,'value':str(value),'n':len(g),'macro_auc':macro,
                          'proxy_brier':float(g.mean_proxy_brier.mean()),'min_positive_count':int(y[indices].sum(0).min()),
                          'min_negative_count':int((~y[indices]).sum(0).min())})
    pd.DataFrame(groups).to_csv(OUT/'exploratory_groups.csv',index=False)
    summary={'eligible_studies':len(study),'series':len(series),'selected_series':len(chosen),
      'studies_losing_plane':int((study.lost_planes>0).sum()),'studies_losing_protocol':int((study.lost_protocols>0).sum()),
      'omitted_protocols':int(study.lost_protocols.sum()),'omitted_series':int(study.omitted_series.sum()),
      'selected_mirror_axis_mismatches':int(chosen.mirror_axis_mismatch.sum()),
      'selected_axis_changes':int(chosen.axis_changes.sum()),'selected_orientation_variation_over_001':int((chosen.orientation_variation>.01).sum()),
      'selected_aspect_distortion_over10pct':int(chosen.aspect_distortion_over10pct.sum()),
      'selected_repeated_input_slices':int((chosen.unique_input_slices<16).sum()),
      'selected_tag_conflicts':int(chosen.tag_conflict.sum()),
      'max_input_gap_mm_quantiles':chosen.max_input_gap_mm.quantile([.05,.5,.95,1]).to_dict(),
      'limitations':['Report-label residuals are biased and historical checkpoints used their evaluation folds for selection.',
                    'Physical aspect distortion is a geometric measurement, not proof of impaired clinical performance.',
                    'No patient identity, clinical adjudication, protected reserve review, model fitting or promotion.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    source=['data/meta/train_series.csv','data/meta/resolution_index/index_train.parquet','outputs/validation_fallback_v2/outer0/roles.csv',
            'outputs/validation_reset_20260924/protocol224_oof.parquet','data/meta/train_labels.parquet','src/model/dataset.py','src/data/cache.py',
            'src/model/lightning.py','tools/audit_baseline_inputs.py']
    (OUT/'source_hashes.json').write_text(json.dumps({n:sha(ROOT/n) for n in source},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
