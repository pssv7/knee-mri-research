"""Stage one CPU-only bounded pixel/side audit on existing Kaggle caches."""
import ast,base64,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BODY=r'''
import base64,io,json,zipfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd
work=Path('/kaggle/working/audit_source');work.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(work)
sys.path.insert(0,str(work))
from src.validation.proxy import sha
root=Path('/kaggle/input')
def one(pattern):
    p=list(root.rglob(pattern))
    if len(p)!=1:raise ValueError(f'{pattern}: {len(p)} matches')
    return p[0]
tech=pd.read_parquet(work/'technical.parquet')
prior=json.loads((work/'prior_inputs.json').read_text())
frames=[];arrays={}
for source in prior['shards']:
    array=one(Path(source['array']['path']).name)
    manifest=one(Path(source['manifest']['path']).name)
    assert array.stat().st_size==source['array']['bytes']
    assert sha(manifest)==source['manifest']['sha256']
    m=pd.read_parquet(manifest);m['array']=str(array);frames.append(m)
    arrays[str(array)]=np.load(array,mmap_mode='r')
sidepath=one('side_predictions.parquet')
assert sha(sidepath)==prior['side_predictions']['sha256']
sides=pd.read_parquet(sidepath)
joined=tech.merge(sides[['SeriesInstanceUID','side_predicted']],on='SeriesInstanceUID',validate='one_to_one')
known=joined[joined.tag_side.isin(['L','R'])]
joined['tag_prediction_disagreement']=joined.tag_side.isin(['L','R']) & (joined.tag_side!=joined.side_predicted)
joined[['SeriesInstanceUID','StudyInstanceUID','selected','tag_side','side_predicted','tag_prediction_disagreement']].to_parquet('/kaggle/working/side_audit.parquet',index=False)
selected=tech[tech.selected].sort_values('SeriesInstanceUID')
# Fixed sample, chosen without label residuals. Separate risk enrichment.
representative=selected.sample(n=96,random_state=20260924).copy();representative['sample']='random_series'
risk=selected[selected.aspect_distortion_over10pct | ((selected.raw_slices>=16)&(selected.unique_input_slices<16))].sample(n=24,random_state=20260924).copy();risk['sample']='enriched_technical'
sample=pd.concat([representative,risk]).drop_duplicates('SeriesInstanceUID')
sample=sample.merge(pd.concat(frames)[['SeriesInstanceUID','position','array']],on='SeriesInstanceUID',validate='one_to_one')
rows=[];keep=np.linspace(0,23,16).round().astype(int)
for row in sample.to_dict('records'):
    volume=arrays[row['array']][int(row['position'])]
    v=volume[keep]
    unique=len({sl.tobytes() for sl in v})
    border=np.concatenate([v[:,:4,:].ravel(),v[:,-4:,:].ravel(),v[:,:, :4].ravel(),v[:,:,-4:].ravel()])
    rows.append({'SeriesInstanceUID':row['SeriesInstanceUID'],'sample':row['sample'],
        'raw_slices':row['raw_slices'],'predicted_unique':row['unique_input_slices'],
        'pixel_unique':unique,'all_zero':bool(not np.any(v)),
        'std':float(v.std()),'zero_fraction':float(np.mean(v==0)),
        'saturated_fraction':float(np.mean(v==255)),'border_nonzero_fraction':float(np.mean(border>0))})
pd.DataFrame(rows).to_csv('/kaggle/working/pixel_sample.csv',index=False)
# Technical montage: middle sampled slice, no labels/reports or clinical interpretation.
from PIL import Image,ImageDraw
sheet=Image.new('L',(6*224,4*248));draw=ImageDraw.Draw(sheet)
for i,row in enumerate(sample.head(24).to_dict('records')):
    v=arrays[row['array']][int(row['position'])]
    x,y=(i%6)*224,(i//6)*248
    sheet.paste(Image.fromarray(v[12]),(x,y));draw.text((x,y+225),str(i+1),fill=255)
sheet.save('/kaggle/working/technical_montage.png')
sample.head(24)[['SeriesInstanceUID','sample','physical_aspect_ratio','raw_slices']].to_csv('/kaggle/working/montage_key.csv',index=False)
r={'status':'complete','training_launched':False,'eligible_series':len(joined),'series_with_tag_side':len(known),
 'tag_prediction_disagreements':int(joined.tag_prediction_disagreement.sum()),
 'studies_with_mixed_predicted_sides':int((joined.groupby('StudyInstanceUID').side_predicted.nunique()>1).sum()),
 'pixel_sample_series':len(rows),'random_series_sample':96,'enriched_sample_requested':24,
 'all_zero_volumes':sum(x['all_zero'] for x in rows),'pixel_unique_matches_header_prediction':sum(x['pixel_unique']==x['predicted_unique'] for x in rows),
 'limitations':['Tags are technical references, not independent laterality adjudication.','Pixel sample is series-based; enrichment is separate, not prevalence evidence.','Border intensity cannot establish clinical crop adequacy.','Small manifests and sides rehashed; full arrays size-checked against prior verified preflight, not rehashed.']}
Path('/kaggle/working/audit_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
'''

def main():
    out=ROOT/'outputs/kaggle_baseline_pixel_audit_v2';out.mkdir(exist_ok=False)
    tree=ast.parse((ROOT/'outputs/kaggle_nested_proxy_preflight_v1/preflight.py').read_text())
    old=base64.b64decode(ast.literal_eval(tree.body[0].value))
    stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old)) as src,zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as dest:
        needed={'src/__init__.py','src/config.py','src/labels/__init__.py','src/labels/calibrate.py','src/validation/__init__.py','src/validation/proxy.py'}
        for name in src.namelist():
            if name in needed:dest.writestr(name,src.read(name))
        import pandas as pd
        columns=['SeriesInstanceUID','StudyInstanceUID','selected','tag_side','aspect_distortion_over10pct','physical_aspect_ratio','raw_slices','unique_input_slices']
        technical=pd.read_parquet(ROOT/'outputs/baseline_input_audit_v1/series_technical.parquet',columns=columns)
        buf=io.BytesIO();technical.to_parquet(buf,index=False)
        dest.writestr('technical.parquet',buf.getvalue())
        dest.write(ROOT/'outputs/kaggle_nested_proxy_preflight_v1_results/nested_proxy_preflight/inputs.json','prior_inputs.json')
    code='PAYLOAD = '+repr(base64.b64encode(stream.getvalue()).decode())+'\n'+BODY
    compile(code,'audit.py','exec');(out/'audit.py').write_text(code)
    meta=json.loads((ROOT/'outputs/kaggle_nested_proxy_preflight_v1/kernel-metadata.json').read_text())
    meta.update(id='pssv27/knee-baseline-pixel-audit-v1',title='Knee Baseline Pixel Audit V1',code_file='audit.py')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(out)
if __name__=='__main__':main()
