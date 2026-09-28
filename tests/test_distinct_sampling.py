import numpy as np
import pandas as pd
import pytest
from src.data import cache, sampling
from src.model.dataset import StudyIndex, load_study
from src.model.lightning import StudyVolumeDataset
from src import config

@pytest.mark.parametrize('n',range(1,101))
@pytest.mark.parametrize('reverse',[False,True])
def test_unique_and_unchanged(n,reverse):
    raw=np.arange(n,dtype=np.uint8)[:,None,None]
    full=raw[cache.select_slices(n,24)]
    v=full[::-1] if reverse else full
    actual=sampling.sample_cached_volume(v,n,duplicate_free=True,depth_reversed=reverse)
    assert actual.shape==(16,1,1)
    assert actual[0]==v[0] and actual[-1]==v[-1]
    if 17<=n<=23:assert len(np.unique(actual))==16
    else:np.testing.assert_array_equal(actual,v[np.linspace(0,23,16).round().astype(int)])

@pytest.mark.parametrize('plane',['Sagittal','Coronal','Axial'])
@pytest.mark.parametrize('side',['L','R'])
@pytest.mark.parametrize('n',[12,16,17,20,23,24,40])
@pytest.mark.parametrize('enabled',[False,True])
def test_training_inference_pixel_parity(monkeypatch,tmp_path,plane,side,n,enabled):
    # Intensity gradients make normalization differences and spatial flips visible.
    raw=np.arange(n*6*8,dtype=np.float32).reshape(n,6,8)
    headers=[{'path':str(i),'StudyInstanceUID':'s','SeriesInstanceUID':'v','InstanceNumber':i,'Rows':6,'Columns':8} for i in range(n)]
    monkeypatch.setattr(cache.dicom_io,'load_series_volume',lambda paths:raw[[int(p) for p in paths]])
    built=cache.build_series_volume(headers,size=8)
    path=tmp_path/'cache.npy';np.save(path,built['volume'][None])
    manifest=pd.DataFrame({'StudyInstanceUID':['s'],'SeriesInstanceUID':['v'],'position':[0],
        'cache_path':[str(path)],'n_slices_raw':[n],'mirrored':[False]})
    meta=pd.DataFrame({'SeriesInstanceUID':['v'],'Anatomical_Plane':[plane],'Fluid_Sensitive':[1],'Fat_Suppression':[0]})
    sides=pd.DataFrame({'SeriesInstanceUID':['v'],'side_predicted':[side]})
    index=StudyIndex(manifest,meta,sides=sides)
    train=StudyVolumeDataset(index,['s'],[str(path)],16,duplicate_free=enabled)[0]['volumes'][0]
    infer=cache.build_model_series_volume(headers,side,plane,duplicate_free=enabled,size=8)['volume']
    np.testing.assert_array_equal(train,infer)

@pytest.mark.parametrize('count',[None,np.nan,0,-1,17.5,True])
def test_bad_provenance_rejected(count):
    with pytest.raises(ValueError):sampling.sample_cached_volume(np.zeros((24,2,2)),count,duplicate_free=True)

def test_changed_repeated_cache_slots_rejected():
    v=np.arange(20,dtype=np.uint8)[cache.select_slices(20,24)][:,None,None]
    pos=np.flatnonzero(np.diff(np.array(cache.select_slices(20,24)))==0)[0]
    v[pos+1]=255
    with pytest.raises(ValueError,match='provenance'):sampling.sample_cached_volume(v,20,duplicate_free=True)

def test_incomplete_inference_decode_rejected(monkeypatch):
    monkeypatch.setattr(cache.dicom_io,'load_series_volume',lambda _:np.ones((23,2,2)))
    with pytest.raises(ValueError,match='Incomplete decode'):
        cache.build_model_series_volume([{'path':'x','InstanceNumber':i} for i in range(20)],'R','Sagittal')

def test_resampling_cannot_be_combined():
    with pytest.raises(ValueError,match='fixed 16'):
        StudyVolumeDataset(None,[],[],16,resample=True,duplicate_free=True)
