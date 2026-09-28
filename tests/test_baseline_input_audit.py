"""Independent synthetic checks of the audit's sampling and geometry claims."""
import numpy as np
from src.data.cache import select_slices, submission_indices, mirror_axis, resize_volume
from src.model.dataset import PLANE_MIRROR_AXIS

def test_two_stage_sampling_can_lose_distinct_images_even_when_twenty_exist():
    keep=np.linspace(0,23,16).round().astype(int)
    expected=np.array(select_slices(20,24))[keep]
    assert submission_indices(20,16)==expected.tolist()
    assert len(set(expected))==12
    assert expected[0]==0 and expected[-1]==19

def test_unique_cache_positions_can_supply_sixteen_without_more_decoding():
    for n in range(16,101):
        coarse=np.array(select_slices(n,24))
        _,first=np.unique(coarse,return_index=True)
        keep=first[np.linspace(0,len(first)-1,16).round().astype(int)]
        actual=coarse[keep]
        assert len(set(actual))==16
        assert actual[0]==0 and actual[-1]==n-1
        assert np.all(np.diff(keep)>0)

def test_geometry_axis_is_not_plane_invariant_in_general():
    assert mirror_axis([1,0,0,0,1,0])==PLANE_MIRROR_AXIS['Axial']
    assert mirror_axis([0,1,0,1,0,0])==1
    assert mirror_axis([0,1,0,0,0,1])==PLANE_MIRROR_AXIS['Sagittal']

def test_resize_is_whole_frame_square_resampling_not_a_crop():
    volume=np.zeros((1,20,40),dtype=np.float32)
    volume[:,:,0:4]=255;volume[:,:,-4:]=255
    resized=resize_volume(volume,size=20)
    assert resized.shape==(1,20,20)
    assert np.all(resized[0,:,0]>0) and np.all(resized[0,:,-1]>0)
