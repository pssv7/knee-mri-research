import numpy as np
import pandas as pd
import pytest

from src.labels import fuse, mode_fusion
from src.labels.extraction_repair import repair
from src.labels.openai_extract import FIELDS
from src.validation.proxy import ID, LABELS, training_only_sharpen


@pytest.fixture
def data():
    ids = [f"s{i}" for i in range(6)]
    roles = pd.DataFrame({ID: ids, "component": ids, "role": ["train"]*4+["selection","outer"]})
    reports = pd.DataFrame({ID:ids,"Report":["The ACL has a tear. Mild joint effusion."]*6}).set_index(ID)
    cat = pd.DataFrame({ID:ids});prob = pd.DataFrame({ID:ids})
    for finding in LABELS:
        cat[finding] = [np.nan,1.,1.,0.,1.,0.]
        cat[finding+"_severity"] = [0.,.35,.85,0.,.85,0.]
        cat[finding+"_confidence"] = 1.
        prob[finding] = [.1,.7,.9,.2,.7,.2]
    lex = fuse.lexicon_frame(reports.loc[ids,'Report'], ids)
    raw = fuse.fuse_scores(lex,mode_fusion.combine(cat,prob))
    ref = fuse.fuse_scores(lex,mode_fusion.from_categorical(cat))
    control = training_only_sharpen(raw,ref,ids[:4])
    return control, roles, cat, prob, reports


def assertion(state='present', severity='moderate'):
    result = {f:{field:'' for field in FIELDS} for f in LABELS}
    for item in result.values():item['report_states']='unmentioned'
    result['ACL'] = dict(report_states=state,anatomical_location='ACL',certainty='possible' if state=='uncertain' else 'definite',
                         severity_text='',severity_norm=severity,evidence_quote='The ACL has a tear.')
    return result


def test_no_candidate_exactly_reproduces_original(data):
    result,changes,_=repair(*data,{})
    pd.testing.assert_frame_equal(result,data[0])
    assert not changes


def test_only_new_definite_call_changes_and_weights_hold(data):
    result,changes,_=repair(*data,{'s0':assertion(),'s1':assertion(severity='severe'),'s2':assertion('uncertain')})
    assert len(changes)==1 and changes[0][ID]=='s0' and changes[0]['finding']=='ACL'
    assert changes[0]['new_target']>changes[0]['old_target']
    original=data[0].set_index(ID);candidate=result.set_index(ID)
    pd.testing.assert_frame_equal(candidate.loc[['s1','s2','s3','s4','s5']],original.loc[['s1','s2','s3','s4','s5']])
    pd.testing.assert_frame_equal(candidate[[f+'_weight' for f in LABELS]],original[[f+'_weight' for f in LABELS]])


def test_rejects_heldout_and_changed_control_mapping(data):
    with pytest.raises(ValueError,match='nontraining'):
        repair(*data,{'s4':assertion()})
    changed=data[0].copy();changed.loc[0,'ACL']+=.01
    with pytest.raises(ValueError,match='does not reproduce'):
        repair(changed,*data[1:],{})


def test_no_repair_from_silence_or_invalid_evidence(data):
    values=assertion();values['ACL']={field:'' for field in FIELDS};values['ACL']['report_states']='unmentioned'
    result,changes,_=repair(*data,{'s0':values})
    assert not changes
    pd.testing.assert_frame_equal(result,data[0])
    values=assertion();values['ACL']['evidence_quote']='Invented quote'
    with pytest.raises(ValueError,match='substring'):
        repair(*data,{'s0':values})
