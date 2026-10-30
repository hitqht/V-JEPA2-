from scripts.validate_real_hazard import classify_window,binary_metrics
import pytest

ANNOTATIONS=[{'start_s':0,'end_s':3,'label':'negative_no_visible_collapse'},
             {'start_s':3,'end_s':5,'label':'excluded_onset_transition'},
             {'start_s':5,'end_s':11,'label':'positive_visible_collapse'},
             {'start_s':11,'end_s':16,'label':'excluded_aftermath'}]


@pytest.mark.parametrize('start,end,expected',[(1,1.9,0),(2,2.9,0),(3,3.9,-1),
                                             (4.5,5.4,-1),(5,5.9,1),(10,10.9,1),
                                             (10.5,11.4,-1),(11,11.9,-1)])
def test_only_fully_contained_future_is_labeled(start,end,expected):
    assert classify_window({'target_start_s':start,'end_s':end},ANNOTATIONS)==expected


def test_real_metrics_keep_false_alarms_and_misses():
    m=binary_metrics([0,0,1,1],[.1,.9,.2,1.],[0,4,1,5])
    assert (m['tp'],m['fp'],m['tn'],m['fn'])==(1,1,1,1)
    assert m['auroc']==.75 and m['precision']==.5 and m['recall']==.5 and m['fpr']==.5


def test_no_prediction_does_not_imply_perfect_precision():
    m=binary_metrics([0,1],[0,0],[0,0])
    assert m['precision'] is None and m['recall']==0 and m['f1']==0
    assert m['auroc']==.5


def test_excluded_labels_must_be_filtered_before_metrics():
    with pytest.raises(ValueError):binary_metrics([-1,1],[0,1],[0,4])
