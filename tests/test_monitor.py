from pathlib import Path
from types import SimpleNamespace
import sys
import cv2
import numpy as np
import pytest
import torch
from transformers import VJEPA2Config,VJEPA2Model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vjepa2_site_monitor as m
from scripts.validate_method import auc,perturb
from scripts.download_model import rename


def test_sample_timestamps_and_exclusive_duration(tmp_path):
    path=tmp_path/'v.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for i in range(30):writer.write(np.full((32,32,3),i,np.uint8))
    writer.release()
    frames=list(m.iter_sampled_frames(str(path),4,0.2,1.0))
    assert [round(t,2) for t,_ in frames]==[.2,.5,.7,1.0]
    # 抽帧率高于源帧率时不复制帧，也不反复抽取同一时刻。
    assert len(list(m.iter_sampled_frames(str(path),20,0,1)))==10


def test_window_target_timestamp_is_actual_frame():
    frames=[(i*.125,np.zeros((24,40,3),np.uint8)) for i in range(24)]
    windows=list(m.iter_windows(frames,16,8,32,'pad'))
    assert len(windows)==2
    assert windows[0][4]==1.0
    assert windows[0][1]==1.875


def test_disjoint_flagged_windows_are_not_merged():
    def row(s,e):return dict(target_start_s=s,end_s=e,surprise_z=4.,change_z=0,flagged=1)
    assert len(m.merge_events([row(1,2),row(4,5)]))==2
    assert len(m.merge_events([row(1,2),row(2,3)]))==1


def test_constant_scale_is_finite():
    med,scale=m.robust_stats(np.ones(5))
    assert med==1 and scale>0 and np.isfinite(scale)


def test_future_invariance_real_architecture():
    # 小型随机模型只检验程序和掩码机制；它不是有效性实验中的预训练权重。
    torch.manual_seed(4)
    config=VJEPA2Config(crop_size=32,patch_size=16,frames_per_clip=8,
                       hidden_size=128,num_hidden_layers=1,num_attention_heads=4,
                       pred_hidden_size=128,pred_num_attention_heads=4,pred_num_hidden_layers=1)
    model=VJEPA2Model(config).eval()
    x=torch.randn(1,8,3,32,32)
    edited=x.clone();edited[:,4:]=torch.randn_like(edited[:,4:])
    full,ctx,pred,target=m.prediction_tensors(model,x)
    _,ctx2,pred2,target2=m.prediction_tensors(model,edited)
    torch.testing.assert_close(ctx,ctx2,rtol=0,atol=0)
    torch.testing.assert_close(pred,pred2,rtol=0,atol=0)
    assert not torch.equal(target,target2)
    scores,heat,emb=m.score_batch(model,x)
    assert scores.shape==(1,) and heat.shape==(1,2,2) and emb.shape==(1,128)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize('kind',['occlusion','exposure','freeze','reverse'])
def test_interventions_preserve_past(kind):
    x=torch.randn(1,16,3,32,32)
    y=perturb(x,kind)
    assert torch.equal(x[:,:8],y[:,:8])
    assert not torch.equal(x[:,8:],y[:,8:])


def test_auroc_ties_and_direction():
    assert auc([1,2],[3,4])==1
    assert auc([3,4],[1,2])==0
    assert auc([1,1],[1,1])==.5


def test_exposure_preserves_preprocessing_padding():
    x=torch.zeros(1,16,3,32,32)
    y=perturb(x,'exposure',(8,24,0,32))
    assert torch.equal(x[:,:, :, :8],y[:,:, :, :8])
    assert torch.equal(x[:,:, :, 24:],y[:,:, :, 24:])
    assert not torch.equal(x[:,8:,:,8:24],y[:,8:,:,8:24])


def test_conversion_name():
    assert rename('module.backbone.blocks.0.attn.proj.weight','encoder')=='encoder.layer.0.attention.proj.weight'
    assert rename('module.backbone.predictor_norm.weight','predictor')=='predictor.layernorm.weight'


@pytest.mark.parametrize('extra',[['--fps','0'],['--batch','0'],['--top-k','-1'],['--duration','nan'],['--window','6']])
def test_invalid_cli(tmp_path,extra):
    video=tmp_path/'exists.mp4';video.touch()
    with pytest.raises(SystemExit):m.parse_args([str(video),*extra])


def test_end_to_end_outputs_and_external_baseline(tmp_path,monkeypatch):
    import json
    torch.manual_seed(8)
    config=VJEPA2Config(crop_size=32,patch_size=16,frames_per_clip=16,
                       hidden_size=128,num_hidden_layers=1,num_attention_heads=4,
                       pred_hidden_size=128,pred_num_attention_heads=4,pred_num_hidden_layers=1)
    model=VJEPA2Model(config).eval()
    monkeypatch.setattr(m,'load_model',lambda *args:model)
    path=tmp_path/'test.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),8,(32,32))
    for i in range(32):writer.write(np.full((32,32,3),30+i*3,np.uint8))
    writer.release()
    first=tmp_path/'first'
    rows,events=m.main([str(path),'--device','cpu','--out',str(first),'--annotated-video'])
    assert len(rows)==3
    assert np.load(first/'features.npz')['heatmaps'].shape==(3,2,2)
    assert json.loads((first/'summary.json').read_text())['calibration']=='self-descriptive'
    review=cv2.VideoCapture(str(first/'review.mp4'))
    assert review.isOpened() and review.get(cv2.CAP_PROP_FRAME_COUNT)==32
    review.release()
    second=tmp_path/'second'
    m.main([str(path),'--device','cpu','--out',str(second),'--baseline-csv',str(first/'scores.csv')])
    assert json.loads((second/'summary.json').read_text())['calibration']=='external'
    with pytest.raises(ValueError,match='输出目录非空'):
        m.main([str(path),'--out',str(first)])
