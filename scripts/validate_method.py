#!/usr/bin/env python3
"""预定义的同源配对扰动实验。真实模型、校准/测试时间隔离、全部结果保留。"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vjepa2_site_monitor import (iter_sampled_frames, iter_windows, load_model, pick_device,
                                prediction_tensors, robust_stats, sha256_file)

KINDS = ('occlusion','exposure','freeze','reverse')
METHODS = ('vjepa_l1','persistence_l1','frame_difference')


def auc(negative, positive):
    n, p = np.asarray(negative), np.asarray(positive)
    return float(((p[:,None] > n[None,:]) + .5*(p[:,None] == n[None,:])).mean())


def perturb(x, kind, content_bounds=None):
    x = x.clone()
    split=x.shape[1]//2
    mean=torch.tensor([.485,.456,.406],device=x.device).view(1,1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=x.device).view(1,1,3,1,1)
    if kind=='occlusion':
        edge=x.shape[-1]//4
        x[:,split:,:,edge:-edge,edge:-edge]=((.5-mean)/std)
    elif kind=='exposure':
        y0,y1,x0,x1=content_bounds or (0,x.shape[-2],0,x.shape[-1])
        # 相机曝光仅改变原画面；预处理时添加的均值色补边应保持不变。
        rgb=(x[:,split:,:,y0:y1,x0:x1]*std+mean+.4).clamp(0,1)
        x[:,split:,:,y0:y1,x0:x1]=(rgb-mean)/std
    elif kind=='freeze':
        x[:,split:]=x[:,split-1:split]
    elif kind=='reverse':
        x[:,split:]=x[:,split:].flip(1)
    else:
        raise ValueError(kind)
    return x


@torch.inference_mode()
def measure(model,x):
    full,ctx,pred,tgt=prediction_tensors(model,x)
    normalize=lambda t:F.layer_norm(t.float(),(t.shape[-1],))
    pred,tgt=normalize(pred),normalize(tgt)
    spatial=(x.shape[-1]//model.config.patch_size)**2
    persist=normalize(ctx[:,-spatial:]).repeat(1,tgt.shape[1]//spatial,1)
    std=torch.tensor([.229,.224,.225],device=x.device).view(1,1,3,1,1)
    pixels=x*std
    values={'vjepa_l1':float((pred-tgt).abs().mean()),
            'persistence_l1':float((persist-tgt).abs().mean()),
            'frame_difference':float((pixels[:,1:]-pixels[:,:-1]).abs().mean())}
    if not all(np.isfinite(v) for v in values.values()):
        raise RuntimeError('非有限分数')
    return values,pred.cpu(),tgt.cpu()


def read_windows(video,start,duration,stride):
    return list(iter_windows(iter_sampled_frames(str(video),8,start,duration),16,stride,256,'pad'))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT/'data/raw/excavator.webm')
    parser.add_argument('--model',default=str(ROOT/'models/vjepa2-vitl'))
    parser.add_argument('--device',default='auto')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/validation')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0)
    device=pick_device(args.device)
    model=load_model(args.model,device,torch.float32)
    if model.config.crop_size!=256 or model.config.tubelet_size!=2:
        raise ValueError('本预定义验证仅支持 256 / tubelet=2')
    calibration=read_windows(args.video,0,8,8)
    tests=read_windows(args.video,8,None,16)
    cap=cv2.VideoCapture(str(args.video))
    width,height=cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    scale=256/max(width,height)
    nw,nh=round(width*scale),round(height*scale)
    y0,x0=(256-nh)//2,(256-nw)//2
    bounds=(y0,y0+nh,x0,x0+nw)
    if len(calibration)<3 or len(tests)<2:
        raise ValueError('验证视频长度不足')
    rows=[];start=time.perf_counter();mosaic=[];leak_diffs=[];target_diffs=[]
    def tensor(w):return torch.stack(w[2]).unsqueeze(0).to(device)
    for i,w in enumerate(calibration):
        scores,_,_=measure(model,tensor(w))
        rows.append(dict(split='calibration',clip=i,start_s=w[0],kind='normal',**scores))
        print(f'Calibration {i+1}/{len(calibration)}',flush=True)
    stats={m:robust_stats(np.array([r[m] for r in rows])) for m in METHODS}
    for i,w in enumerate(tests):
        original=tensor(w);normal,pred0,tgt0=measure(model,original)
        rows.append(dict(split='test',clip=i,start_s=w[0],kind='normal',**normal))
        imgs=[]
        for kind in ('normal',*KINDS):
            edited=original if kind=='normal' else perturb(original,kind,bounds)
            if kind!='normal':
                scores,pred,tgt=measure(model,edited)
                rows.append(dict(split='test',clip=i,start_s=w[0],kind=kind,**scores))
                leak_diffs.append(float((pred-pred0).abs().max()))
                target_diffs.append(float((tgt-tgt0).abs().mean()))
            std=np.array([.229,.224,.225]);mean=np.array([.485,.456,.406])
            rgb=((edited[0,12].cpu().permute(1,2,0).numpy()*std+mean)*255).clip(0,255).astype('uint8')
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr,(0,0),(256,27),(30,30,30),-1)
            cv2.putText(bgr,f'{w[0]:.0f}s {kind}',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            imgs.append(bgr)
        mosaic.append(np.concatenate(imgs,axis=1))
        print(f'Test {i+1}/{len(tests)}',flush=True)
    cv2.imwrite(str(args.out/'interventions.jpg'),np.concatenate(mosaic,axis=0))
    for r in rows:
        for m in METHODS:
            med,scale=stats[m];r[m+'_z']=(r[m]-med)/scale
    write_csv(args.out/'scores.csv',rows)
    metrics=[]
    normal=[r for r in rows if r['split']=='test' and r['kind']=='normal']
    for kind in KINDS:
        pos=[r for r in rows if r['kind']==kind]
        for m in METHODS:
            n=np.array([r[m] for r in normal]);p=np.array([r[m] for r in pos])
            metrics.append(dict(kind=kind,method=m,n_positive=len(p),n_negative=len(n),
                                auroc=auc(n,p),paired_mean_delta=float((p-n).mean()),
                                paired_win_rate=float((p>n).mean()),
                                tpr_z3=float(np.mean([r[m+'_z']>=3 for r in pos])),
                                fpr_z3=float(np.mean([r[m+'_z']>=3 for r in normal]))))
    write_csv(args.out/'metrics.csv',metrics)
    summary={'design':'predefined paired synthetic interventions; not real incident detection',
             'protocol_version':'v2-exposure-preserves-padding', 'content_bounds_y0_y1_x0_x1':bounds,
             'device':str(device),'dtype':'float32','calibration_interval_s':[0,8],
             'calibration_windows':len(calibration),'test_windows_per_condition':len(tests),
             'test_starts_s':[w[0] for w in tests],'threshold_z':3,
             'video_sha256':sha256_file(args.video),'method':'past-only-encoder-l1-v1',
             'calibration_statistics':stats,'analysis_seconds':time.perf_counter()-start,
             'future_leak_check_max_abs_prediction_difference':max(leak_diffs),
             'target_change_mean_abs':float(np.mean(target_diffs)),
             'limitations':['single source','synthetic positives','paired dependency','small sample',
                            'camera and illumination confounds','noncausal offline teacher targets'],
             'metrics':metrics}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    if max(leak_diffs)>1e-5:
        raise AssertionError('未来扰动影响了预测器输出')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(len(KINDS));width=.24
    for i,m in enumerate(METHODS):
        ax.bar(x+(i-1)*width,[r['auroc'] for r in metrics if r['method']==m],width,label=m)
    ax.set_xticks(x,KINDS);ax.set_ylim(0,1.08);ax.axhline(.5,color='gray',linestyle='--',linewidth=1)
    ax.set_ylabel('AUROC (higher residual = positive)')
    ax.set_title(f'Controlled interventions: {len(tests)} paired clips per condition')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
