#!/usr/bin/env python3
"""真实施工倒塌案例：固定校准、显式时间真值、全部结果、同算法基线。"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vjepa2_site_monitor import (iter_sampled_frames,iter_windows,load_model,pick_device,
                                robust_stats,sha256_file)
from scripts.validate_method import measure,auc,write_csv,METHODS


def classify_window(row,annotations):
    """仅当所有未来采样帧落在一个完整已标区间才纳入统计。"""
    start,end=row['target_start_s'],row['end_s']
    if end<start:raise ValueError('目标时间反向')
    for a in annotations:
        if start>=a['start_s']-1e-6 and end<a['end_s']-1e-6:
            if a['label']=='positive_visible_collapse':return 1
            if a['label']=='negative_no_visible_collapse':return 0
    return -1


def divide(a,b):return float(a/b) if b else None


def binary_metrics(labels,scores,zscores):
    labels=np.asarray(labels);scores=np.asarray(scores);pred=np.asarray(zscores)>=3
    if not np.isin(labels,[0,1]).all():raise ValueError('指标输入含排除标签')
    tp=int(((labels==1)&pred).sum());fn=int(((labels==1)&~pred).sum())
    fp=int(((labels==0)&pred).sum());tn=int(((labels==0)&~pred).sum())
    recall=divide(tp,tp+fn);specificity=divide(tn,tn+fp)
    return {'n_positive':tp+fn,'n_negative':tn+fp,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'auroc':auc(scores[labels==0],scores[labels==1]) if (labels==0).any() and (labels==1).any() else None,
            'precision':divide(tp,tp+fp),'recall':recall,'f1':divide(2*tp,2*tp+fp+fn),
            'fpr':divide(fp,fp+tn),'balanced_accuracy':(recall+specificity)/2
            if recall is not None and specificity is not None else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/real_hazard_manifest.json')
    p.add_argument('--model',default=str(ROOT/'models/vjepa2-vitl'))
    p.add_argument('--device',default='auto')
    p.add_argument('--out',type=Path,default=ROOT/'reports/real_hazard')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    if manifest['dataset_version']!='real-hazard-v1':raise ValueError('未知数据集版本')
    cfg=manifest['evaluation_settings']
    if cfg!={'fps':8,'window':16,'stride':8,'fit':'pad','dtype':'float32','threshold_z':3}:
        raise ValueError('本次预定义设置发生改变')
    # 在载入模型前核查输入和三份不交叠切片。
    intervals=sorted((s['source_start_s'],s['source_end_s']) for s in manifest['samples'])
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('切片共享源帧')
    for s in manifest['samples']:
        if sha256_file(ROOT/s['path'])!=s['sha256']:raise ValueError('切片 SHA256 不符')
    torch.manual_seed(0)
    device=pick_device(args.device);model=load_model(args.model,device,torch.float32)
    if model.config.crop_size!=256 or model.config.tubelet_size!=2:raise ValueError('模型配置不符')
    rows=[];t0=time.perf_counter();max_diff=0
    for sample in manifest['samples']:
        reference=ROOT/'outputs'/sample['id']/'scores.csv'
        if not reference.exists():raise FileNotFoundError(f'先运行主分析：{reference}')
        standard=list(csv.DictReader(reference.open()))
        windows=iter_windows(iter_sampled_frames(str(ROOT/sample['path']),8,0,None),16,8,256,'pad')
        count=0
        for i,w in enumerate(windows):
            x=torch.stack(w[2]).unsqueeze(0).to(device)
            values,_,_=measure(model,x)
            row={'sample':sample['id'],'split':sample['split'],'window':i,
                 'start_s':round(w[0],3),'end_s':round(w[1],3),
                 'target_start_s':round(w[4],3),'source_offset_s':sample['source_start_s']}
            row['label']=classify_window(row,sample['annotations'])
            if sample['split']=='calibration' and row['label']!=0:raise ValueError('校准包含危险/不明帧')
            row.update(values);rows.append(row);count+=1
            if i>=len(standard):raise ValueError('主分析窗口数不一致')
            r=standard[i]
            if any(abs(row[k]-float(r[k]))>1e-3 for k in ('start_s','end_s','target_start_s')):
                raise ValueError('主分析时间戳不一致')
            max_diff=max(max_diff,abs(values['vjepa_l1']-float(r['surprise'])))
        if count!=len(standard):raise ValueError('主分析窗口数不一致')
        print(f'{sample["id"]}: {count} windows',flush=True)
    if max_diff>1e-5:raise ValueError(f'主分析和评估分数不一致: {max_diff}')
    calibration=[r for r in rows if r['split']=='calibration']
    stats={m:robust_stats(np.array([r[m] for r in calibration])) for m in METHODS}
    for r in rows:
        for m in METHODS:r[m+'_z']=(r[m]-stats[m][0])/stats[m][1]
    tests=[r for r in rows if r['split']=='test' and r['label']>=0]
    metrics=[]
    positive_sample=next(s for s in manifest['samples'] if any(a['label']=='positive_visible_collapse' for a in s['annotations']))
    event=next(a for a in positive_sample['annotations'] if a['label']=='positive_visible_collapse')
    for m in METHODS:
        metric={'method':m,**binary_metrics([r['label'] for r in tests],[r[m] for r in tests],
                                          [r[m+'_z'] for r in tests])}
        flagged_positive=[r for r in tests if r['label']==1 and r[m+'_z']>=3]
        metric['event_detected']=bool(flagged_positive)
        metric['first_detection_last_frame_delay_s']=min(r['end_s'] for r in flagged_positive)-event['start_s'] if flagged_positive else None
        metrics.append(metric)
    write_csv(args.out/'scores.csv',rows);write_csv(args.out/'metrics.csv',metrics)
    summary={'dataset_version':manifest['dataset_version'],'manifest_sha256':sha256_file(args.manifest),
             'model_manifest':json.loads((Path(args.model)/'download_manifest.json').read_text()),
             'device':str(device),'dtype':'float32','settings':cfg,'calibration_windows':len(calibration),
             'independent_real_incidents':1,'evaluated_test_windows':len(tests),
             'excluded_test_windows':sum(r['split']=='test' and r['label']==-1 for r in rows),
             'calibration_statistics':stats,'main_score_max_absolute_difference':max_diff,
             'analysis_seconds':time.perf_counter()-t0,'metrics':metrics}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    make_plot(args.out,rows,manifest)
    write_report(args.out,summary,manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_plot(out,rows,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,m in zip(axes,METHODS):
        ax.axvspan(34,42,color='gray',alpha=.14,label='calibration')
        ax.axvspan(55,61,color='red',alpha=.12,label='visible collapse')
        for sample in manifest['samples']:
            rs=[r for r in rows if r['sample']==sample['id']]
            ts=[r['end_s']+r['source_offset_s'] for r in rs]
            ax.plot(ts,[r[m+'_z'] for r in rs],marker='.',lw=1.5,color='#2368ad')
            for r,t in zip(rs,ts):
                if r['split']=='test' and r['label']>=0:
                    ax.scatter([t],[r[m+'_z']],c='#c83232' if r['label']==1 else '#26784a',s=28,zorder=4)
        ax.axhline(3,ls='--',color='gray',lw=1);ax.set_ylabel(m+' z');ax.grid(alpha=.2)
    axes[0].legend(loc='upper left',fontsize=8)
    axes[0].set_title('One real crane-collapse incident; green = negative, red = positive test windows')
    axes[-1].set_xlabel('Original source time (s); score available after last sampled input frame')
    fig.tight_layout();fig.savefig(out/'timeline.png',dpi=160);plt.close(fig)


def fmt(value):return '未定义' if value is None else f'{value:.3f}'


def pct(value):return '未定义' if value is None else f'{value:.1%}'


def write_report(out,summary,manifest):
    metrics=summary['metrics'];v=metrics[0]
    table='\n'.join(f"| {m['method']} | {fmt(m['auroc'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['fpr'])} |" for m in metrics)
    detected='检出' if v['event_detected'] else '未检出'
    delay='未检出，无法计算' if v['first_detection_last_frame_delay_s'] is None else f"{v['first_detection_last_frame_delay_s']:.2f} 秒"
    report=f'''# 真实施工危险正样本：Big Blue 倒塌案例实测

本报告为新增主实验；原有合成扰动实验继续保留在 [VALIDATION.md](VALIDATION.md)，二者不混算。日期：2026-10-30。

## 视频与逐段真值

选用 [Big Blue crane collapse at Miller Park](https://commons.wikimedia.org/wiki/File:Big_Blue_crane_collapse_at_Miller_Park.webm)。Commons 标记为未具名 OSHA 雇员拍摄的美国联邦政府公共领域作品；[OSHA 案例页](https://obis.osha.gov/dcsp/success_stories/compliance_assistance/abbott/abbott_casestudies/slide30.html)说明其为施工中的真实起重机倒塌。已核对源文件 SHA1 `{manifest['source']['sha1']}` 与 Commons 公开记录一致。

下载原片后，在读取新视频的模型分数前，人工查看原始帧并固定 [评估方案](../docs/real_hazard_protocol.md)和 [manifest](../data/real_hazard_manifest.json)。源 SHA256：`{manifest['source']['sha256']}`。

| 切片 | 源时间 | 时长 | 用途 |
|---|---|---:|---|
| big_blue_calibration.mp4 | 34–42 s | 8 s | 无可见倒塌的校准片段 |
| big_blue_negative.mp4 | 42–50 s | 8 s | 无可见倒塌的测试对照，包含运镜 |
| big_blue_positive.mp4 | 50–66 s | 16 s | 包含真实失稳/倒塌过程的危险正样本 |

正片内 5–11 秒（源 55–61 秒）标为可见失稳/倒塌，0–3 秒为无可见倒塌，3–5 秒过渡和 11–16 秒事后画面排除统计。边界约 ±0.5 秒不确定性；“无可见倒塌”不表示工况安全。三个片段没有共享源帧，只包含 **1 个独立真实事故**。未插入合成危险或拼接不同事件。

![短样本逐段画面](real_hazard/samples_contact_sheet.jpg)

## 设置和实际检查

沿用真实预训练 V-JEPA 2 ViT-L（`{summary['model_manifest']['revision']}`）、MPS/float32、8 fps、16 帧窗口、8 帧步长、pad、LayerNorm 后 L1 和 z≥3。仅事前 {summary['calibration_windows']} 个窗口估计每种方法自己的正常基线，不训练、不用测试标签调阈值。

分类要求未来半窗全部采样帧落在已标正或负区间；跨边界、过渡和事后窗口保存为 excluded。测试纳入 {summary['evaluated_test_windows']} 个窗口，排除 {summary['excluded_test_windows']} 个；正负数量见表。未来半窗才定义评分所对应的事件范围，标签只用于推理后评估。

主分析程序分别生成了三份 CSV、图像及正/负片段复核视频；评估重算 V-JEPA 分数与主程序最大绝对差为 `{summary['main_score_max_absolute_difference']:.9g}`。评估阶段耗时 {summary['analysis_seconds']:.2f} 秒，不含三次主分析、模型加载或视频编码。

## 固定方案结果

| 方法 | AUROC | TP | FP | TN | FN | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

`persistence_l1` 是保持最后上下文 tubelet 的潜特征基线；`frame_difference` 是平均相邻帧像素差。全部指标和 F1/平衡准确率见 [metrics.csv](real_hazard/metrics.csv)。AUROC 高不保证固定阈值误报低，且不同方法原始量纲不同。

V-JEPA 2 残差对本次事故：**{detected}**。首次检出窗口的最后输入帧相对可见倒塌起点的时间差：**{delay}**。该值不含推理开销；事前高分按负例误报计，不能称为事故提前预警。

窗口 Recall={pct(v['recall'])}、Precision={pct(v['precision'])}、FPR={pct(v['fpr'])}、平衡准确率={fmt(v['balanced_accuracy'])}。本次事件被命中不代表方法可靠；尤其运镜产生了明显的高分误报。事件数只有 1，不能由“本例检出”报告普遍的 100% 事故检出率。

![源时间轴、真值和分数](real_hazard/timeline.png)

## 能支持的结论和局限

这次已实际使用包含真实危险正样本的短视频完成完整分析和定量评估。应同时根据上表的检出和误报评价方法，不能只展示最高热图或只报 AUROC。

独立事故数仅 1，所有测试窗口来自同一录像并有相邻依赖，分辨率仅 352×262。手持运镜与倒塌运动混杂，校准段短，人工边界未经第二位标注者复核。当前结果不能支持跨工地泛化准确率、施工规则识别或事故前危险预判。负样本的运镜和事后排除区间均记录在原始分数表，未隐藏失败窗口。

此前合成扰动的结果和不足不变；它们是补充的软件/敏感性实验。模型权重、主分析方法和阈值没有因本案例结果改动。

复现：`bash scripts/run_real_hazard.sh`。原始视频、派生切片和模型保存在本机；来源、权利说明及编辑记录见 [ATTRIBUTION.md](../docs/ATTRIBUTION.md)。
'''
    (ROOT/'reports/REAL_HAZARD_VALIDATION.md').write_text(report)

if __name__=='__main__':main()
