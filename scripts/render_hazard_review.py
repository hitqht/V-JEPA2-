#!/usr/bin/env python3
"""基于已保存分数生成可读的真值对照视频；不重新推理，不将真值送入模型。"""
import csv
import json
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'data/real_hazard_manifest.json').read_text())
    for sample in manifest['samples']:
        if sample['split']!='test':continue
        out=ROOT/'outputs'/sample['id']
        scores=[]
        for raw in csv.DictReader((out/'scores.csv').open()):
            scores.append({k:float(raw[k]) for k in ('end_s','target_start_s','surprise_z','flagged')})
        cap=cv2.VideoCapture(str(ROOT/sample['path']))
        fps=cap.get(cv2.CAP_PROP_FPS)
        width,height=int(cap.get(3)),int(cap.get(4))
        size=(960,round(height*960/width)//2*2)
        writer=cv2.VideoWriter(str(out/'review_with_truth.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
        if not writer.isOpened():raise RuntimeError('无法写入对照视频')
        count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=count/fps;count+=1
                frame=cv2.resize(frame,size,interpolation=cv2.INTER_LINEAR)
                a=next(a for a in sample['annotations'] if a['start_s']<=t<a['end_s'])
                label=a['label']
                if label=='positive_visible_collapse':truth='MANUAL GT: VISIBLE CRANE COLLAPSE';color=(35,45,235)
                elif label=='negative_no_visible_collapse':truth='MANUAL GT: NO VISIBLE COLLAPSE';color=(60,210,100)
                else:truth='MANUAL GT: EXCLUDED TRANSITION / AFTERMATH';color=(30,195,245)
                # 只有窗口最后输入帧已到达才显示该分数，防止把离线对齐误读为提前预警。
                completed=[r for r in scores if r['end_s']<=t+1e-6]
                r=completed[-1] if completed else None
                cv2.rectangle(frame,(0,0),(size[0],70),(20,20,20),-1)
                cv2.putText(frame,f'Clip {t:05.2f}s | source {t+sample["source_start_s"]:05.2f}s',
                            (16,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(frame,truth,(16,55),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
                cv2.rectangle(frame,(0,size[1]-90),(size[0],size[1]),(20,20,20),-1)
                if r:
                    status='REVIEW' if r['flagged'] else 'below threshold'
                    text=f'Last completed window: z={r["surprise_z"]:.3f} | {status}'
                    detail=f'Scored future frames: {r["target_start_s"]:.2f}-{r["end_s"]:.2f}s | threshold z >= 3'
                else:text='Waiting for first complete input window';detail='No score available yet'
                cv2.putText(frame,text,(16,size[1]-62),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(frame,detail,(16,size[1]-37),cv2.FONT_HERSHEY_SIMPLEX,.54,(220,220,220),1,cv2.LINE_AA)
                cv2.putText(frame,'Offline replay; inference time excluded. Manual labels used only for review.',
                            (16,size[1]-13),cv2.FONT_HERSHEY_SIMPLEX,.49,(200,200,200),1,cv2.LINE_AA)
                writer.write(frame)
                if sample['id']=='big_blue_positive' and abs(t-8)<1/fps/2:
                    cv2.imwrite(str(ROOT/'reports/real_hazard/review_example.jpg'),frame)
        finally:
            cap.release();writer.release()
        if count!=sample['frames']:raise RuntimeError('对照视频帧数不一致')
        print(out/'review_with_truth.mp4')

if __name__=='__main__':main()
