#!/usr/bin/env python3
"""下载可追溯的真实施工倒塌录像，生成短正样本及互不重叠的校准/测试片段。"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vjepa2_site_monitor import sha256_file
URL='https://upload.wikimedia.org/wikipedia/commons/0/0f/Big_Blue_crane_collapse_at_Miller_Park.webm'
PAGE='https://commons.wikimedia.org/wiki/File:Big_Blue_crane_collapse_at_Miller_Park.webm'
SHA256='fa4c0de9e1f7e5961d10fdd85e636b10dc99acb5fa41e5f7b0e3059426c71e4e'
SHA1='73fc5052f91beddbcfc18a8c9988cb00042e1886'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-download',action='store_true')
    args=p.parse_args()
    raw=ROOT/'data/raw/big_blue_crane_collapse.webm'
    raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists():
        if args.skip_download:raise SystemExit('缺少源文件')
        part=raw.with_suffix('.webm.part')
        subprocess.run(['curl','-fL','--retry','2',URL,'-o',str(part)],check=True)
        if sha256_file(part)!=SHA256:raise SystemExit('下载文件 SHA256 不符')
        part.replace(raw)
    if sha256_file(raw)!=SHA256 or hashlib.sha1(raw.read_bytes()).hexdigest()!=SHA1:
        raise SystemExit('源文件版本与已核查版本不符')
    dest=ROOT/'data/derived';dest.mkdir(exist_ok=True)
    specs=[
        ('big_blue_calibration',34,42,'calibration',
         [{'start_s':0,'end_s':8,'label':'negative_no_visible_collapse'}]),
        ('big_blue_negative',42,50,'test',
         [{'start_s':0,'end_s':8,'label':'negative_no_visible_collapse'}]),
        ('big_blue_positive',50,66,'test',
         [{'start_s':0,'end_s':3,'label':'negative_no_visible_collapse'},
          {'start_s':3,'end_s':5,'label':'excluded_onset_transition'},
          {'start_s':5,'end_s':11,'label':'positive_visible_collapse'},
          {'start_s':11,'end_s':16,'label':'excluded_aftermath'}]),
    ]
    samples=[];sheets=[]
    for sample_id,start,end,split,annotations in specs:
        path=dest/(sample_id+'.mp4')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-ss',str(start),
                        '-t',str(end-start),'-an','-vf','fps=30','-c:v','libx264','-crf','18',
                        '-pix_fmt','yuv420p',str(path)],check=True)
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps!=30 or frames!=30*(end-start):raise RuntimeError('切片帧数/帧率不符')
        images=[]
        for t in np.arange(0,end-start,2):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,img=cap.read()
            if not ok:raise RuntimeError('切片解码失败')
            img=cv2.resize(img,(264,196));cv2.rectangle(img,(0,0),(264,25),(20,20,20),-1)
            label=f'{sample_id.removeprefix("big_blue_")} {t:.0f}s / source {start+t:.0f}s'
            cv2.putText(img,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            images.append(img)
        cap.release()
        for i in range(0,len(images),4):sheets.append(np.concatenate(images[i:i+4],axis=1))
        samples.append({'id':sample_id,'path':str(path.relative_to(ROOT)),
                        'sha256':sha256_file(path),'split':split,'source_start_s':start,
                        'source_end_s':end,'duration_s':end-start,'fps':fps,'frames':frames,
                        'annotations':annotations})
    cv2.imwrite(str(ROOT/'reports/real_hazard/samples_contact_sheet.jpg'),np.concatenate(sheets,axis=0))
    manifest={'dataset_version':'real-hazard-v1','created_date':'2026-10-30',
              'independent_real_incidents':1,
              'source':{'id':'big_blue_1999','path':str(raw.relative_to(ROOT)),
                        'url':URL,'page':PAGE,'author':'Unnamed OSHA employee (as identified by Commons)',
                        'rights':'Commons PD-USGov / Public Domain Mark; US federal government work',
                        'rights_url':'https://creativecommons.org/publicdomain/mark/1.0/',
                        'corroborating_source':'https://obis.osha.gov/dcsp/success_stories/compliance_assistance/abbott/abbott_casestudies/slide30.html',
                        'sha256':SHA256,'sha1':SHA1,'bytes':raw.stat().st_size},
              'annotation':{'origin':'manual visual review of decoded source before model scoring',
                            'target':'visible crane instability/collapse, not pre-incident risk',
                            'source_event_start_s':55,'source_event_end_s':61,
                            'boundary_uncertainty_s':0.5,
                            'negative_means':'no visible collapse; not a certification of safety',
                            'confounds':['handheld camera motion','compression','low resolution','single annotator'],
                            'unused_source_interval_s':[0,34],
                            'edits':'contiguous temporal cuts, remove audio, CFR 30 fps H.264; no synthetic hazard or inserted transition'},
              'evaluation_settings':{'fps':8,'window':16,'stride':8,'fit':'pad','dtype':'float32','threshold_z':3},
              'samples':samples}
    (ROOT/'data/real_hazard_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'samples':[{k:s[k] for k in ('id','duration_s','split')} for s in samples],
                      'independent_incidents':1},indent=2))

if __name__=='__main__':main()
