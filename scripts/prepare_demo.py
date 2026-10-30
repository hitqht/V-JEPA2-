#!/usr/bin/env python3
"""下载带来源记录的视频，制作 8 fps 小样本和可视复核图。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vjepa2_site_monitor import sha256_file

URL = 'https://upload.wikimedia.org/wikipedia/commons/c/cb/Excavator_digging_land_and_loading_truck.webm'
PAGE = 'https://commons.wikimedia.org/wiki/File:Excavator_digging_land_and_loading_truck.webm'
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-download', action='store_true')
    args = p.parse_args()
    raw = ROOT / 'data/raw/excavator.webm'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        if args.skip_download:
            raise SystemExit('缺少 data/raw/excavator.webm')
        part = raw.with_suffix('.webm.part')
        subprocess.run(['curl','-fL','--retry','2',URL,'-o',str(part)], check=True)
        part.replace(raw)
    expected = '40b48f4a0a29f58416023ecd32453060b361b707c62c782887845217c99c4d8f'
    actual = sha256_file(raw)
    if actual != expected:
        raise SystemExit('源视频 SHA256 不符，请核查来源版本')
    derived = ROOT / 'data/derived'
    derived.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-an','-vf',
                    'fps=8,scale=640:-2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                    str(derived/'excavator_8fps.mp4')], check=True)
    # 8 fps 文件是便于播放的小副本；定量实验始终直接解码原始文件。
    info = {'title':'Excavator digging land and loading truck','author':'KEmel49',
            'source_page':PAGE,'download_url':URL,'license':'CC BY-SA 4.0',
            'license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
            'raw_sha256':actual,'raw_bytes':raw.stat().st_size,
            'derived_sha256':sha256_file(derived/'excavator_8fps.mp4'),
            'changes':'derived MP4: no audio, 8 fps, width 640, H.264; contact sheet: resized timestamped frames',
            'retrieved_date':'2026-10-30'}
    (ROOT/'data/source_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
