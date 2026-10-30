#!/usr/bin/env python3
"""下载官方权重；auto 优先 HF，网络/服务错误时回退 Meta 官方检查点。"""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vjepa2_site_monitor import sha256_file

META_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt'
META_ETAG = '71cf6756d55670a39bb369fca4716a54-612'
META_BYTES = 5127726842
META_SHA256 = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'


def rename(key, prefix):
    key = key.removeprefix('module.backbone.')
    key = key.replace('predictor_blocks.', 'layer.').replace('blocks.', 'layer.')
    key = key.replace('patch_embed.', 'embeddings.patch_embeddings.')
    key = key.replace('predictor_embed.', 'embeddings.predictor_embeddings.')
    key = key.replace('.attn.qkv.', '.attention.qkv.')
    key = key.replace('.attn.proj.', '.attention.proj.')
    if key.startswith('predictor_norm.'):
        key = key.replace('predictor_norm.', 'layernorm.', 1)
    elif key.startswith('norm.'):
        key = key.replace('norm.', 'layernorm.', 1)
    key = key.replace('predictor_proj.', 'proj.')
    return prefix + '.' + key


def convert_meta(checkpoint, destination):
    import torch
    from transformers import VJEPA2Config, VJEPA2Model
    with zipfile.ZipFile(checkpoint) as archive:
        invalid = archive.testzip()
        if invalid is not None:
            raise ValueError(f"检查点 ZIP CRC 校验失败: {invalid}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != META_SHA256:
        raise ValueError('官方检查点 SHA256 与本次验证版本不符')
    # 官方 checkpoint，仍使用 weights_only 以禁止执行任意 pickle 类。
    original = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = {}
    for group, prefix in [('target_encoder','encoder'),('predictor','predictor')]:
        for key, value in original[group].items():
            if 'mask_tokens.' in key:
                continue
            key=rename(key, prefix)
            if '.attention.qkv.' in key:
                for name,part in zip(('query','key','value'),value.chunk(3,dim=0)):
                    state[key.replace('.qkv.',f'.{name}.')] = part.contiguous()
            else:
                state[key] = value
    masks = original['predictor']
    state['predictor.embeddings.mask_tokens'] = torch.cat(
        [masks[f'module.backbone.mask_tokens.{i}'] for i in range(10)], dim=0).unsqueeze(1)
    config = VJEPA2Config()
    with torch.device('meta'):
        model = VJEPA2Model(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.save_pretrained(destination, safe_serialization=True)
    metadata = {'source':'Meta official checkpoint','url':META_URL,
                'checkpoint_sha256':checkpoint_sha256,'checkpoint_bytes':checkpoint.stat().st_size,
                'source_etag':META_ETAG,'revision':'meta-vitl-'+checkpoint_sha256[:16],
                'zip_crc_verified':True,
                'encoder_key':'target_encoder','predictor_key':'predictor','strict_load':True,
                'transformers':'4.57.1','converted_weights_sha256':sha256_file(destination/'model.safetensors')}
    (destination/'download_manifest.json').write_text(json.dumps(metadata,indent=2))
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/model_manifest.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


def download_hf(destination):
    from huggingface_hub import HfApi, snapshot_download
    repo = 'facebook/vjepa2-vitl-fpc64-256'
    revision = HfApi().model_info(repo).sha
    snapshot_download(repo,revision=revision,local_dir=destination,
                      allow_patterns=['config.json','model.safetensors','preprocessor_config.json'])
    metadata={'source':repo,'revision':revision,
              'converted_weights_sha256':sha256_file(destination/'model.safetensors')}
    (destination/'download_manifest.json').write_text(json.dumps(metadata,indent=2))
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/model_manifest.json').write_text(json.dumps(metadata,indent=2))


def download_meta(checkpoint,destination):
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists():
        part=checkpoint.with_suffix('.pt.part')
        subprocess.run(['curl','-fL','--retry','2','-C','-',META_URL,'-o',str(part)],check=True)
        if part.stat().st_size != META_BYTES:
            raise SystemExit('官方检查点大小不符，保留 .part 便于续传')
        part.replace(checkpoint)
    if checkpoint.stat().st_size != META_BYTES:
        raise SystemExit('官方检查点大小不符')
    convert_meta(checkpoint,destination)


def reuse_verified(destination):
    manifest=destination/'download_manifest.json'
    weights=destination/'model.safetensors'
    if not (manifest.is_file() and weights.is_file() and (destination/'config.json').is_file()):
        return False
    metadata=json.loads(manifest.read_text())
    expected=metadata.get('converted_weights_sha256')
    if not expected or sha256_file(weights)!=expected:
        raise ValueError('已有模型哈希不符；请核查文件或选择新的 --out 目录')
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/model_manifest.json').write_text(json.dumps(metadata,indent=2))
    print(f'已验证并复用本地模型：{destination}',flush=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['auto','hf','meta'],default='auto')
    parser.add_argument('--out',type=Path,default=ROOT/'models/vjepa2-vitl')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/vitl.pt')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.source=='auto' and reuse_verified(args.out):
        return
    if args.source=='meta':
        download_meta(args.checkpoint,args.out)
        return
    if args.source=='hf':
        download_hf(args.out)
        return
    from requests import RequestException
    from huggingface_hub.errors import HfHubHTTPError,LocalEntryNotFoundError,FileMetadataError
    try:
        download_hf(args.out)
    except (RequestException,HfHubHTTPError,LocalEntryNotFoundError,FileMetadataError) as error:
        print(f'HF 下载不可用（{type(error).__name__}），改用 Meta 官方检查点。',flush=True)
        download_meta(args.checkpoint,args.out)

if __name__ == '__main__':
    main()
