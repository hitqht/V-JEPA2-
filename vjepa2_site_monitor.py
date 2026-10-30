#!/usr/bin/env python3
"""Mac 本地 V-JEPA 2 施工视频离线复核工具。

前半窗独立编码后预测未来 patch 表征，与完整窗口的离线目标编码比较。
残差是掩码潜表征一致性分数，不是事故概率；change 仅提供辅助线索，
不能自动判断变化是由机械、镜头还是光照引起。用法、验证和边界见 README。
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L ≈ 0.3B；更大：facebook/vjepa2-vitg-fpc64-256


# ----------------------------------------------------------------------------
# 设备与模型
# ----------------------------------------------------------------------------
def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        device = torch.device(pref)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS 不可用；请使用 --device cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(pref: str, device: torch.device) -> torch.dtype:
    if pref == "auto":
        return torch.float32  # 默认保留数值精度；可显式选择 float16 节省内存
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[pref]


def load_model(model_id: str, device: torch.device, dtype: torch.dtype, revision=None):
    """加载 V-JEPA 2（编码器 + predictor）。"""
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(model_id, revision=revision, attn_implementation="sdpa")
    except Exception:
        print(
            "\n[!] 模型下载/加载失败。检查网络与证书，或运行\n"
            "    python scripts/download_model.py --source meta\n"
            "    再指定 --model models/vjepa2-vitl；详见 README。\n",
            file=sys.stderr,
        )
        raise
    model.to(device=device, dtype=dtype).eval()
    return model


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# 视频读取与预处理
# ----------------------------------------------------------------------------
def preprocess(frame_bgr: np.ndarray, crop: int, fit: str) -> np.ndarray:
    """返回 crop×crop×3 的 uint8 RGB 图像（模型实际"看到"的画面）。

    fit="center"：近似官方预处理几何（短边缩放 + 中心裁剪），宽屏会裁掉左右两侧。
    fit="pad"   ：整幅画面等比缩放后补边成正方形，保留全部视野，适合广角工地摄像头。
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if fit == "center":
        short = int(crop * 256 / 224)
        scale = short / min(h, w)
        rgb = cv2.resize(rgb, (max(crop, round(w * scale)), max(crop, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        return np.ascontiguousarray(rgb[y0:y0 + crop, x0:x0 + crop])

    scale = crop / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    # 均值色补边归一化后接近 0；不能保证无分布偏移。
    canvas = np.empty((crop, crop, 3), dtype=np.uint8)
    canvas[:] = (IMAGENET_MEAN * 255).astype(np.uint8)
    y0, x0 = (crop - nh) // 2, (crop - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)  # (3, H, W)


def iter_sampled_frames(path: str, sample_fps: float, start: float, duration: float | None):
    """按目标帧率均匀抽帧，产出 (时间戳秒, BGR帧)。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 0:
        cap.release()
        raise ValueError("视频帧率无效，先用 ffmpeg 转为恒定帧率视频")
    if sample_fps > src_fps:
        print(f"[i] 目标抽帧率 {sample_fps} 高于视频帧率 {src_fps:.1f}，将使用全部帧。")
    start_frame = int(np.ceil(start * src_fps - 1e-9))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    next_t = start
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (start_frame + i) / src_fps
            i += 1
            if duration is not None and t >= start + duration - 1e-9:
                break
            if t + 1e-6 >= next_t:
                while next_t <= t + 1e-6:
                    next_t += 1.0 / min(sample_fps, src_fps)
                yield t, frame
    finally:
        cap.release()


def iter_windows(frames, window: int, stride: int, crop: int, fit: str):
    """滑动窗口，产出 (窗口起始时间, 窗口结束时间, 帧张量列表, 代表帧RGB)。

    代表帧取"被预测的后半段"的中间一帧，用于导出图片。
    """
    buf: deque = deque(maxlen=window)
    n = 0
    for t, frame in frames:
        rgb = preprocess(frame, crop, fit)
        buf.append((t, to_tensor(rgb), rgb))
        n += 1
        if n >= window and (n - window) % stride == 0:
            yield (buf[0][0], buf[-1][0], [b[1] for b in buf],
                   buf[(3 * window) // 4][2], buf[window // 2][0],
                   float(np.median(np.diff([b[0] for b in buf]))))


# ----------------------------------------------------------------------------
# 核心：用 predictor 预测后半段，计算意外度
# ----------------------------------------------------------------------------
@torch.no_grad()
def prediction_tensors(model, clips: torch.Tensor):
    """独立编码过去；仅离线目标编码器可见未来。返回 full/context/pred/target。"""
    B, T = clips.shape[:2]
    if T < 4 or T % (2 * model.config.tubelet_size):
        raise ValueError("时间长度必须可均分为完整 tubelet")
    full = model.encoder(pixel_values_videos=clips).last_hidden_state
    ctx = model.encoder(pixel_values_videos=clips[:, : T // 2]).last_hidden_state
    N, Nc = full.shape[1], ctx.shape[1]
    context_mask = [torch.arange(Nc, device=clips.device).expand(B, -1)]
    target_mask = [torch.arange(Nc, N, device=clips.device).expand(B, -1)]
    pred = model.predictor(encoder_hidden_states=ctx, context_mask=context_mask,
                           target_mask=target_mask).last_hidden_state
    return full, ctx, pred, full[:, Nc:]


@torch.no_grad()
def score_batch(model, clips: torch.Tensor):
    """clips: (B, T, 3, H, W)。

    返回：
      surprise (B,)          后半段预测误差（L1，表征空间）
      heat     (B, g, g)     误差在空间上的分布（g = crop / patch_size）
      emb      (B, D)        整个窗口的平均表征，用于计算窗口间变化
    """
    B, T = clips.shape[:2]
    full, ctx, pred, tgt = prediction_tensors(model, clips)

    d = pred.shape[-1]
    pred = F.layer_norm(pred.float(), (d,))
    tgt = F.layer_norm(tgt.float(), (d,))
    err = (pred - tgt).abs().mean(-1)                                           # (B, N-Nc)

    if pred.shape != tgt.shape:
        raise RuntimeError(f"预测/目标形状不一致: {pred.shape} / {tgt.shape}")
    if not torch.isfinite(err).all():
        raise RuntimeError("模型输出含 NaN/Inf，请使用 --dtype float32")
    g = clips.shape[-1] // model.config.patch_size
    heat = err.view(B, -1, g, g).mean(1)                                        # (B, g, g)
    emb = full.float().mean(1)                                                  # (B, D)
    return err.mean(-1).cpu().numpy(), heat.cpu().numpy(), emb.cpu()


# ----------------------------------------------------------------------------
# 统计与输出
# ----------------------------------------------------------------------------
def robust_stats(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-9:
        mad = float(np.std(x)) or 1.0
    return med, mad


def fmt_t(sec: float) -> str:
    """秒 → mm:ss.s（超过 1 小时为 h:mm:ss.s）。"""
    sec = max(0.0, sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):d}:{int(m):02d}:{s:04.1f}" if h >= 1 else f"{int(m):02d}:{s:04.1f}"


def target_start(r) -> float:
    """窗口中被预测的后半段的起点（意外发生在这一段里）。"""
    return r["target_start_s"] if "target_start_s" in r else (r["start_s"] + r["end_s"]) / 2


def merge_events(rows, flag_key="flagged"):
    """把连续被标记的窗口合并成事件段（时间范围取各窗口被预测的后半段）。"""
    events, cur = [], None
    for r in rows:
        end = r.get("target_end_s", r["end_s"])
        if cur is not None and (not r[flag_key] or target_start(r) > cur["end"] + 0.05):
            events.append(cur)
            cur = None
        if r[flag_key]:
            if cur is None:
                cur = {"start": target_start(r), "end": end, "peak_z": r["surprise_z"],
                       "peak_t": r["end_s"], "change_z": r["change_z"]}
            else:
                cur["end"] = end
                if r["surprise_z"] > cur["peak_z"]:
                    cur.update(peak_z=r["surprise_z"], peak_t=r["end_s"], change_z=r["change_z"])
        elif cur is not None:
            events.append(cur)
            cur = None
    if cur is not None:
        events.append(cur)
    return events


def save_overlay(path: Path, rgb: np.ndarray, heat: np.ndarray, label: str):
    """左：模型视角画面；右：意外位置的单色（红）热力叠加。"""
    crop = rgb.shape[0]
    h = cv2.resize(heat.astype(np.float32), (crop, crop), interpolation=cv2.INTER_CUBIC)
    lo = np.percentile(h, 70)  # 只高亮误差最大的约 30% 区域，避免整幅泛红
    h = np.clip((h - lo) / (h.max() - lo + 1e-8), 0, 1)
    alpha = (0.7 * h)[..., None]
    red = np.zeros_like(rgb, dtype=np.float32)
    red[:] = (208, 59, 59)  # RGB
    over = (rgb.astype(np.float32) * (1 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
    canvas = np.concatenate([rgb, np.full((crop, 6, 3), 255, np.uint8), over], axis=1)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    scale = 2 if crop <= 256 else 1
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_timeline(path: Path, rows, events, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = rows[-1]["end_s"] - rows[0]["start_s"]
    div, unit = (60.0, "minutes") if span > 180 else (1.0, "seconds")
    t = np.array([r["end_s"] for r in rows]) / div
    sz = np.array([r["surprise_z"] for r in rows])
    cz = np.array([r["change_z"] for r in rows])

    ink, muted, grid, series, flag = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#d03b3b"
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, facecolor="#fcfcfb")
    panels = [
        (axes[0], sz, "Surprise (prediction error, robust z)"),
        (axes[1], cz, "Change vs previous window (robust z)"),
    ]
    for ax, y, title in panels:
        ax.set_facecolor("#fcfcfb")
        for ev in events:
            ax.axvspan(ev["start"] / div, ev["end"] / div, color=flag, alpha=0.12, lw=0)
        ax.plot(t, y, color=series, lw=2)
        ax.axhline(threshold, color=muted, lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="y", color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
    axes[0].text(t[0], threshold, f"threshold {threshold:g}", va="bottom", ha="left",
                 fontsize=8, color=muted)
    axes[1].set_xlabel(f"Time ({unit}, end of each window)", color=muted, fontsize=9)
    fig.text(0.01, 0.005, "Shaded = flagged for review (surprise z above threshold)",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_review_video(source, path, rows, start=0.0, duration=None):
    """离线复核用：分数覆盖被预测的时间区间，不表示逐帧即时告警。"""
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, 960 / width)
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建复核视频")
    index = int(np.ceil(start * fps - 1e-9))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    try:
        while True:
            ok, frame = cap.read()
            t = index / fps
            index += 1
            if not ok or (duration is not None and t >= start + duration):
                break
            frame = cv2.resize(frame, size)
            active = [r for r in rows if target_start(r) <= t < r["target_end_s"]]
            r = max(active, key=lambda r: r["surprise_z"]) if active else None
            flag = bool(r and r["flagged"])
            color = (45, 60, 215) if flag else (60, 165, 65)
            cv2.rectangle(frame, (0, 0), (size[0], 56), (25, 25, 25), -1)
            text = (f"{fmt_t(t)} | z={r['surprise_z']:.2f} | "
                    f"{'REVIEW' if flag else 'below threshold'}") if r else f"{fmt_t(t)} | no target window"
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
            cv2.putText(frame, "Offline latent prediction residual; not a safety verdict", (12, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="V-JEPA 2 工地视频离线潜表征残差分析")
    p.add_argument("video", help="视频文件路径（mp4/mov/avi 等）")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face 模型名（默认 {DEFAULT_MODEL}）")
    p.add_argument("--revision", default=None, help="Hugging Face commit SHA；本地目录无需指定")
    p.add_argument("--out", default=None, help="输出目录（默认 <视频名>_vjepa2）")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已有结果并清理旧 Top-K 图像")
    p.add_argument("--fps", type=float, default=8.0, help="抽帧率（默认 8）")
    p.add_argument("--window", type=int, default=16, help="每个窗口帧数，须为 4 的倍数（默认 16）")
    p.add_argument("--stride", type=int, default=8, help="窗口步长帧数（默认 8）")
    p.add_argument("--batch", type=int, default=1, help="批大小（默认 1）")
    p.add_argument("--annotated-video", action="store_true", help="导出原视野的复核视频 review.mp4")
    p.add_argument("--fit", choices=["center", "pad"], default="center",
                   help="center=近似官方中心裁剪；pad=保留全视野的实验性补边")
    p.add_argument("--threshold", type=float, default=3.0, help="surprise z 分数告警阈值（默认 3）")
    p.add_argument("--top-k", type=int, default=5, help="导出最意外时刻的数量（默认 5）")
    p.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    p.add_argument("--duration", type=float, default=None, help="只分析多少秒")
    p.add_argument("--baseline-csv", default=None,
                   help="用另一段'正常'视频的 scores.csv 作为 z 分数基线（同一机位效果最好）")
    p.add_argument("--device", default="auto", help="auto / mps / cpu / cuda")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = p.parse_args(argv)
    if args.window % 4 != 0 or args.window < 4:
        p.error("--window 必须是 4 的倍数（tubelet=2，且前后两半各需整数个 tubelet）")
    if args.stride < 1:
        p.error("--stride 必须 ≥ 1")
    for key in ("fps", "threshold"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key} 必须为有限正数")
    if args.batch < 1 or args.top_k < 0 or not np.isfinite(args.start) or args.start < 0:
        p.error("batch ≥ 1，top-k ≥ 0，start 为有限非负数")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        p.error("duration 必须为有限正数")
    if not Path(args.video).is_file():
        p.error(f"找不到视频：{args.video}")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out or f"{Path(args.video).stem}_vjepa2")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise ValueError(f"输出目录非空：{out}；请另选目录或传入 --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in out.glob("top_*.jpg"):
            path.unlink()
        if (out / "review.mp4").exists():
            (out / "review.mp4").unlink()

    print(f"[i] 设备 {device} · 精度 {str(dtype).replace('torch.', '')} · 模型 {args.model}")
    model = load_model(args.model, device, dtype, args.revision)
    crop = model.config.crop_size
    max_frames = model.config.frames_per_clip
    if args.window > max_frames:
        raise SystemExit(f"--window 不能超过模型的 frames_per_clip={max_frames}")

    frames = iter_sampled_frames(args.video, args.fps, args.start, args.duration)
    windows = iter_windows(frames, args.window, args.stride, crop, args.fit)

    if args.window % (2 * model.config.tubelet_size):
        raise ValueError("窗口须能均分为整数个 tubelet")
    rows, embs, heats = [], [], []
    cap = max(args.top_k * 6, 12)
    top_heap: list = []   # (surprise, idx, rgb, heat)，只保留最可能入选的少数窗口，内存恒定
    t0 = time.time()

    def flush(batch):
        clips = torch.stack([torch.stack(b[2]) for b in batch]).to(device=device, dtype=dtype)
        s, heat, emb = score_batch(model, clips)
        for j, b in enumerate(batch):
            idx = len(rows)
            rows.append({"window": idx, "start_s": round(b[0], 3), "end_s": round(b[1], 3),
                         "target_start_s": round(b[4], 3),
                         "target_end_s": round(b[1] + b[5], 3),
                         "surprise": float(s[j])})
            embs.append(emb[j])
            heats.append(heat[j])
            item = (float(s[j]), idx, b[3], heat[j])
            if len(top_heap) < cap:
                heapq.heappush(top_heap, item)
            elif item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)

    batch = []
    for w in windows:
        batch.append(w)
        if len(batch) == args.batch:
            flush(batch)
            batch = []
            print(f"    {len(rows)} 窗口 · 至 {fmt_t(rows[-1]['end_s'])}", flush=True)
            if len(rows) % (args.batch * 10) == 0:
                el = time.time() - t0
                print(f"    已处理至 {fmt_t(rows[-1]['end_s'])} · {len(rows)} 个窗口 · "
                      f"{len(rows) / el:.2f} 窗口/秒")
    if batch:
        flush(batch)

    if not rows:
        raise SystemExit("视频太短：至少需要 window/fps 秒的画面。")

    # 相邻窗口表征变化（余弦距离）
    E = F.normalize(torch.stack(embs), dim=-1)
    change = np.zeros(len(rows), dtype=np.float32)
    if len(rows) > 1:
        change[1:] = (1 - (E[1:] * E[:-1]).sum(-1)).numpy()
    for r, c in zip(rows, change):
        r["change"] = float(c)

    # 记录本次运行的关键设置；用作基线时，这些设置必须一致才有可比性
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    manifest = Path(args.model) / "download_manifest.json"
    if manifest.exists():
        revision = json.loads(manifest.read_text())["revision"]
    run_cfg = {"model": args.model, "revision": revision, "fps": args.fps, "window": args.window,
               "stride": args.stride, "fit": args.fit, "crop_size": crop,
               "dtype": str(dtype), "method": "past-only-encoder-l1-v1"}

    # z 分数：默认以本视频自身为基线；也可指定"正常日"基线
    if args.baseline_csv:
        base_cfg_path = Path(args.baseline_csv).with_name("run_config.json")
        if base_cfg_path.exists():
            base_cfg = json.loads(base_cfg_path.read_text())
            diff = {k: (base_cfg.get(k), v) for k, v in run_cfg.items() if base_cfg.get(k) != v}
            if diff:
                raise ValueError(f"基线与本次设置不一致（基线值, 本次值）：{diff}")
        else:
            raise ValueError("基线必须带有同目录的 run_config.json")
        with open(args.baseline_csv, newline="") as f:
            base = list(csv.DictReader(f))
        if len(base) < 3:
            raise ValueError("基线至少需 3 个有效窗口；建议使用较长的正常同机位视频")
        if not all(np.isfinite(float(b[k])) for b in base for k in ("surprise", "change")):
            raise ValueError("基线分数包含 NaN/Inf")
        s_med, s_mad = robust_stats(np.array([float(b["surprise"]) for b in base]))
        c_med, c_mad = robust_stats(np.array([float(b["change"]) for b in base[1:]]))
        print(f"[i] 使用基线 {args.baseline_csv}（{len(base)} 个窗口）")
    else:
        s_med, s_mad = robust_stats(np.array([r["surprise"] for r in rows]))
        c_med, c_mad = robust_stats(change[1:] if len(rows) > 1 else change)
    for r in rows:
        r["surprise_z"] = round((r["surprise"] - s_med) / s_mad, 3)
        r["change_z"] = round((r["change"] - c_med) / c_mad, 3) if r["window"] > 0 else 0.0
        r["flagged"] = int(r["surprise_z"] >= args.threshold)

    # 写 CSV
    fields = ["window", "start_s", "end_s", "target_start_s", "target_end_s", "surprise", "surprise_z", "change", "change_z", "flagged"]
    with open(out / "scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in fields})

    events = merge_events(rows)
    (out / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2))
    (out / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))
    np.savez_compressed(out / "features.npz", embeddings=torch.stack(embs).numpy(),
                        heatmaps=np.stack(heats))
    plot_timeline(out / "timeline.png", rows, events, args.threshold)

    # 导出 Top-K（相互不重叠的窗口）
    picked = []
    min_gap = max(1, args.window // args.stride)
    for s, idx, rgb, heat in (sorted(top_heap, key=lambda x: -x[0]) if args.top_k else []):
        if all(abs(idx - p[1]) >= min_gap for p in picked):
            picked.append((s, idx, rgb, heat))
        if len(picked) >= args.top_k:
            break
    for rank, (s, idx, rgb, heat) in enumerate(picked, 1):
        r = rows[idx]
        label = (f"#{rank}  {fmt_t(r['start_s'])}-{fmt_t(r['end_s'])}  "
                 f"surprise z={r['surprise_z']:.1f}  change z={r['change_z']:.1f}")
        save_overlay(out / f"top_{rank:02d}_{fmt_t(r['end_s']).replace(':', '-')}.jpg", rgb, heat, label)

    # 总结
    el = time.time() - t0
    import transformers
    summary = {"video": str(Path(args.video).resolve()), "video_sha256": sha256_file(args.video),
               "device": str(device), "dtype": str(dtype), "windows": len(rows),
               "analysis_seconds": el, "windows_per_second": len(rows) / el,
               "events": len(events), "threshold": args.threshold,
               "calibration": "external" if args.baseline_csv else "self-descriptive",
               "baseline_csv": args.baseline_csv,
               "statistics": {"surprise_median": s_med, "surprise_scale": s_mad,
                              "change_median": c_med, "change_scale": c_mad},
               "python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "platform": platform.platform()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.annotated_video:
        write_review_video(args.video, out / "review.mp4", rows, args.start, args.duration)
    print(f"\n[✓] 完成：{len(rows)} 个窗口，耗时 {el:.0f} 秒，输出目录 {out}/")
    if events:
        print(f"[!] {len(events)} 段超过阈值 z≥{args.threshold:g}，建议人工复核：")
        for ev in sorted(events, key=lambda e: -e["peak_z"])[:20]:
            hint = "（整体画面也在变，可能是镜头/光线）" if ev["change_z"] >= args.threshold else ""
            print(f"    {fmt_t(ev['start'])} – {fmt_t(ev['end'])}  峰值 z={ev['peak_z']:.1f} "
                  f"@ {fmt_t(ev['peak_t'])}{hint}")
    else:
        print(f"[i] 没有窗口超过阈值 z≥{args.threshold:g}。可用 --threshold 调低，或看 timeline.png 的相对高点。")
    return rows, events


if __name__ == "__main__":
    main()
