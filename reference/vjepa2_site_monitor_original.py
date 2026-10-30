#!/usr/bin/env python3
"""
V-JEPA 2 工地视频"意外度"分析 —— Apple Silicon (Mac) 入门脚本
=================================================================

思路（这正是 V-JEPA 2 作为"世界模型"的用法）：
  1. 把视频切成重叠的短窗口（默认每窗 16 帧 ≈ 2 秒，步长 1 秒）。
  2. 只给模型看窗口的前半段，让 V-JEPA 2 的 predictor 在表征空间里"想象"后半段。
  3. 把想象结果和后半段的真实表征对比，误差越大 = 越出乎模型意料（surprise）。
  4. 同时计算相邻窗口的整体表征变化（change），用于区分"场景整体变了"
     （镜头移动、光线突变）和"局部发生了意外"（物料坠落、机械异常动作）。
  5. 用中位数/MAD 做稳健 z 分数，超过阈值的时间段标记为"值得人工复核"。

输出（默认写到 <视频名>_vjepa2/ 目录）：
  scores.csv      每个窗口的 surprise / change 原始值与 z 分数
  timeline.png    两条时间曲线 + 被标记的时间段
  top_XX_*.jpg    最"意外"的若干时刻：模型视角画面 | 意外位置热力叠加

安装（建议在虚拟环境里）：
  pip install torch transformers opencv-python matplotlib numpy

用法：
  python vjepa2_site_monitor.py site.mp4
  python vjepa2_site_monitor.py site.mp4 --fit pad --start 600 --duration 300
  # 先用"正常一天"的视频跑一次，再以它为基线分析别的视频：
  python vjepa2_site_monitor.py normal_day.mp4 --out baseline_run
  python vjepa2_site_monitor.py today.mp4 --baseline-csv baseline_run/scores.csv

国内下载模型较慢时，可先设置镜像：
  export HF_ENDPOINT=https://hf-mirror.com

重要提示：surprise 高只代表"与模型预期不符"，不等于安全违规。它适合用来
从几小时录像里筛出值得人看的片段；判断是否违规仍需人工或后续的 VLM（如 Cosmos Reason）。
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L ≈ 0.3B；更大：facebook/vjepa2-vitg-fpc64-256


# ----------------------------------------------------------------------------
# 设备与模型
# ----------------------------------------------------------------------------
def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(pref: str, device: torch.device) -> torch.dtype:
    if pref == "auto":
        return torch.float32 if device.type == "cpu" else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[pref]


def load_model(model_id: str, device: torch.device, dtype: torch.dtype):
    """加载 V-JEPA 2（编码器 + predictor）。"""
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(model_id)
    except Exception:
        print(
            "\n[!] 模型下载/加载失败。若在国内网络环境，可尝试：\n"
            "    export HF_ENDPOINT=https://hf-mirror.com\n"
            "    然后重新运行本脚本。\n",
            file=sys.stderr,
        )
        raise
    model.to(device=device, dtype=dtype).eval()
    return model


# ----------------------------------------------------------------------------
# 视频读取与预处理
# ----------------------------------------------------------------------------
def preprocess(frame_bgr: np.ndarray, crop: int, fit: str) -> np.ndarray:
    """返回 crop×crop×3 的 uint8 RGB 图像（模型实际"看到"的画面）。

    fit="center"：与 V-JEPA 2 训练方式一致（短边缩放 + 中心裁剪），宽屏监控会裁掉左右两侧。
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
    # 用 ImageNet 均值色补边，归一化后接近 0，对模型干扰最小
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
    if not src_fps or src_fps != src_fps or src_fps > 240:  # 0 / NaN / 异常值
        src_fps = 25.0
    if sample_fps > src_fps:
        print(f"[i] 目标抽帧率 {sample_fps} 高于视频帧率 {src_fps:.1f}，将使用全部帧。")
    start_frame = int(round(start * src_fps))
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
            if duration is not None and t - start > duration:
                break
            if t + 1e-6 >= next_t:
                next_t += 1.0 / sample_fps
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
            yield buf[0][0], buf[-1][0], [b[1] for b in buf], buf[(3 * window) // 4][2]


# ----------------------------------------------------------------------------
# 核心：用 predictor 预测后半段，计算意外度
# ----------------------------------------------------------------------------
@torch.no_grad()
def score_batch(model, clips: torch.Tensor):
    """clips: (B, T, 3, H, W)。

    返回：
      surprise (B,)          后半段预测误差（L1，表征空间）
      heat     (B, g, g)     误差在空间上的分布（g = crop / patch_size）
      emb      (B, D)        整个窗口的平均表征，用于计算窗口间变化
    """
    B, T = clips.shape[:2]
    dev = clips.device

    # 目标：编码器看完整窗口，取后半段 token 作为"真实未来"
    full = model.encoder(pixel_values_videos=clips).last_hidden_state          # (B, N, D)
    # 上下文：编码器只看前半段，保证 predictor 拿不到未来信息
    ctx = model.encoder(pixel_values_videos=clips[:, : T // 2]).last_hidden_state  # (B, Nc, D)

    N, Nc = full.shape[1], ctx.shape[1]
    # token 按 时间→行→列 排列，前 Nc 个正好对应前半段
    context_mask = [torch.arange(Nc, device=dev).unsqueeze(0).repeat(B, 1)]
    target_mask = [torch.arange(Nc, N, device=dev).unsqueeze(0).repeat(B, 1)]

    pred = model.predictor(
        encoder_hidden_states=ctx, context_mask=context_mask, target_mask=target_mask
    ).last_hidden_state                                                         # (B, N-Nc, D)
    tgt = full[:, Nc:]

    d = pred.shape[-1]
    pred = F.layer_norm(pred.float(), (d,))
    tgt = F.layer_norm(tgt.float(), (d,))
    err = (pred - tgt).abs().mean(-1)                                           # (B, N-Nc)

    g = model.config.crop_size // model.config.patch_size
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
    return (r["start_s"] + r["end_s"]) / 2


def merge_events(rows, flag_key="flagged"):
    """把连续被标记的窗口合并成事件段（时间范围取各窗口被预测的后半段）。"""
    events, cur = [], None
    for r in rows:
        if r[flag_key]:
            if cur is None:
                cur = {"start": target_start(r), "end": r["end_s"], "peak_z": r["surprise_z"],
                       "peak_t": r["end_s"], "change_z": r["change_z"]}
            else:
                cur["end"] = r["end_s"]
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


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="用 V-JEPA 2 在 Mac 上分析工地视频的意外事件")
    p.add_argument("video", help="视频文件路径（mp4/mov/avi 等）")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face 模型名（默认 {DEFAULT_MODEL}）")
    p.add_argument("--out", default=None, help="输出目录（默认 <视频名>_vjepa2）")
    p.add_argument("--fps", type=float, default=8.0, help="抽帧率（默认 8）")
    p.add_argument("--window", type=int, default=16, help="每个窗口帧数，须为 4 的倍数（默认 16）")
    p.add_argument("--stride", type=int, default=8, help="窗口步长帧数（默认 8）")
    p.add_argument("--batch", type=int, default=4, help="批大小（默认 4；内存紧张时调小）")
    p.add_argument("--fit", choices=["center", "pad"], default="center",
                   help="center=中心裁剪（与训练一致）；pad=保留整幅画面（广角监控推荐）")
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
    return args


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    out = Path(args.out or f"{Path(args.video).stem}_vjepa2")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[i] 设备 {device} · 精度 {str(dtype).replace('torch.', '')} · 模型 {args.model}")
    model = load_model(args.model, device, dtype)
    crop = model.config.crop_size
    max_frames = model.config.frames_per_clip
    if args.window > max_frames:
        raise SystemExit(f"--window 不能超过模型的 frames_per_clip={max_frames}")

    frames = iter_sampled_frames(args.video, args.fps, args.start, args.duration)
    windows = iter_windows(frames, args.window, args.stride, crop, args.fit)

    rows, embs = [], []
    cap = max(args.top_k * 6, 12)
    top_heap: list = []   # (surprise, idx, rgb, heat)，只保留最可能入选的少数窗口，内存恒定
    t0 = time.time()

    def flush(batch):
        clips = torch.stack([torch.stack(b[2]) for b in batch]).to(device=device, dtype=dtype)
        s, heat, emb = score_batch(model, clips)
        for j, b in enumerate(batch):
            idx = len(rows)
            rows.append({"window": idx, "start_s": round(b[0], 3), "end_s": round(b[1], 3),
                         "surprise": float(s[j])})
            embs.append(emb[j])
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
    run_cfg = {"model": args.model, "fps": args.fps, "window": args.window,
               "stride": args.stride, "fit": args.fit}
    with open(out / "run_config.json", "w") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    # z 分数：默认以本视频自身为基线；也可指定"正常日"基线
    if args.baseline_csv:
        base_cfg_path = Path(args.baseline_csv).with_name("run_config.json")
        if base_cfg_path.exists():
            base_cfg = json.loads(base_cfg_path.read_text())
            diff = {k: (base_cfg.get(k), v) for k, v in run_cfg.items() if base_cfg.get(k) != v}
            if diff:
                print("[!] 基线与本次设置不一致，z 分数不可比（基线值, 本次值）：", diff)
        with open(args.baseline_csv, newline="") as f:
            base = list(csv.DictReader(f))
        s_med, s_mad = robust_stats(np.array([float(b["surprise"]) for b in base]))
        c_med, c_mad = robust_stats(np.array([float(b["change"]) for b in base]))
        print(f"[i] 使用基线 {args.baseline_csv}（{len(base)} 个窗口）")
    else:
        s_med, s_mad = robust_stats(np.array([r["surprise"] for r in rows]))
        c_med, c_mad = robust_stats(change[1:] if len(rows) > 1 else change)
    for r in rows:
        r["surprise_z"] = round((r["surprise"] - s_med) / s_mad, 3)
        r["change_z"] = round((r["change"] - c_med) / c_mad, 3) if r["window"] > 0 else 0.0
        r["flagged"] = int(r["surprise_z"] >= args.threshold)

    # 写 CSV
    fields = ["window", "start_s", "end_s", "surprise", "surprise_z", "change", "change_z", "flagged"]
    with open(out / "scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in fields})

    events = merge_events(rows)
    plot_timeline(out / "timeline.png", rows, events, args.threshold)

    # 导出 Top-K（相互不重叠的窗口）
    picked = []
    min_gap = max(1, args.window // args.stride)
    for s, idx, rgb, heat in sorted(top_heap, key=lambda x: -x[0]):
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
