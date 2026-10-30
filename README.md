# V-JEPA 2 工地视频分析 · Mac 本地实验

基于用户提供的 `vjepa2_site_monitor.py`，实现可运行、可复核的施工视频分析和验证。使用真实预训练 V-JEPA 2 ViT-L 编码器及 predictor；视频解码、推理和结果生成都在本机进行，无需云端推理服务或 API 密钥。现已加入**真实施工倒塌危险正样本**，此前正常作业和合成扰动实验作为补充保留。

**用途是离线筛选值得复核的片段。** 输出潜表征预测残差、相邻窗口表征变化、时间曲线、Top-K 热图、事件 JSON、特征和复核视频。模型没有施工安全分类头，不能直接识别“违规”“碰撞”“人员危险接近”。真实倒塌案例已实测，但只有一个独立事故，检出和误报结果仍不足以支持可靠事故识别或事故前预警。

## 快速开始

仓库包含完整分析代码、数据下载/切片、模型下载/转换、测试、实验报告和使用文档。**不包含模型权重、原始/派生视频、虚拟环境或缓存**；首次运行时下载到本机。

```bash
git clone https://github.com/hitqht/V-JEPA2-.git
cd V-JEPA2-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

# Mac：若未安装 ffmpeg，请先执行 brew install ffmpeg
python scripts/download_model.py --source auto
bash scripts/run_real_hazard.sh
```

使用 Python 3.12，首次安装及下载建议预留约 10 GB 空间。默认优先 Hugging Face，遇到下载网络/服务错误后回退到 Meta 官方检查点并校验。若已知 HF 无法连接，可直接使用 `python scripts/download_model.py --source meta`。

运行后打开 `outputs/big_blue_positive/review_with_truth.mp4` 查看人工真值与分数对照；完整结果见 [真实危险实验报告](reports/REAL_HAZARD_VALIDATION.md)。发布前实测结果随仓库保留，自己的重跑结果会重新生成。

## 1. 当前实验与交付物

**主实验：真实危险正样本。** 选用 [Big Blue 起重机施工倒塌录像](https://commons.wikimedia.org/wiki/File:Big_Blue_crane_collapse_at_Miller_Park.webm)，源文件 SHA1 与公开记录核对一致。Commons 将其标记为 OSHA 雇员履职拍摄的美国联邦政府公共领域作品。已保留约 71.67 秒的完整源片，并制作以下互不交叠短片：

| 本地样本 | 时长 | 原片区间 | 用途 |
|---|---:|---|---|
| `data/derived/big_blue_calibration.mp4` | 8 秒 | 34–42 秒 | 事前无可见倒塌，独立校准 |
| `data/derived/big_blue_negative.mp4` | 8 秒 | 42–50 秒 | 无可见倒塌的测试对照，含运镜 |
| `data/derived/big_blue_positive.mp4` | 16 秒 | 50–66 秒 | 包含真实起重机失稳/倒塌的正片 |

上述视频不提交 Git；运行 `python scripts/prepare_real_hazard.py` 或主实验脚本后生成。

正片的 **5–11 秒** 为人工标注的可见失稳/倒塌区间；其余时间分别为事前上下文、边界过渡和事后画面。不会把整段 16 秒都标为正类。边界约 ±0.5 秒不确定性。“无可见倒塌”也不等于工况安全。

- [真实危险验证报告](reports/REAL_HAZARD_VALIDATION.md)：主实验设置、全部指标和局限。
- [样本及时间真值 manifest](data/real_hazard_manifest.json)、[预先固定方案](docs/real_hazard_protocol.md)、[逐段画面](reports/real_hazard/samples_contact_sheet.jpg)。
- `outputs/big_blue_positive/`、`outputs/big_blue_negative/`：CSV、热图、事件和复核视频；`reports/real_hazard/`：真实样本指标及对照曲线。
- 危险片段真值对照视频 `outputs/big_blue_positive/review_with_truth.mp4`：运行后生成，放大到 960 像素宽以便阅读文字，不增加原片图像信息。显示当前人工标签，以及最后一张输入帧已经到达的最近完整窗口分数；不会把未来窗口的离线分数提前显示成预警。

**补充实验：正常施工与合成扰动。** 以下原有素材和结果不变：

- 原视频：约 **17.67 秒**、1920×1080、30 fps，固定机位可见挖掘、提斗、回转、向卡车卸料。
- 来源：[Wikimedia Commons / KEmel49](https://commons.wikimedia.org/wiki/File:Excavator_digging_land_and_loading_truck.webm)，[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。已下载到 `data/raw/excavator.webm`，SHA256 和署名见 `data/source_manifest.json`、[素材署名](docs/ATTRIBUTION.md)。
- `data/derived/excavator_8fps.mp4` 是便于观看的小副本；**定量分析直接从源 WebM 抽帧**，避免再次压缩影响比较。
- `reports/source_contact_sheet.jpg`：逐秒画面，`data/annotations.json`：人工目视的近似动作阶段，未把这些阶段标作事故。
- `outputs/excavator/`：完整源视频的离线分析结果。
- `reports/validation/`：真实预训练模型的校准/测试分数、逐类指标、扰动示例和基线对比图。
- [合成扰动验证报告](reports/VALIDATION.md)：补充实验的实测结论与局限。

## 2. 环境安装

本机环境：Apple M4 Max（40 核 GPU）、64 GB 内存，macOS 26.1、Python 3.12.13。其他机器的速度需实际测量。Intel Mac 可使用 CPU；本项目没有 CUDA 运行验证。

先克隆仓库并在项目目录打开终端：

```bash
git clone https://github.com/hitqht/V-JEPA2-.git
cd V-JEPA2-
python3 -m venv .venv
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -m pip install -r requirements-lock.txt
```

`requirements.txt` 固定直接依赖；`requirements-lock.txt` 记录本次实际安装的全部版本。Python 建议固定为 3.12。也可使用 `UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python -r requirements-lock.txt`。

素材准备脚本另外需要系统 `ffmpeg`（本机已安装；没有时可用 `brew install ffmpeg`）。主分析使用 OpenCV，无需 torchvision、torchcodec 或 Decord。

确认 GPU：

```bash
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

应输出 `True`。在限制 GPU 访问的执行沙箱中可能输出 `False`；应在本机普通终端执行，或使用具有 Metal 访问权限的运行环境。`--device auto` 会选 MPS，再选 CUDA，最后 CPU；`--device mps` 不可用时会报错，避免悄悄变成 CPU。

## 3. 下载实验视频与模型

```bash
python scripts/prepare_demo.py
python scripts/prepare_real_hazard.py
python scripts/download_model.py --source auto
```

视频下载会验证预先固定的源文件 SHA256。`auto` 默认先使用 HF，遇到网络/服务错误后回退 Meta；`--source hf` 或 `--source meta` 可明确指定来源。HF 路径将仓库当前 commit 固定到 manifest，下载官方 `facebook/vjepa2-vitl-fpc64-256` 到 `models/vjepa2-vitl/`。再次执行 auto 时会校验已有模型的哈希，匹配则复用。

本次环境访问 Hugging Face 发生 TLS/域名错误，镜像也重定向到该域名，故实测使用 **Meta 官方检查点备用路径**：

```bash
python scripts/download_model.py --source meta
```

该路径从 `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt` 下载约 **5.13 GB** 的完整训练检查点，可通过 `.part` 续传；用 `torch.load(weights_only=True)` 读取，取 `target_encoder` 与 `predictor`，拆分 Q/K/V，合并 mask tokens，严格检查全部参数名称与形状，再存为约 1.3 GB 的 safetensors。本机实验所用来源、SHA256、转换标识见 `reports/model_manifest.json`。备用路径对应官方 ViT-L 架构和参数，但本项目没有额外进行与 HF 发布文件逐位相等的验证。

不要关闭 TLS 证书校验。模型下载失败时先解决网络，或用上述官方备用路径。首次安装及下载应预留约 10 GB 空间；大模型、缓存和视频均已列入 `.gitignore`。

## 4. 一键复现

安装依赖之后：

```bash
# 自动选择可用下载源
bash scripts/run_demo.sh

# 需要 Meta 备用下载时
MODEL_SOURCE=meta bash scripts/run_demo.sh
```

已有本地模型时不会重复下载。脚本会执行原有正常作业/合成扰动实验，再执行新增真实危险实验；重跑会覆盖相应结果。只复现新增主实验可运行：

```bash
bash scripts/run_real_hazard.sh
```

该脚本准备三个短片，用现有主程序分别分析校准、负片段和危险正片，再运行固定真值评估。模型、打分方法和 z≥3 阈值沿用之前设置。真实危险结果见 `reports/REAL_HAZARD_VALIDATION.md`，原有合成实验见 `reports/VALIDATION.md`。

单独分析：

```bash
python vjepa2_site_monitor.py data/raw/excavator.webm \
  --model models/vjepa2-vitl --device mps --dtype float32 \
  --fit pad --fps 8 --window 16 --stride 8 --batch 1 \
  --annotated-video --out outputs/excavator --overwrite
```

分析自己的视频：

```bash
python vjepa2_site_monitor.py /absolute/path/site.mp4 \
  --model models/vjepa2-vitl --fit pad --start 600 --duration 120 \
  --out outputs/site_10min --annotated-video
```

CFR（恒定帧率）视频优先。本实现根据解码帧编号/容器帧率给时间戳，**不保证 VFR 或损坏时间基视频的逐帧时间精度**；遇到此类文件，先转码：

```bash
ffmpeg -i input.mov -vf fps=30 -an -c:v libx264 -crf 18 -pix_fmt yuv420p input_cfr.mp4
```

## 5. 分析方法

1. 以 8 fps 抽帧，组成 16 帧（约 2 秒）窗口，每 8 帧移动一次。保留源视频时间戳；不足一窗的尾段丢弃，不填充伪造帧。
2. 将前 8 帧**单独送入编码器**获得上下文。以完整 16 帧编码得到离线目标，按时间—行—列索引选后半窗 token。
3. predictor 接收上下文 token 和未来位置索引，输出未来表征。索引提供时间位置，不含未来像素或未来编码。
4. 对预测和目标逐 token 做 LayerNorm，再对特征维度求 L1，平均为 `surprise`。按时间聚合每个 patch 的误差得到 16×16 热图。
5. 完整窗口平均特征的相邻余弦距离为 `change`。它只是变化辅助量，不能自动区分相机运动、光照与局部机械运动。
6. `z = (score - median) / (1.4826 × MAD)`。MAD 退化时用标准差；完全恒定时用 1 防止除零。`surprise_z ≥ 3` 标记复核。z 是稳健标准化量，**不是高斯概率或事故置信度**。

这是对通用掩码预训练模型的实验性使用。未来目标由看完整窗口的非因果编码器生成，不能称为真实物理状态的因果预测。没有动作控制输入，也没有使用 V-JEPA 2-AC。短窗口、整幅补边与预训练设置存在分布差异。

`center` 近似官方缩放/中心裁剪几何，宽屏两侧会被裁去；`pad` 将全视野等比缩放到正方形并以均值色补边，适合本例同时保留挖掘机和卡车，但不保证与训练分布一致。OpenCV 插值与官方处理器实现也可能有数值差别。请勿比较不同预处理设置下的原始残差。

## 6. 参数和输出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | HF 官方 ViT-L ID | 可改成本地目录，实测使用 `models/vjepa2-vitl` |
| `--revision` | 空 | 在线 HF 模型可固定 commit；本地转换用 manifest 标识 |
| `--fps` | 8 | 抽帧率；高于源帧率时只取全部源帧，不复制 |
| `--window` / `--stride` | 16 / 8 | 帧数；窗口须为 4 的倍数且不超过配置帧数 |
| `--batch` | 1 | MPS 先用 1，确认内存后再增大 |
| `--fit` | center | 本例明确使用 pad |
| `--dtype` | auto→float32 | 显式 float16 可省内存，但应重新验证数值和基线 |
| `--threshold` | 3 | 固定复核 z 阈值，不应在测试数据上调优后再报告效果 |
| `--top-k` | 5 | 导出相互错开的高分窗口；0 可禁用 |
| `--start` / `--duration` | 0 / 全片 | 源视频的绝对起点与截取长度（秒） |
| `--baseline-csv` | 空 | 外部同配置正常视频的分数 |
| `--annotated-video` | 关 | 导出全视野离线复核视频 |
| `--overwrite` | 关 | 输出目录非空时需显式允许覆盖 |

| 文件 | 内容 |
|---|---|
| `scores.csv` | 每窗原始残差、z、变化量、flag；目标起止时间 |
| `run_config.json` | 模型、权重标识、采样和预处理、精度、方法版本 |
| `summary.json` | 视频 SHA256、设备、软件版本、统计基线和耗时 |
| `events.json` | 超阈值目标区间及峰值，空列表表示未检出 |
| `timeline.png` | 残差与变化量的时间曲线 |
| `features.npz` | 窗口平均特征和原始 patch 残差；顺序与 CSV 对应 |
| `top_*.jpg` | 左为模型视野，右为残差热图；图上文字为窗口分数 |
| `review.mp4` | 原视野附分数/复核提示，启用对应选项后生成 |

`end_s` 是窗口最后一张采样帧的时间；`target_start_s` 是第一张未来帧时间，`target_end_s` 使用最后一帧加估计采样间隔，采用半开区间。事件合并允许 0.05 秒的采样/取整间隙；更大的空隙不会合并。热图是未来半窗的时间平均，叠在一张代表帧上，不能作精确逐帧定位。

热图按每张图误差的高分位相对拉伸，所以即使原始分数很低也可能显示红色；应结合 CSV/时间线判断，不凭红色面积断言异常。复核视频是离线输出，至少需等窗口结束才能算分数，不代表零延迟实时告警。

## 7. 独立正常基线

不指定外部基线时，以待分析视频自身统计量归一化，仅适合描述性排序；异常占比高时会污染统计，不能作为前瞻监测的独立校准。

```bash
python vjepa2_site_monitor.py data/raw/excavator.webm \
  --model models/vjepa2-vitl --fit pad --duration 8 --out outputs/calibration
python vjepa2_site_monitor.py data/raw/excavator.webm \
  --model models/vjepa2-vitl --fit pad --start 8 \
  --baseline-csv outputs/calibration/scores.csv --out outputs/holdout
```

这是复现管线的小示例，8 秒不能代表真实工地的正常分布。部署时应用独立、同机位、覆盖正常工况和光照变化的较长视频，并由人工确认正常。程序拒绝缺少配置、模型/权重/精度/预处理不一致、空或非有限分数的基线。分布可比性仍需使用者判断。

## 8. 有效性验证与结论范围

### 真实危险主实验

```bash
bash scripts/run_real_hazard.sh
```

使用事前 8 秒、7 个窗口校准，纳入测试的未来半窗为 **6 个真实倒塌正窗口、9 个无可见倒塌负窗口**，另保留但排除统计 7 个过渡/事后窗口。三个片段没有共享源帧，窗口按未来采样帧完全落在已标区间的规则分类；标签不会传入模型。

| 方法 | AUROC | TP / FN | FP / TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| V-JEPA predictor L1 | 0.5926 | 2 / 4 | 3 / 6 | 33.33% | 33.33% |
| 潜特征保持 L1 | 0.5926 | 0 / 6 | 0 / 9 | 0% | 0% |
| 像素帧差 | 0.3704 | 2 / 4 | 4 / 5 | 33.33% | 44.44% |

V-JEPA 残差检出了这个事故，但只检出 2/6 个正窗口，Precision=40%、F1=0.3636、平衡准确率=0.5，不能称为可靠危险识别。首次命中正窗口的最后输入帧约在可见事件起点之后 0.9 秒，这不是事故前预警，也不含实际推理开销。分数最高的若干时刻来自运镜，体现摄像机干扰。

**独立事故数是 1。** 6 个相邻正窗口不能当作 6 起独立事故；源片较低分辨率且存在手持运动。完整标签、误报、漏检、对照、原始分数和运行记录见[真实危险报告](reports/REAL_HAZARD_VALIDATION.md)及 `reports/real_hazard/`。主分析与评估重算分数最大绝对差约 `1.19e-7`。

### 原有合成扰动补充实验

预先定义的方案在 [docs/validation_protocol.md](docs/validation_protocol.md)。执行：

```bash
python scripts/validate_method.py --device mps
python scripts/write_validation_report.py
```

- 校准为原视频 `[0,8)` 秒；测试从 8 秒起每两秒一窗，无帧交叠，也不与校准共享帧。
- 只编辑每个测试窗的未来半段：中央 50%×50% 灰色遮挡、亮度 +0.4 后截断到 `[0,1]`、冻结为最后一张上下文画面、未来帧逆序。
- 每种扰动与同一窗的原始版本配对。报告 V-JEPA L1、潜特征保持（复制最后上下文 tubelet）和像素帧差三种方法的 AUROC、配对增量/胜率、固定 z≥3 的 TPR/FPR。
- 使用预训练模型实际检查：未来变化不应改变上下文预测，但应改变目标表征。小型随机模型仅用于代码单元测试，**不用于有效性结果**。
- 单一视频的少量合成正样本可暴露漏检和摄像头干扰，不能证明跨工地泛化、真实事故识别、违规识别或优于专用检测器。所有扰动结果都会保留，包含无效或劣于基线的结果。

实测数值由 `reports/validation/metrics.csv` 和 `summary.json` 给出，[验证报告](reports/VALIDATION.md)解释这些结果。未在测试数据上更改阈值来追求更高检出率。

本次 MPS/float32 实测，全片 16 窗分析耗时 9.59 秒（不含下载、加载和复核视频编码）。最终 v2 受控实验结果如下，每类仅有 4 个配对正样本：

| 扰动 | V-JEPA 残差 AUROC | 固定 z≥3 检出率 |
|---|---:|---:|
| 中央遮挡 | 1.0000 | 100% |
| 曝光增亮（保持补边不变） | 0.8125 | 75% |
| 未来冻结 | 0.5000 | 0% |
| 未来逆序 | 0.4375 | 25% |

未编辑的 4 个测试窗口有 1 个误报，FPR=25%。遮挡的 AUROC 与潜特征保持基线相同；本例没有证明 predictor 整体优于简单基线。热图的补边区域也可能有较高残差，当前标量评分包含这些 patch。完整视频自身归一化没有告警，而独立短基线会将约 12–14 秒的正常回转标为复核，说明校准选择影响很大。

图像核对时修正了初次曝光扰动错误地改变补边的问题；全部初次记录保留在 `reports/validation_v1_padding_exposure/`，最终结论采用 v2。该修正没有改变划分、阈值或其他扰动。

## 9. 检查、故障与限制

```bash
python -m pytest -q
```

测试包括：抽帧时间、截取右边界、预测区间、事件不跨空隙合并、退化统计量、未来像素隔离、扰动只修改未来、AUROC 平分、转换名称、参数校验。真实预训练模型另由验证脚本检查，不将单元测试通过等同于科学有效性。

发布前 **33 项测试通过**，包含原有检查、真实危险标签/混淆矩阵，以及模型自动回退、已有模型哈希复用和损坏文件拒绝检查，记录见 `reports/test_results.txt`。

- **模型下载失败**：检查网络/证书，使用官方 Meta 备用路径；不需要 HF token，不要关闭证书验证。
- **MPS 内存不足**：使用 `--batch 1`；关闭其他占用 GPU 的应用；再考虑 float16 并单独校准。CPU 可运行但速度不同。
- **NaN/Inf**：使用 float32；程序会拒绝非有限分数。
- **视频过短/帧率无效**：至少需要一个完整窗口；检查视频或先转 CFR。
- **没有超阈值片段**：这是合法结果；查看原始曲线与 Top-K，不把“没有告警”解释为现场安全。
- **摄像头移动/曝光变化**：可能导致高残差；需要现场校准及人工复核，本项目不能自动消除这些干扰。
- **长录像内存**：解码窗口和 Top-K 缓存有界；为导出特征会积累每个窗口的特征/热图，内存随窗数增长。建议按 `--start/--duration` 分段分析，并用独立正常基线保持可比性。
- **结果覆盖**：主分析默认拒绝非空目录；需要重跑时用新目录或显式 `--overwrite`。验证目录固定，重跑会更新验证文件。

仓库地址：[hitqht/V-JEPA2-](https://github.com/hitqht/V-JEPA2-)。提交内容包含完整代码、文档、下载脚本、标注和小型实验结果；`.venv/`、缓存、模型目录、权重扩展名、原/派生视频与 `outputs/` 由 `.gitignore` 排除。

## 10. 官方依据

- [Meta V-JEPA 2 官方代码和模型](https://github.com/facebookresearch/vjepa2)
- [官方 ViT-L 模型卡与配置](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
- [固定版本 Transformers V-JEPA 2 实现](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/vjepa2)
- [Transformers V-JEPA 2 接口说明](https://huggingface.co/docs/transformers/en/model_doc/vjepa2)
- [真实倒塌录像及权利记录](https://commons.wikimedia.org/wiki/File:Big_Blue_crane_collapse_at_Miller_Park.webm)
- [OSHA Miller Park 起重机倒塌案例](https://obis.osha.gov/dcsp/success_stories/compliance_assistance/abbott/abbott_casestudies/slide30.html)

用户参考代码原样保留在 `reference/`；本项目对其方法性描述作了上述限定，没有把参考注释中的判断当成已经得到实验证明的事实。

## 11. 代码目录

```text
vjepa2_site_monitor.py            主分析 CLI：特征预测、打分、事件、热图与复核视频
scripts/
  download_model.py               HF / Meta 官方权重下载、转换、校验与复用
  prepare_real_hazard.py          真实危险录像下载、校验、短片切分及标签 manifest
  run_real_hazard.sh              真实危险实验一键流程
  validate_real_hazard.py         固定真值下的检出、误报、AUROC 和基线比较
  render_hazard_review.py         事后真值/模型分数对照视频
  prepare_demo.py                 正常挖掘机视频下载及小副本
  validate_method.py              合成扰动补充实验
  write_validation_report.py      补充实验报告生成
  run_demo.sh                     原有补充实验与真实危险实验
tests/                           无需预训练权重即可运行的测试
data/*.json                      来源、短片映射与人工真值
docs/                            协议、实现说明、第三方素材署名
reports/                         已实测指标、图像和报告；不包含权重
reference/                       原始参考代码
requirements*.txt                依赖与本次环境锁定版本
```

本机生成的 `models/`、`data/raw/`、`data/derived/`、`outputs/` 和 `.cache/` 均不随 Git 上传。需要重现完整功能时运行下载/实验脚本即可。
