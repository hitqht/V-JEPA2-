# 素材与模型来源

## 视频和派生图像/视频

- 作品：[Excavator digging land and loading truck.webm](https://commons.wikimedia.org/wiki/File:Excavator_digging_land_and_loading_truck.webm)
- 作者：[KEmel49](https://commons.wikimedia.org/wiki/User:KEmel49)
- 许可：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- 下载：2026-10-30；源文件 28,408,467 字节。
- SHA256：`40b48f4a0a29f58416023ecd32453060b361b707c62c782887845217c99c4d8f`。
- 本项目改动：缩放、抽帧、添加时间戳与分析文字、热力叠加；验证数据另外施加灰色遮挡、曝光增亮、冻结、逆序。这些是本项目编辑，原作者不为实验结论背书。
- `data/derived/`、`reports/source_contact_sheet.jpg`、`reports/validation/interventions.jpg`、输出复核视频和包含原画面的 Top-K 图像，作为原视频的改编素材，沿用 CC BY-SA 4.0。再次分发时请保留本署名和改动说明。

## 模型与实现

- [Meta 官方 V-JEPA 2 仓库](https://github.com/facebookresearch/vjepa2)，其许可证以原仓库为准。
- [官方 HF ViT-L 模型卡](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)，标记 MIT。
- 本地备用权重从 [Meta 公共检查点](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt) 获取；仅将 `target_encoder` 和 `predictor` 参数转换为 Transformers 格式，严格匹配每个参数；具体校验和见 `reports/model_manifest.json`。
- 使用 [Transformers v4.57.1 实现](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/vjepa2)，Apache-2.0。
- 用户提供的参考脚本保留在 `reference/vjepa2_site_monitor_original.py`，只作为实现参考；未将脚本中的说明当作经验证的科学结论，也未替用户设定它的再分发许可。

## 新增真实危险录像：Big Blue 施工起重机倒塌

- 作品：[Big Blue crane collapse at Miller Park.webm](https://commons.wikimedia.org/wiki/File:Big_Blue_crane_collapse_at_Miller_Park.webm)。Commons 标识原作者为未具名 OSHA 雇员，按美国联邦政府雇员履职作品标记 `PD-USGov` / [Public Domain Mark](https://creativecommons.org/publicdomain/mark/1.0/)。这是原作品的权利记录；不将其误标为挖掘机素材的 CC BY-SA 许可。
- 下载日期：2026-10-30；原片 4,368,085 字节，源 SHA1 `73fc5052f91beddbcfc18a8c9988cb00042e1886` 与 Commons 公开记录一致；SHA256 `fa4c0de9e1f7e5961d10fdd85e636b10dc99acb5fa41e5f7b0e3059426c71e4e`。
- 事件来源核查：[OSHA Miller Park 案例页](https://obis.osha.gov/dcsp/success_stories/compliance_assistance/abbott/abbott_casestudies/slide30.html)。本次浏览工具未能完整加载该页，搜索索引可读的官方页摘要说明了该录像；视觉事件真值还由实际下载的视频逐帧核对。
- 本项目改动：从源片 [34,42)、[42,50)、[50,66) 秒连续截取三份短片、去音频、恒定 30 fps H.264 编码；导出逐帧/热图和分数文字；人工标注倒塌/无可见倒塌/排除区间。没有插入合成危险，也没有拼接独立事件。
- 原始资料不代表 OSHA 或原作者认可本项目模型、标签或结果。完整编辑映射、权利来源和切片哈希在 `data/real_hazard_manifest.json`。
