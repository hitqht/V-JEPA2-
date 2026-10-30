#!/usr/bin/env python3
"""仅依据保存的实测 JSON/CSV 生成报告，不填入推测的指标。"""
import csv
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABELS={'occlusion':'中央遮挡','exposure':'全局增亮','freeze':'未来冻结','reverse':'未来逆序'}


def main():
    base=ROOT/'reports/validation'
    summary=json.loads((base/'summary.json').read_text())
    model=json.loads((ROOT/'reports/model_manifest.json').read_text())
    metrics=list(csv.DictReader((base/'metrics.csv').open()))
    rows=[]
    for r in metrics:
        rows.append(f"| {LABELS[r['kind']]} | {r['method']} | {float(r['auroc']):.3f} | "
                    f"{float(r['paired_mean_delta']):+.6f} | {float(r['paired_win_rate']):.1%} | "
                    f"{float(r['tpr_z3']):.1%} | {float(r['fpr_z3']):.1%} |")
    report=f'''# 本机 V-JEPA 2 工地视频实验记录（合成扰动补充实验）

新增真实施工危险正样本的主实验见 [REAL_HAZARD_VALIDATION.md](REAL_HAZARD_VALIDATION.md)。本报告仅记录原有正常视频和合成扰动，以下“无真实事故标签”的范围指本补充子实验。

实验日期：2026-10-30。以下数值来自真实预训练模型在本机运行产生的文件，不是示例输出。

## 条件与来源

- MacBook Pro，Apple M4 Max，40 核 GPU，64 GB 内存；macOS 26.1。
- 推理设备 `{summary['device']}`，精度 `{summary['dtype']}`，PyTorch 2.8.0，Transformers 4.57.1。
- 权重来源：`{model['source']}`。标识：`{model['revision']}`；本地 safetensors SHA256：`{model['converted_weights_sha256']}`。
- 视频：KEmel49 的 [Excavator digging land and loading truck](https://commons.wikimedia.org/wiki/File:Excavator_digging_land_and_loading_truck.webm)，CC BY-SA 4.0；视频 SHA256：`{summary['video_sha256']}`。
- 采样 8 fps，16 帧/窗，256×256 均值色补边，无微调。
- 校准使用 `[0,8)` 秒的 {summary['calibration_windows']} 个正常窗口；测试每类 {summary['test_windows_per_condition']} 个互不重叠的窗口，起点为 {summary['test_starts_s']} 秒。校准和测试不共享帧。
- 原视频没有经核实的事故正例。正样本是确定性编辑的画面/时序扰动，详见[预定义方案](../docs/validation_protocol.md)。
- 最终协议为 `{summary.get('protocol_version','v1')}`：曝光编辑只作用于原画面，预处理补边保持不变。初次包含补边增亮的实现已修正；初次结果完整保留在 `validation_v1_padding_exposure/`，未作为最终指标。

## 预训练模型的未来输入隔离检查

保持前半窗不变，修改未来半窗后，预测向量最大绝对差为 **{summary['future_leak_check_max_abs_prediction_difference']:.9g}**；目标表征平均绝对变化为 **{summary['target_change_mean_abs']:.6f}**。前者低于预先设定的 `1e-5` 容差，符合预测器没有使用未来像素的要求。

这不把离线教师目标变为因果表征。目标编码器仍可见完整窗口，分数应解释为掩码潜表征残差。

## 全部受控结果

AUROC 以“分数越高越像被编辑”为方向；0.5 为随机排序参考。配对胜率为编辑后分数高于同窗原版的比例。固定阈值使用校准集 median/MAD 得到的 z≥3，没有用测试分数调参。FPR 是相同的未编辑测试窗口误报率。

| 扰动 | 方法 | AUROC | 配对平均增量 | 配对胜率 | TPR z≥3 | FPR z≥3 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

方法：`vjepa_l1` 为预训练 predictor 的残差；`persistence_l1` 为复制最后一个过去 tubelet 的潜特征残差；`frame_difference` 为模型视野的相邻帧 RGB 差分平均值。三种方法各用自己的正常校准统计。

![全部方法对比](validation/comparison.png)

![原版与四种确定性未来扰动](validation/interventions.jpg)

## 逐项解释
'''
    for kind in LABELS:
        r=next(r for r in metrics if r['method']=='vjepa_l1' and r['kind']==kind)
        a=float(r['auroc']);t=float(r['tpr_z3'])
        direction='高于' if a>.5 else ('低于' if a<.5 else '等于')
        report+=f"\n- **{LABELS[kind]}**：V-JEPA 残差 AUROC={a:.3f}，{direction}本次 0.5 排序参考；固定阈值检出 {t:.1%}。"
        if kind=='exposure':
            report+='曝光是摄像头干扰，即使能检出，也不能作为识别危险施工行为的证据。'
        if kind in ('freeze','reverse') and (a<=.5 or t<.5):
            report+='该条件显示时序变化可能漏检，不能默认世界模型残差会识别所有时序异常。'
    report+='\n\n## 原视频的离线分析\n'
    full_path=ROOT/'outputs/excavator/summary.json'
    if full_path.exists():
        full=json.loads(full_path.read_text())
        shutil.copyfile(ROOT/'outputs/excavator/timeline.png',base/'source_timeline.png')
        report+=(f"\n完整源视频产生 **{full['windows']} 个窗口**、**{full['events']} 段复核事件**；"
                 f"分析阶段耗时 **{full['analysis_seconds']:.2f} 秒**，约 **{full['windows_per_second']:.2f} 窗口/秒**。"
                 '该耗时不含模型下载/加载及复核视频编码；完整视频采用自身统计归一化，属于描述性分析。'
                 '这里的复核事件没有事故标签，不能计算真实施工事故 precision/recall。\n\n'
                 '![完整视频时间线](validation/source_timeline.png)\n')
    holdout_path=ROOT/'outputs/holdout/summary.json'
    if holdout_path.exists():
        holdout=json.loads(holdout_path.read_text())
        report+=(f"\n另用 `[0,8)` 秒的正常基线分析后续视频，真实模型的外部基线管线产生 "
                 f"{holdout['windows']} 个窗口和 {holdout['events']} 段复核事件。"
                 '其中约 12–14 秒为正常回转阶段，说明动作阶段变化也可能触发高残差。'
                 '该流程的窗口步长为 1 秒，与上表不重叠 2 秒测试窗的统计口径不同。\n')
    report+=f'''

受控验证耗时 {summary['analysis_seconds']:.2f} 秒（不含模型加载）。复核视频和 Top-K 图在 `outputs/excavator/`；分数、特征、事件及运行元数据一并保留。

## 结论与不能推出的结论

已验证本机可以使用真实 V-JEPA 2 权重完成视频抽帧、过去独立编码、未来潜特征预测、残差打分、可视输出和固定方案的对照评估。检测效果必须逐项按上表判断；代码运行成功不代表事故识别有效。

遮挡条件下 predictor 和特征保持基线的 AUROC 相同，后者的固定阈值误报更低；冻结和逆序未获得可靠检出。当前结果没有证明 predictor 残差整体优于简单基线。热图中的补边区域也可能有较高残差，当前标量评分包含这些 patch，不宜作物理位置的直接解释。

本实验只有一个来源，每类 {summary['test_windows_per_condition']} 个配对正样本；动作阶段、背景和时序相关，正负样本也有配对依赖。没有独立工地、独立标注者、真实异常正例或统计显著性结论。不能报告泛化准确率、事故检出率、施工违规识别能力或整体优于简单基线。

若用于研究结论，下一步应获取依法可用的多机位正常/真实异常视频，独立定义危险事件标签，按工地/日期分离训练校准与测试，并与帧差、光流、目标检测/跟踪及经过训练的异常检测器公平比较。光照、遮挡、镜头移动需要单列干扰集。

原视频和包含其画面的派生图像按 CC BY-SA 4.0 署名，见[素材说明](../docs/ATTRIBUTION.md)。
'''
    (ROOT/'reports/VALIDATION.md').write_text(report)
    print(ROOT/'reports/VALIDATION.md')

if __name__=='__main__':main()
