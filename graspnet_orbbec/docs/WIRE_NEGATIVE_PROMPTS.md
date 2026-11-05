# 自动SAM2逐候选正负点实验

本功能仅改变 `segmentation/automatic.py` 的自动点提示。正点仍由[任务2种子算法](WIRE_SEED_REGIONS.md)生成；不改变人工框选/正负点接口、掩码合并、YOLO、14mm策略、MoveIt或机械臂控制。

## 开关与可信背景

- 正式 `configs/default.yaml`：`wire.auto.negative_prompts.enabled: false`，保持原正点模式。
- 实验 `configs/wire_negative_prompts_experimental.yaml`：负点和调试开关默认打开；这是完整YAML，使用时显式传入，不自动替换正式配置。
- 实验配置的 `trusted_regions`默认为空，**不会臆测背景**，此时每个候选仍退回正点模式。需操作者按当前RGB确认无真实线材的背景矩形后再填写；场景变化后重新核实。

```yaml
negative_prompts:
  enabled: true
  coordinate_space: original_rgb
  trusted_regions: []
  max_points: 2
  min_distance_px: 12
  max_distance_px: 80
  wire_clearance_px: 6
  max_contrast: 6
  patch_radius_px: 2
  max_patch_range: 8
  spacing_px: 12
```

矩形沿用任务2半开整数xyxy规则，`original_rgb`表示完整RGB，`roi`表示当前推理ROI局部坐标。越界部分取交集；空列表、无交集或没有安全像素时退回正点，`null`或非法配置明确报错。不能把种子排除区自动当可信背景：被排除的棋盘、夹具区域不等于适合放负点的区域。

对可信背景还会做保守检查：局部补丁须完全在可信矩形内；对比度与局部灰度范围足够低；远离所有检测到的高对比线条和所有正点。然后在**每个正点自己的距离环带**内选点，负点之间保持间隔。不是从全图挑固定一组，再无条件施加给每条线材。

这些是视觉启发式，不是线材语义证明。低对比、透明或细于像素的真实线仍可能逃过保护，因此可信区域的人工确认不能省略。区域改变后应重新检查调试图。

## 使用现有入口

在 `graspnet_orbbec`工作目录，用实验配置替换现有命令的配置参数：全局 `--config configs/wire_negative_prompts_experimental.yaml`必须放在 `preview`或`recognize`子命令之前。快照、平面、实测高度和工具参数仍按[日常手册](REAL_GRASP_STARTUP.md)提供；本实验不会自动开启相机、运动或使能。

纯离线测试命令（模拟predictor，无权重和真机要求）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .conda/bin/python -m pytest tests/test_wire_negative_prompts.py -q -p no:cacheprovider
```

本功能仅支持 `wire.auto.method: line_prompts`；`dense`模式开启负点时明确报错，不能无声忽略开关。

## SAM2批次与回退

已核对仓库内 `third_party/sam2/sam2/sam2_image_predictor.py` 的 `_prep_prompts()`和`_predict()`：

- 同一已编码图片上的独立候选使用 `point_coords: B×N×2`（float32局部xy）及 `point_labels: B×N`（int32）。每组首点标签1，后续背景点标签0。
- 不同负点数量的候选分批提交；无负点候选为 `B×1×2`、`B×1`。不复制共享负点，不用伪坐标填充，也不把多条线材的正点合成一组。
- 返回值兼容单候选 `C×H×W/C` 与多候选 `B×C×H×W/B×C`；恢复批次维后检查尺寸。
- 混合提示的合格掩码必须包含本候选正点、不包含本候选负点，并满足原SAM评分阈值。若没有合格掩码，仅将该候选以正点模式重试，保留其他候选结果。
- 混合请求被predictor以ValueError拒绝或返回批次尺寸不匹配时，对该小批次回退正点；模型/GPU等RuntimeError不会被当作“无可靠背景”吞掉。

回退可能恢复正点模式原有的背景误分割，不能把回退当作结果正确的证据。该操作只影响自动提示与其对应的预测记录，不另改掩码筛选或合并算法。

## 调试信息

现有 `seed_debug.enabled: true`时，`output/wire_seeds/seeds_<唯一ID>/`中保留任务2的 `seeds_roi.png`、`seeds.json`，另输出：

- `prompts.json`：逐候选正点、负点的ROI/完整RGB坐标、初始 `point_labels`、实际最终 `final_point_labels`、`final_mode`、回退原因及混合/纯正点计数。
- `prompt_000.png`等：每个候选独立一张ROI大小图。绿圆为正点，红叉为选取的负点，上方文字标注最终模式；不同候选的负点不会混画成同一组。

若最终为 `positive_only`，红叉仅表示尝试过的负点，**不表示重试时仍使用了这些负点**。JSON的 `final_point_labels=[1]`及回退原因说明实际回退。

GUI按S保存的 `recognition.json`亦记录 `automation.sampling.automatic_prompts`。`mixed_candidates`统计未回退的混合提示候选，不代表真实线材检出数量；调试图与JSON会随刷新累积，注意归档。

## API与范围

`automatic_background_points(image, points, cfg, roi_xyxy=None)`返回逐正点的负点列表和可序列化诊断；输入正点、返回负点均为ROI局部坐标，诊断同时记录原图坐标。它依赖自动种子给出的有效图内正点。

`predict_automatic_prompts(...)`只处理该自动模式的批次提交、提示一致性检查及按候选回退，最终记录仍按原正点顺序交给既有掩码筛选。人工鼠标逻辑及 `Sam2WireSegmenter.segment(box=..., points=..., labels=...)`保持不变。

当前验证来自合成图、模拟predictor和原有回归测试；未验证真实SAM2模型对当前工位的分割精度提升，也未运行真机控制。需要在标注照片上比较默认/实验配置，才能评估是否适合正式启用。
