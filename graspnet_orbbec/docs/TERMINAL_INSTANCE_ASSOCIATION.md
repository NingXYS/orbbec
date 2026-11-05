# YOLO端子与SAM2线材实例关联

## 边界

本功能只关联已有检测与线材实例，不训练/替换模型，不修改YOLO推理、权重、手眼、14mm几何或运动代码。YOLO检测只提供语义线索；匹配成功不代表交界真实、尺度有效或可执行。

默认 `terminal_auto_boundary: false`与 `terminal_use_seeds: false`保持不变。未增加自动执行入口；日常流程继续采用SAM2与Shift+左键确认交界。YOLO漏检、误检或关联歧义，不清除可用的SAM实例，也不阻止在清晰SAM路径上进行人工确认。

## 实例匹配

每个端子对**每个独立WireInstance**评估，不读取合成OR掩码作为唯一中心线依据：

- 端子掩码到该实例主路径的距离、到实例掩码的局部间隔。
- 端子PCA长轴与最近中心线局部切线的一致性。轴不可靠时保留歧义，不靠最近距离强配。
- 最近路径点到该线材可见端点的弧长，防止把线身中部的金属块当端子。
- 端子短轴/长轴相对线宽的比例，以及实例自身的歧义状态。
- 端子与本实例直接重叠，或短间隔内存在不碰其他实例的局部连接区域。短间隔只是几何支持，不填补分割像素，不声称证明物理相连。
- 端子或连接区域接触其他实例、线材交叉、多个实例支持近似时返回 `ambiguous`。
- 两个检测竞争同一实例同一端部时，两个都标为歧义，不按置信度任意二选一；同一线材的两个相反端部可以分别匹配。

亮度与亮且低色差像素比例只写入 `appearance_auxiliary`，`used_for_match_decision=false`；不用于阈值通过或加分。亮银色、最高置信度或最近距离均不能单独决定匹配。

## 阈值及评分

`wire.auto`下新增配置，未填写时使用同样默认值：

```yaml
terminal_path_max_distance_px: 12
terminal_match:
  max_gap_px: 6
  endpoint_zone_px: 24
  max_axis_angle_deg: 35
  min_aspect_ratio: 1.5
  min_score: 0.55
  ambiguity_margin: 0.15
terminal_auto_boundary: false
terminal_use_seeds: false
```

比例硬条件：端子短轴/线宽在0.3–6、长轴/线宽在1–20；端子前景至少6像素。多片端子掩码在靠近线材时保留歧义，不能把反光裂片直接认定为唯一端子。端子置信度还须满足原 `terminal_confidence`，非法/非有限置信度不参与有效匹配。

评分由路径距离25%、轴方向30%、局部连接25%、端部位置15%、检测置信度5%组成。所有硬条件通过且达到最低分才进入有效候选；多个有效候选分差不超过0.15时判歧义。分数是几何排序值，不是正确率或执行概率，不能覆盖硬拒绝条件。

这些像素阈值和比例为保守启发式，未通过真实标注集校准。相机分辨率、目标大小或工位变化后应离线复核，不能只为让结果通过而放宽。

## 保存schema与拒绝原因

保持 `WireResult`和现有JSON字段；不用新的构造参数。`terminal_detections[].wire_association`新增schema内容，同时在 `automation.terminal_association`中汇总（`schema_version: 1`、原图像素坐标）：

| 字段 | 含义 |
|---|---|
| `status` | matched、ambiguous或unmatched |
| `instance_id`、`endpoint_index` | 唯一通过匹配的实例与可见端部；歧义时不保留有效实例归属 |
| `score` | 最佳匹配或竞争前的分数，不能仅据此使用目标 |
| `semantic_evidence_only` | 固定true，不表示执行资格 |
| `matches[]` | 每个实例的score、eligible、features、rejection_reasons，以及最近路径点/端部信息 |
| `appearance_auxiliary` | 亮度及低色差比例，仅辅助排查 |
| `rejection_reasons` | 整体失败/竞争原因；单对几何拒绝在matches内 |

features包含 `path_distance_px`、`mask_gap_px`、`endpoint_arc_px`、`tangent_xy`、`terminal_axis_xy`、轴夹角、宽/长比例、重叠状态、连接类型及局部竞争实例ID。所有点、掩码和路径在ROI恢复到完整RGB后匹配，坐标不变换为机器人系。

主要拒绝码包括：`terminal_too_far_from_instance_path`、`no_local_connection_support`、`terminal_not_near_wire_end`、`terminal_axis_not_aligned_with_wire`、`terminal_axis_unreliable`、`terminal_wire_geometry_implausible`、`wire_instance_ambiguous`、`competing_instance_local_contact`、`multiple_instances_with_similar_support`、`multiple_terminals_compete_for_same_wire_end`、`terminal_mask_fragmented`及缺失/非法掩码。无匹配时仍保留原始检测。

几何端部候选只接收同实例、同端部且距离合理的匹配证据，保留 `confirmed`原状态。整帧 `terminal_semantics_verified`仍为false。

## 可选模型交界实验

`attach_terminal_guided_boundaries()`现在也检查 `terminal_auto_boundary`开关；默认关闭时不产生预测点。显式开启后，只接受无歧义实例匹配，按已匹配端部的朝内路径产生预测，不再在所有路径中取最近点或选择较长一边。

预测候选标记 `confirmed=false`、`boundary_confirmation_source=model_prediction`。即使后续算出坐标，[任务5门控](VISUAL_PATH_VALIDATION.md)仍因未人工确认而阻断执行；YOLO不能单独解除此门控。

旧测试中“线材中部一个近邻检测就生成已确认Y候选”的行为已取消。历史候选与实验坐标只可追溯，不自动升级为当前目标。

## API与离线测试

- `associate_terminal_instances(result, cfg)`：返回逐检测报告，同时更新检测与automation中的关联证据；不改实例掩码、路径、交界或执行字段。
- `attach_terminals(result, detections, masks, cfg=None)`：保留原三个位置参数兼容，新增可选配置；仅更新语义证据与警告。
- `attach_terminal_guided_boundaries(result, cfg)`：默认不生成交界；实验输出也只是未人工确认的预测。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .conda/bin/python -m pytest tests/test_terminal_instance_association.py -q -p no:cacheprovider
```

测试使用合成RGB、检测掩码和模拟推理，覆盖正确匹配、漏检、相邻端子、反光、错误金属物体、交叉和ROI。不加载真实模型或运行机器人。

如果误检金属物体恰好在端部且几何与端子相似，仍可能产生错误语义匹配；本方法不能证明材质、端子类别或遮挡下的物理连接。短粗/圆形端子、严重反光碎片和密集接触线束可能被保守拒绝。需要真实标注数据才能评估效果，不能把合成测试成功当成真实识别准确率或抓取成功率。
