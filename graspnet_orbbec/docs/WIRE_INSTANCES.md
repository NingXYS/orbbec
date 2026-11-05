# 多线材实例与歧义schema

## 范围与行为

自动识别保留每个 `selected` SAM掩码，逐来源、逐连通分量提取骨架与候选；不再对OR合成结果分析并以其作为唯一实例依据。`mask`、`skeleton`仍是GUI显示层的合成图，兼容既有窗口、保存文件及统计。

近乎相同的重复SAM掩码仍去重，但改用对称IoU大于0.90；不再因小掩码被包含就直接删除它。被保留的独立假设不等于物理线材数量：多个假设可描述同一根线，单个假设也可能包含多根粘连线。

- 单线：独立掩码及明确的二端点路径；普通端部确认和内部交界点击保持可用。
- 平行且不接触的独立SAM掩码：保留独立ID、中心线、端部候选和端部区域；不拼成一条线。
- 同一来源含多个不连通分量：保留各分量，不补像素、不跨断口连接，标记 `disconnected_source_mask`。不能自动判断它们是断线还是不同物体。
- 两个掩码重叠或8邻域接触：仍保留两份，建立 `touch_or_crossing_unresolved`关系，两者均标记歧义。即使单份掩码的中心线连续，也不据此证明交叉处身份已解算。
- 同一掩码出现分叉、闭环或异常端点数：不拆成“已确认的多根线”，保留几何假设及歧义原因。
- 两个片段端点间隔不超过12像素、各自8步切线均朝向缺口且方向余弦大于0.8时，只记录 `possible_fragment_continuation`，不接线、不认定同一实例。较大或遮挡复杂的缺口不保证能检测。
- ROI边缘4像素内的骨架端点标记截断，不作为物理端部候选；保留可见片段。`roi_truncated`不单独等同于身份歧义，但也不证明观察到了完整线材。

没有使用截图中特定棋盘或端子坐标，不改变自动正负点生成、14mm策略或机械臂代码。

## 几何检查

每个连通分量独立记录：骨架像素数、端点数、分叉像素数、线宽中位数/95百分位、主路径切线转角及ROI截断状态。数字骨架转角中有正交连接时不再增加冗余对角边，减少伪分叉。

除接触关系外，以下检查会标记歧义：存在骨架分叉；端点数不为2；骨架有环；中心线不可用；95百分位线宽大于中位线宽2.5倍；主路径8步窗口最大转角大于75°。这些是固定的保守像素启发式，不是经过标注集调参的物理判据。真实急弯也可能被拒绝，均匀宽物体也可能通过。

主路径仍可用最远点搜索作**预览**；有分叉/异常几何时，候选不会因它是较长路径就得到 `path_ambiguous=false`。分叉实例还保留各端部的局部路径，但不解算交叉处的物理延续方向。

## WireResult兼容性

原有dataclass字段顺序及构造参数保留，新增字段位于末尾：

| 字段 | 内存内容 |
|---|---|
| `instances` | `WireInstance`列表，含独立二值mask、路径、来源、几何和歧义原因 |
| `path_instance_ids` | 与原 `paths_xy`逐项对应的实例ID |
| `source_masks` | selected来源掩码列表，均为原图尺寸 |
| `instance_relations` | 未解决的接触/交叉及疑似片段延续关系 |

`paths_xy`和旧JSON的 `centerlines_by_component`仍提供每个分量的预览主路径；可能包含歧义路径，使用者必须检查其所有者。`centerline_xy`只在恰有一条预览路径且实例无歧义时提供，不把多个实例串起来。GUI叠加 `W001`等ID；问号表示歧义，不表示新的已确认目标。

实例ID只在本帧结果内唯一，不是跨帧跟踪ID。`source_mask_id`对应来源索引；分量拆分或排序变化可能改变ID，不能沿用旧帧的ID定位新场景。

## recognition.json扩展

保持既有字段，新增独立版本号 **`wire_instances_schema_version: 1`**；不是把整个旧文件格式替换为新格式。

| 新字段 | 含义 |
|---|---|
| `instances[]` | `instance_id`、`source_mask_id`、`mask_file`、`mask_pixels`、`centerlines_xy`、`geometry`、`instance_ambiguous`、`ambiguity_reasons`、`identity_scope` |
| `source_masks[]` | 每个来源的ID和PNG文件名 |
| `path_instance_ids[]` | 旧预览主路径的实例归属 |
| `instance_relations[]` | 两实例ID、关系类型及重叠像素数 |
| `instance_ambiguous` | 任一实例存在歧义即为true，属于整帧摘要 |

`identity_scope`为 `single_image_mask_hypothesis`，避免把分割假设冒充物理身份。中心线路径及端点均使用完整原图像素坐标；ROI会先恢复再分析，掩码不拉伸。

`candidates[]`新增 `instance_id`；自动聚合结果还带 `instance_ambiguity_reasons`。`path_ambiguous`继承实例歧义。旧 `component_id`在自动聚合时为该结果内实例编号，不是物理线材编号。

保存目录新增 `source_mask_S001.png`及 `instance_W001.png`等原图尺寸0/255掩码，JSON用相对文件名引用。原有 `color.png`、`mask.png`、`skeleton.png`、`overlay.png`和端部文件保持。

早期手工构造的WireResult默认没有实例字段内容，仍可显示、保存；不能从缺少实例数据推断已完成分实例。已有冻结JSON不自动补写或伪造实例。

## 人工交界确认

`confirm_body_boundary()`先查找路径所属实例。点击距离在12像素内且最近候选归属相差不超过2像素时，若存在多个归属或所有者本身有歧义，就明确报错；不修改上一次确认结果。清晰单线的内部交界仍可从主路径生成M候选，但拥有实例ID，端部区域只在该实例掩码内扩展。

旧端点确认API仍可记录交界；歧义候选继续输出 `wire_instance_ambiguous`，不会因此获准规划。端子辅助关联只在无歧义路径中查找，不改变YOLO模型、权重或检测调用。

这只调整实例归属与确认校验，没有更改14mm弧长计算、手眼变换、TCP或运动接口。GUI仍是快照预览，不会发送机器人命令。

## 离线验证及限制

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .conda/bin/python -m pytest tests/test_wire_instances.py -q -p no:cacheprovider
```

测试覆盖单线、平行双线、交叉双线、T接触、单/多来源局部断线、ROI截断、宽度变化、急转切线、重复假设去重、保存schema和人工点击。全部使用合成图或模拟掩码，不运行机械臂。

当前不是跨帧实例跟踪或完整遮挡重建：不能自动解开真实交叉、恢复消失的线段、确定上下层关系或证明线材类别。保守歧义会减少可点击目标；应分开线材、改善画面或提供更清晰的独立掩码。未经真实标注集验证，不能声称多实例识别准确率或真机抓取成功率。
