# 交界至14mm夹持中心的视觉路径门控

## 范围

新增 `segmentation/path_validation.py`。它验证视觉数据，不控制机械臂；原 `plan_candidate()`、`plan_candidate_v2()`、Phase1.7弧长选择、手眼和TCP变换算法保持原样。`path_ambiguous=false`或坐标算出来，不再单独构成真机执行许可。

从人工/模型提供的交界沿原始米制路径，使用既有 `select_center()`复核14mm中心。只验收交界至该中心的局部区间；不把全图最长路径、掩码像素连通或SAM评分当成物理线材身份凭据。

## 四种状态分开记录

每个 `grasp_plans[]`增加：

| 字段 | 意义 |
|---|---|
| `boundary_verification` | `source`为manual、model_prediction或none；`human_confirmed`单列；旧 `boundary_confirmed`只表示算法拿到了点 |
| `visual_path_validation` | schema_version、passed、rejection_reasons、checks及可复算evidence |
| `verification_stages` | 人工确认、实例路径验证和坐标计算成功分别记录，不相互替代 |
| `coordinate_verification` | 复核所用相机K/D、T_base_rgb及T_flange_grasp的副本，非新的标定结果 |
| `instance_id` | 必须与候选、实例记录及证据匹配 |
| `readiness_blockers` | 保留局部路径拒绝原因，手眼层合并原有安装、平面和工具阻断，不覆盖新门控 |

本版本的模型关联只记录为 `model_prediction`。模型可以提供交界并进行局部路径分析，但不能仅凭预测自动获得执行许可；仍需操作者确认。人工Shift点击更新其来源为manual，随后重新生成米制路径证据。人工确认不代表真实金属/绝缘语义已获得独立计量验收。

坐标计算成功时仍可保留预览位姿，便于定位问题；即使 `status=base_frame_preview_ready`，局部验证失败也必须阻断执行。上游始终保留 `executable=false`和 `motion_commands_enabled=false`，真实接口仍由官方适配层及原安全诊断检查控制。

## 局部检查

1. 候选ID、实例ID、交界输入/吸附位置与路径索引一致；交界到主路径距离不超过原有12像素。
2. 从原米制路径与既有交界投影复算中心，必须与保存的夹持中心、切线一致；偏移必须为14mm。
3. 局部像素路径相邻步长不超过sqrt(2)，不可跳过缺口或反复绕回像素；像素和其间栅格连线必须落在同一实例掩码、同一连通分量内。
4. 路径2像素邻域内不得出现其他实例，骨架不得出现分叉。局部二值细化仅依赖NumPy，避免官方系统Python缺少视觉Conda的skimage而无法复核。已有实例歧义也不能通过局部计算抹掉。
5. 线宽使用局部距离变换，最大值超过“10百分位宽度×1.8”和“该宽度+4像素”两者的较大值时拒绝。
6. 使用最多4步窗口检查像素及米制方向，转角超过75°或方向无法估计时拒绝。

这些像素阈值是保守启发式，会拒绝部分真实急弯或宽度变化，不是标注集准确率或绝对定位精度。该门控支持的是“同一分割实例的局部路径”，不证明遮挡处的物理身份。

## 证据与最终层独立复核

`visual_path_validation.evidence`保存schema 1的候选/实例ID、原图像素SHA256、完整相机系米制路径，以及带原图左上角坐标的局部实例/其他实例0/1行字符串。局部补丁围绕验收段扩展8像素，保留分叉和线宽检查上下文。

`grasp_policy`增加 `local_path_validation_schema_version`、`image_rgb_sha256`、`instance_mask_evidence`和逐候选 `visual_readiness_by_candidate`。实例证据用原图尺寸与前景游程 `[start,length]`保存所有实例，按C顺序展开，不新增可执行脚本。

`build_motion_spec()`不信任上游passed、空阻断列表或无歧义布尔值：

- 重新执行局部路径验收；缺字段、格式损坏或上游未通过均拒绝执行。
- 核对候选/实例列表唯一且完整，并从所有实例游程重建补丁，确认局部证据没有漏掉其他实例。
- 核对图像指纹及人工确认来源；模型预测、源标记不一致不能升级为人工确认。
- 手动提示分割须有单独的线身复核记录，核对当前原图、实例ID和完整实例掩码的SHA256；Shift交界确认不能替代V线身复核，见[手动纠正schema](MANUAL_WIRE_CORRECTION.md#新增数据字段)。
- 用保存的K/D、平面和实测高度再次投影原路径，再用原变换函数复算基座、法兰及预抓取矩阵，核对与目标一致。
- 原有平面、工具、末端偏置和运行时真实反馈/使能/夹爪门槛继续有效。名义工具授权不绕过本视觉门控。

复核结果在 `motion_spec.visual_path_review`；所有拒绝原因进入 `motion_spec.execution_blockers`。`visual_path_*`阻断允许保留历史或失败目标的**只规划分析**，但 `execution_allowed=false`。其他原有硬规划阻断保持原行为。

JSON内的证据是数据一致性检查，不是防篡改签名。它不能证明现场未移动、相机标定绝对正确、像素掩码真的是线材，也不能代替碰撞检查、反馈时效与实体安全检查。

## 主要拒绝原因

| 原因 | 处理方向 |
|---|---|
| `visual_path_validation_missing` | 旧结果或无新证据，按兼容性报告重新采集和识别 |
| `visual_path_instance_missing/mismatch` | 核对实例分离和候选所属，不手工改ID迎合检查 |
| `visual_path_metric_path_missing` | 缺平面、高度或可用三维路径，先处理定位输入 |
| `visual_path_pixel_path_disconnected/outside_instance_mask/instance_disconnected` | 局部路径跳点、出掩码或跨缺口，改善识别/场景 |
| `visual_path_branch_detected/competing_instance/instance_ambiguous` | 分开交叉/接触线材或取得清晰独立掩码 |
| `visual_path_width_abnormal/direction_discontinuity/metric_direction_discontinuity` | 检查线宽、线身方向及局部几何，不放宽阈值强行通过 |
| `visual_path_boundary_not_manually_verified/boundary_source_mismatch` | 重新进行本次人工交界确认 |
| `visual_path_manual_mask_review_required` | 手动纠正后核对当前掩码并按V复核；修改提示后须重新复核 |
| `visual_path_instance_mask_evidence_mismatch` | 局部补丁与原始实例不一致，重新生成完整证据 |
| `visual_path_coordinate_evidence_invalid_or_missing` | 投影或坐标来源不完整/不一致，重新识别转换 |
| `visual_path_evidence_invalid/verification_stages_incomplete/upstream_validation_not_passed` | 证据格式或验证阶段不完整，不手工填passed |

本次兼容性变化及新数据复核步骤见 [兼容性报告](VISUAL_PATH_COMPATIBILITY.md)。
