# 六项复核问题的修复与兼容性

日期：2026-11-04，Asia/Taipei。对应 `output/tasks1_8_audit_20261104_624be198/AUDIT_REPORT.md` 的F1–F6。仅经离线与模拟控制入口验证，没有使能、规划或执行真机动作。

## F1：最终来源门控

- 新增 `arm/source_validation.py`。最终层重读RGB/深度、相机参数、帧元数据、反馈和pair，核对文件及解码像素指纹、冻结标定。
- 用配对反馈和入口 `--binding` 指定的标定重建相机变换；重新加载工具和固定平面文件、核对指纹，复算平面和目标变换。不从recognition自行选择标定，也不只靠JSON里两个相同hash获准执行。
- `grasp_policy.source_provenance` schema v1保存快照、7个必要文件指纹、工具/平面文件合同、安装/支撑确认及实测高度；传播 `data_origin`。没有证据的旧结果仍可分析/规划。
- `motion_spec.source_review`记录独立结果；`execution_source_missing`或`execution_source_invalid`阻断执行。名义工具授权不能绕过来源或局部路径门控。
- 首次执行资格失败在ROS初始化前退出；每个动作阶段之前重验来源及recognition文件是否改变。

新采集的pair写 `capture_schema_version=2`、`data_origin=physical_camera`及 `capture_kind`。序列帧另带 `sequence_manifest=../../manifest.json`，最终层重验父manifest与唯一归属，不能用子目录绕过失败序列。

**旧快照/recognition仍可离线分析，但不能直接获得新增来源门控的执行许可。** 需用修复后的原有入口重采，再重新确认安装、实测高度、工具及本次交界。禁止手补版本号、来源标签或清空阻断。安装改变后，固定平面和手眼仍须重新核实。

指纹用于检查错配和文件变化，不是防止本机恶意改写全部数据的数字签名，也不证明拍照后物理场景一直未变。

## F2：重复假设与人工确认

只对无拓扑歧义、路径长度比≥95%、较小掩码被包含≥98%、中心线95%距离≤0.5px、端点/最远距离≤3px的假设去重。选择评分较高的一份，不合并像素。

全部原始source masks和显示OR保留；`automation.duplicate_wire_hypotheses`记录去重来源。分叉、交叉、断线、移位平行线及长度明显不同的假设不通过此规则。修复单根线仅线宽不同就阻断人工确认的反例，**不等于解决所有噪声和实例质量问题**；无法证明等价的假设继续拒绝。

## F3：局部静止检查

原有全局光流/亮度检查之外，增加完整ROI内16×16滑动窗口检查：灰度变化>20的比例>12%，且局部平均变化>3时拒绝；ROI小于16px时证据不足。

修复线材移动8px被背景角点/全图平均掩盖的反例。遮挡、低于成像分辨率或曝光间运动仍不能完全排除。ROI必须覆盖目标，不能选纯背景绕过；像素阈值不等于毫米精度。

## F4：逐轴反馈新鲜度

`arm/feedback_safety.py`保存每轴自己的接收时间。夹爪不刷新六轴年龄；NaN、重复名称或位置数量错误使对应旧值失效。启动、规划/动作入口及夹爪操作检查必需关节，夹爪到位也要求新鲜反馈。未改变CAN、使能或轨迹参数。

## F5：动作报告schema v2

| 字段 | 含义 |
|---|---|
| `motion_commands_attempted` | 已进入的动作尝试阶段，不代表已被控制器接受或物理运动成功 |
| `motion_evidence` | `no_motion_requested`、`unknown_after_attempt`或`feedback_change_observed` |
| `physical_motion_performed` | 未尝试时`false`；尝试后证据不足时`null`；观察到关节变化时`true` |

**不能把null当false。** 失败、超时或部分执行后结合事件、反馈与现场状态处理，不能因没有成功结果就声称未运动。只规划仍为false；观察到运动也不等于抓取成功。

## F6：manifest格式损坏

数组、null、错误嵌套类型及缺字段转为验证拒绝；离线分析保存 `manual_confirmation_required` 的 `analysis.json`。不以吞掉所有运行时异常来掩盖模型/GPU故障。

## 验证与当前边界

新增31项回归覆盖来源成功/篡改、模拟/旧数据、文件变化、名义工具漏阻断、ROS初始化前拒绝、执行中输入变化、逐轴反馈、部分失败、重复掩码、局部运动及异常manifest。完整离线测试 **442项通过**。

```bash
cd ~/orbbec/graspnet_orbbec
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda/bin/python -m pytest -q
```

原真实五帧通过更新后的只读验证；离线重放输出在 `output/tasks1_8_fixes_20261104/`。真实YOLO仍未检出端子、跨帧实例连续性仍未通过，保持人工复核。没有改权重、14mm算法、手眼/TCP数值，不能据此宣布真机抓取通过验收。
