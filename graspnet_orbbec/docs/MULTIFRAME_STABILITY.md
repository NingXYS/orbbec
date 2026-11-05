# 离线多帧YOLO/SAM2稳定性验证

输入是[任务7静止采集](STATIONARY_RGB_SEQUENCE.md)的完整序列根目录。此工具不打开相机、不连接CAN、不调用MoveIt；不修改YOLO权重、SAM2算法、14mm策略或机械臂执行逻辑。

## 运行

工作目录 `~/orbbec/graspnet_orbbec`，使用现有项目环境：

```bash
PYTHONNOUSERSITE=1 .conda/bin/python scripts/analyze_wire_sequence.py \
  --sequence /path/to/real_sequence \
  --out output/multiframe/my_review
```

输出目录必须尚不存在。默认拒绝模拟数据；仅测试时可显式添加 `--allow-simulated`，报告仍保留 `data_origin=simulated`。这不会绕过manifest、图片、时间戳或配对验证。

默认参考帧为manifest第一张。可用 `--reference-frame FRAME_ID` 明确选其他帧；ID必须来自该manifest。报告记录原始RGB、时间戳、图片/配对/标定指纹及像素、位姿的同一参考帧ID。参考帧无需包含所有目标：若目标只在其他帧出现，会报告 `absent_in_reference`，要求人工重新选择参考帧。

`--config`指定现有视觉配置。实验稳定性配置单独由 `--stability-config` 指定，默认为 [multiframe_experimental.yaml](../configs/multiframe_experimental.yaml)。检出率0.7、中心4px、边界5px等**只是可配置实验初值，不是安全标准或定位精度**，必须通过真实标注数据评估。

如需参考帧的离线交界/三维计算，可添加 `--geometry-preview` 及现有手眼参数：`--center-height-mm`、`--support-relation-confirmed`、`--same-installation-confirmed`、`--fixed-plane`、`--base-z-offset-mm`、`--tool`。`--boundary-point X Y`只允许填写这张参考原图上人工确认的交界。安装、平面、高度和工具适用条件仍按[日常手册](REAL_GRASP_STARTUP.md)核实，不默认44mm。

三维入口只接受物理采集序列，使用该参考帧自身 `capture/`、`robot.json`、`pair.json`，推理后再次验证全序列与manifest未变化。统计框、其他帧的像素和平均位姿都不传入手眼计算。几何结果始终加 `OFFLINE_MULTIFRAME_ANALYSIS_ONLY` 阻断，`planning_allowed=false`、`executable=false`。本工具没有执行选项。

## 处理方式

1. 首先重验任务7的全序列合同。帧缺失、图像篡改、时间戳异常、配对不符或采集静止验证不通过时，不开始推理，保存人工复核报告。
2. 对每张完整RGB独立运行一次YOLO端子检测，保存原框及原始分割掩码。不把一帧预测复制到其他帧。检测异常单独记录，统计分母仍为全部请求帧。
3. 以参考帧端子为初始轨迹，其他帧新目标建立独立轨迹。类别、中心距离、四边变化、尺寸比例共同限制候选关系；只有两侧各自唯一的匹配才接受。同一框竞争两个目标或同一目标有多个候选时返回歧义。轨迹锚框保持原观测，不漂移跟随、不平均未匹配框；移动过大可能形成新轨迹并要求人工确认。
4. 每轨迹统计匹配检出比例、中心标准差与相对锚框最大位移、四边标准差/范围/最大偏差、端子掩码IoU和最大轮廓距离、歧义帧及拒绝原因。检出比例表示稳定性，不能证明端子类别正确；持续误检也可能稳定。
5. 参考帧运行一次完整SAM2，复用**同一帧**的YOLO预测，支持现有ROI坐标恢复；不重复检测。端子仍需经过现有具体线材实例关联。自动交界强制关闭，保留Shift+左键人工确认流程。
6. 默认再选一张挑战帧运行SAM2，依据框位移和灰度变化排序。`wire_check_frames`支持0–2，最多检查参考帧以外两张；0意味着未验证跨帧线材，必须人工复核。每张仍独立分实例，按完整原图中的路径最大偏差、端点、方向、长度和线宽验收唯一对应关系。断线、分叉、接触歧义、实例缺失或竞争均拒绝，不合并不同帧掩码。

不能建立实例连续性、端子漏检过多、框移动、边界不稳定或证据冲突时，状态为 `manual_confirmation_required`。没有上述问题也只返回 `experimental_consistent_manual_boundary_required`，仍需要人工确认交界，绝不表示可以执行。

任务7会先拒绝明显的移动或光照变化；任务8不放宽采集验证。细小、局部或被遮挡的运动仍可能漏检。像素阈值不随工作距离自动等价为毫米；未做图像配准，不能把移动相机拍到的场景当作本工具的静止序列。

## 输出与A/B解释

- `analysis.json`：schema v1，逐帧YOLO结果/错误/耗时、轨迹及歧义、参考快照来源、实例复核、实验配置、拒绝原因、A/B摘要。
- `yolo/frame_XXX/terminal_XXX.png`：逐帧模型端子掩码，不是融合结果。
- `reference/recognition.json`及掩码/画布：参考帧自身的现有recognition格式，新增 `automation.multiframe_reference` 来源记录；不写入跨帧平均抓取点。参考分析失败时此目录可能不存在。

A组只看同一参考帧的YOLO/SAM2；B组增加其他帧独立YOLO和少量SAM2连续性证据。两组**共享参考SAM2结果**，用于比较证据增量，不是两套分割算法的精度赛跑。逐帧YOLO与SAM2/实例分析的墙钟耗时分别保存；首帧可能包含加载开销，后续可能为热运行，不能直接声称独立冷启动加速。`accuracy_measured=false`表示未提供独立真值。

## 2026-11-04离线验证

真实USB在主机可见，但没有CAN接口，无法采集符合任务7合同的真实配对序列。未启动相机或机械臂，未伪造反馈。真实多帧识别准确率及实际场景A/B **尚未验收**。

可复现模拟预测A/B及真实模型在合成五帧上的报告位于 `output/multiframe_validation_20261104_task8/`。真实模型实验使用已有权重、CPU四线程及4个正点预算，数据仍明确为合成：YOLO五帧总耗时约2.54s；参考SAM2/实例分析约3.18s，挑战帧约1.39s。未检出有效端子轨迹，正确保留人工复核状态。这些时间仅说明本机该合成实验，不代表默认64点预算或真实场景速度。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .conda/bin/python -m pytest tests/test_multiframe.py -q
```

测试包含稳定目标、顺序互换、漏检、孤立误检、邻近目标竞争、移动、边界/光照变化、时间异常、实例连续性、参考快照对应、模拟三维拒绝、推理期间文件替换及旧ROI兼容。

本次新增38项测试通过，全项目411项通过；[A/B比较报告](../output/multiframe_validation_20261104_task8/AB_COMPARISON.md)明确区分模拟预测与真实模型在合成输入上的结果。已有相机采集、SAM2/YOLO算法及机械臂源码均未改动。
