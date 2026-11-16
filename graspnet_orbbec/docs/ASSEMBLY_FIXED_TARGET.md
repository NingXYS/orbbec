# 线材固定装配目标

记录时间：2026-11-06T11:31:19.993318+08:00。当前目标：[wire_assembly_fixed_20261106.json](../configs/wire_assembly_fixed_20261106.json)。本次使用用户重新调整后的状态，[上一版目标备份](../configs/wire_assembly_fixed_20261106_superseded_103514.json)保持可追溯。下方历史验证和对比只适用于当时目标。

本次仅被动接收 CAN，未发送控制帧、使能或运动。两次各3秒采样得到 600 / 600 组完整法兰反馈，采样末端关节和法兰值完全一致，不据此宣称整个区间连续静止。原始文件见配置 observation。

## 自己重新记录装配位置

当前实际生效的数据始终以 `configs/wire_assembly_fixed_20261106.json` 为准；本页数值表与历史报告是带时间的记录，运行记录命令后以终端输出和新生成的 `target_record.json` 为准。

1. 等待当前动作结束，用示教把机械臂摆到新的**最终装配点**，同时调好位置和姿态；不是摆到提前30 mm的入口。保持机械臂静止，记录时不要拖动。
2. 在终端运行只读记录命令（约6秒）：

```bash
cd ~/orbbec/graspnet_orbbec
bash scripts/record_assembly_target.sh
```

看到“已更新”才表示保存成功。命令被动读取两次CAN状态，同步重算六轴、法兰、四元数与135 mm夹持TCP；检查采样一致性、关节限位和工具合同。旧配置自动保存为同目录下的 `wire_assembly_fixed_20261106_backup_<时间>.json`，原始两次反馈和新目标副本写入终端打印的 `output/wire_grasp/assembly_record_<时间>/`。

记录失败不会更新目标。两次采样偏差超过0.5 mm、0.1°或任一关节0.1°时拒绝保存。该检查不是连续静止证明。记录命令不发送使能、运动或夹爪命令，不自动退出示教。

3. 退出示教，按日常操作手册核验真实反馈。使用本次新采集、新确认的 `recognition` 和 `candidate_id`，先只规划：

```bash
bash scripts/run_moveit_wire_grasp.sh \
  --recognition "${recognition:?请先设置本次识别文件}" \
  --candidate-id "${candidate_id:?请先设置本次候选ID}" \
  --assembly-target configs/wire_assembly_fixed_20261106.json
```

4. 检查新目标的五段规划和现场路径，再使用[操作手册第7步](REAL_GRASP_STARTUP.md#7-终端b人工确认后可选执行)完整执行命令。每次启动都会读取新配置，不需要重启官方驱动。不要在抓取程序正在运行时修改目标；会话中的文件变化会触发中断。

固定目标更新会清除旧的规划和控制状态验证结论。现有末段 `base_link -Y` 直线30 mm、10 mm/s上限及85–90°抓取闭合夹角不变。当前入口仍以记录六轴为最终目标，不能只手改JSON中的XYZ或RPY，否则它们会与关节目标不一致。

## 当前固定值

坐标系为 `base_link`，位置单位mm、姿态RPY单位°，旋转顺序 `Rz @ Ry @ Rx`。

- 六轴角度：[-75.28, 96.263, -88.088, 74.344, 2.735, 21.752]。
- 法兰XYZ：[91.48, -354.848, 299.074]。
- 135 mm夹持中心TCP XYZ：[92.645193, -384.646862, 167.409008]。
- 两者RPY：[-167.41, 2.105, 172.9]。

TCP由真实法兰反馈乘以当前135 mm工具偏置计算，不是独立实测TCP。没有叠加桌面 -12 mm 修正。官方零偏移 `tcp_link` 对应法兰，不能把夹持TCP坐标直接送给该零偏移末端。

## 当前状态与流程

目标已接入；六轴均在官方URDF限位内，joint4=74.344000°。采样时 `ctrl_mode=2`、`teach_status=2`，当前仍需退出示教，再核验实际控制状态；本次不切换模式。

原参数保持：夹取并抬升 → 装配直线入口 → 沿 `base_link -Y` 直线30 mm到新目标，姿态固定，速度上限10 mm/s，到位后保持夹持。原执行命令及目标文件路径不变，每次启动重新加载配置。

新目标的规划检查独立记录；下方旧目标验证不自动沿用。现场负载及未建模障碍物路径仍需操作者测试。

## 本次新目标验证

[新目标五段只规划报告](../output/wire_grasp/assembly_reteach_y30_20261106_01/report.json)：预抓取、抓取、抬升、直线入口和末段均成功。末段Y向行程 30.000137 mm，16 个正解采样，横向最大偏差 0.001732 mm，旧报告按相邻采样点均速记录 10.000 mm/s。六轴目标保持本次记录，未发送运动请求。目标/角度数据专项19项测试通过。

此结果只验证新目标下的模型规划。采样时仍为示教模式，真实执行前须退出示教并通过已有入口实时检查。

## 历史零位到目标的只规划验证（2026-11-06）

当前零位到装配目标规划成功，两轮结果均通过；最终报告为
[assembly_validation_20261106_02/report.json](../output/wire_grasp/assembly_validation_20261106_02/report.json)。
报告保留目标输入副本与哈希、运行模型限位与链接、规划场景、完整轨迹及检查结果。

- 起点六轴均为 0°，夹爪当前开口 48.9 mm，规划不改变夹爪。
- 目标与起点的状态有效性检查均通过，无接触报告。
- 目标六轴的官方模型法兰正解与记录值相差约 0.263 mm / 0.00199°。
- 轨迹共 78 点，轨迹预计时长约 7.668 秒（不是实测运动时长）；最大关节步长 0.02 rad 细分后检查 139 个状态，均通过碰撞及关节限位检查。
- 当前场景无世界障碍物、无附着线材、无 octomap；机器人模型不包含相机/安装支架。因此该检查只覆盖现有模型，不能代表桌面、装配支架、相机或携带线材的现场避碰通过。
- 没有发送执行、使能或夹爪命令。`execution_allowed=false` 保留，尚未验证负载转移；软件现已通过下述可选参数接入。

复核入口只使用 `GetMotionPlan`、`GetStateValidity`、`GetPositionFK`、`GetPlanningScene` 和读取参数服务，没有执行接口。保持官方链路运行，在项目目录中加载环境后使用新的报告路径：

```bash
cd ~/orbbec/graspnet_orbbec
source /opt/ros/humble/setup.bash
source ../agilex_official_ws/install_humble/setup.bash
export ROS_DOMAIN_ID=93 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/usr/bin/python3 scripts/validate_assembly_target.py \
  --target configs/wire_assembly_fixed_20261106.json \
  --report output/wire_grasp/assembly_validation_new/report.json
```

脚本拒绝覆盖已有报告。`planned_only` 仅表示本次起点、目标与现有场景下可规划，不会更改固定目标或自动执行。

## 接入抓取流程：操作者测试路径

在本次有效识别、V/Shift交界确认和真实反馈检查完成后，对原抓取命令追加 `--assembly-target`。请使用本次新采集的 `recognition` 和对应 `candidate_id`，不要使用本节回归报告中的历史识别文件。

先只规划五段：预抓取 → 抓取 → 抬升 → 直线入口 → 沿 base_link -Y 直线到装配目标。此模式抬升和装配的虚拟起始状态使用闭合夹爪，不发送夹爪动作：

```bash
bash scripts/run_moveit_wire_grasp.sh \
  --recognition "${recognition:?请先设置本次识别文件}" \
  --candidate-id "${candidate_id:?请先设置本次候选ID}" \
  --assembly-target configs/wire_assembly_fixed_20261106.json
```

操作者执行测试（会从打开夹爪开始完整抓取，再转移到装配点）：

```bash
bash scripts/run_moveit_wire_grasp.sh \
  --recognition "${recognition:?请先设置本次识别文件}" \
  --candidate-id "${candidate_id:?请先设置本次候选ID}" \
  --assembly-target configs/wire_assembly_fixed_20261106.json \
  --execute
```

执行顺序为：打开夹爪 → 预抓取 → 到夹持位并确认85–90°闭合夹角 → 闭合 → 抬升并验收到位 → 到装配直线入口 → 沿 base_link -Y 直线30 mm到记录的六轴装配目标 → 稳定验收关节和末端位姿。装配前后检查闭合反馈（开口不超过10 mm），保持夹爪闭合，不自动松爪、插入或返回。闭合反馈本身不独立证明夹到线材；现场需观察。

六轴入口规划容差为 0.001 rad。装配入口和最终点使用更严格的同时稳定验收：位置不超过2 mm、姿态不超过0.01 rad（约0.573°）、各轴偏差不超过 `min(--arrival-joint-tolerance-rad, 0.005)` rad（最多约0.286°），持续0.5秒。原来的10 mm / 0.10 rad仍用于预抓取、抓取和抬升，不用于新装配末段验收。这些阈值不等同于独立装配精度保证。任一阶段失败即结束，不继续转移，也不自动打开夹爪；不要在仍夹住线材时直接重跑整条命令。

入口在打开夹爪之前检查装配工具文件哈希、运行模型关节限位和关节正解/法兰一致性；运行中检查目标文件未变化，转移前复查真实诊断、夹爪状态及运行模型。既有视觉来源和安全门控仍生效。

`planning_contract.execution_allowed=false` 保留表示完整现场路径尚未验收，不能当作实时许可。操作者显式指定装配配置并加 `--execute` 是本次路径测试入口，不要求把该字段手工改成 true；也不解除视觉、模型或真实反馈阻断。

历史四段实际 MoveIt 只规划回归已通过，见 [integration report](../output/wire_grasp/assembly_integration_plan_20261106_01/report.json)。该回归采用历史识别样本，样本存在 `visual_path_manual_mask_review_required`，仅用于验证四段规划调用，不可用于执行。全程 `physical_motion_performed=false`、`motion_commands_attempted=[]`。当前现场障碍物及负载建模限制仍如上所述。

软件验证：全量离线测试 `490 passed`；系统 Humble 环境下装配流程专项 `10 passed`。覆盖默认流程兼容、四段只规划、前序失败中断、目标变化阻断及装配到位失败不松爪。

## 当前末段：base_link -Y直线30 mm

当前配置 `final_approach` 为：`frame_id=base_link`、`axis=y`、`direction=-1`、`distance_m=0.03`、`max_speed_m_s=0.01`。入口位于固定目标的 +Y 侧30 mm，姿态与目标一致。

重新示教时从当前项目同级的 `agilex_official_ws` 加载官方 Piper X 模型核对限位；旧目标模板中的历史绝对路径仅作记录。

程序先以记录六轴为起点调用 `GetCartesianPath`，反向求到入口的完整避碰路径，再反转时序及速度符号，保留最终六轴构型。一般转移使用关节目标到入口；最后30 mm使用官方 `ExecuteTrajectory` 执行该笛卡尔轨迹，不使用普通关节规划代替直线。

当前限速按控制器轨迹点的位置、速度和加速度推导各插值区间的速度上界，并使用运行模型中各转轴到法兰的最大力臂。报告中的 `cartesian_check.max_speed_m_s` 是缩放并复核后的保守上界，`speed_check=spline_upper_bound` 标识该算法。保守限速可能延长末段时间。

仅接受100%完整率、合法递增时间戳、有限六轴与导数、单步关节变化不超过0.1 rad的轨迹；逐点官方正解复核固定X/Z、固定姿态、Y单调及长度。时间缩放只减速，规划速度上限10 mm/s。执行前逐点以当前夹爪开口复查碰撞，并核对入口关节差不超过0.005 rad。该检查依赖当前规划场景，不能替代未建模的桌面、支架及线材现场检查。

固定目标可由用户重新示教更新。路径以当前记录六轴的官方正解生成；末段终点关节值仍逐项等于当前示教记录。下方约0.263 mm模型差异为上一版目标的验证结果。

只规划验证报告：[assembly_y30_integration_20261106_01](../output/wire_grasp/assembly_y30_integration_20261106_01/report.json)。五段均成功，末段15个正解采样、行程30.000062 mm、横向最大误差0.001729 mm、姿态最大变化约0.000026°；没有发送运动。样本用于软件规划回归，不作为后续真实执行的新场景来源。

## 2026-11-06 当前反馈与原目标对比

[对比记录](../output/wire_grasp/assembly_post_execution_comparison_20261106.json)及[本次被动CAN反馈](../output/wire_grasp/assembly_state_20261106_post_execution_compare.json)。该检查未移动机械臂。

| 项目 | 保存目标 | 本次稳定反馈 |
|---|---|---|
| 法兰XYZ（mm） | 101.850 / -344.965 / 292.566 | 102.286 / -344.768 / 291.818 |
| 135 mm夹持TCP（mm） | 91.591 / -374.647 / 161.270 | 92.019 / -374.273 / 160.482 |
| RPY（°） | -166.774 / -2.476 / 171.349 | -166.854 / -2.515 / 171.454 |

法兰位置差0.888 mm，姿态差0.135°，六轴最大差0.125°。上次执行报告曾在第6轴仍差约1.131°时通过旧0.02 rad验收；随后反馈进一步稳定。本次改为上述同时稳定判定，避免过早报告装配到位，不据此宣称物理TCP或线材端子计量精度已验收。

## 抓取闭合夹角85–90°

这里是夹爪闭合轴与局部线材切线的锐角，正负轴等价；不是接近方向对桌面或竖直方向的夹角。历史策略为70–90°，当前执行入口收紧为85–90°，生成目标仍优先90°。识别结果和运动报告记录范围；入口对本次真实来源的目标重算夹角，执行时在闭合前以真实末端姿态再检查一次。不改写旧识别文件、不删除视觉或来源阻断，现有90°目标符合新范围。

本次Y轴末段与夹角收紧后验证：全量离线测试 `510 passed`，系统 Humble 装配流程专项 `19 passed`；五段实际只规划通过。未执行新增直线或任何真机动作。
