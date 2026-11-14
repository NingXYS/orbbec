# 视觉坐标到官方 MoveIt 抓取适配

## 结论与边界

现有线材识别结果中的 `T_base_flange_pregrasp`、`T_base_flange_grasp` 已可转换为官方 MoveIt `PoseConstraint`，通过 `/piper_x/move_action` 完成规划。执行顺序为：

1. 官方夹爪开到预抓取宽度；
2. MoveIt 到预抓取位；
3. MoveIt 沿夹持接近轴下降到抓取位；
4. 通过官方门控和 `gripper_controller/follow_joint_trajectory` 闭合夹爪；
5. MoveIt 抬升回预抓取位。

入口默认只规划。只有显式增加 `--execute` 才会发送真机动作。

可选固定装配转移：增加 `--assembly-target configs/wire_assembly_fixed_20261106.json`，抬升到位后保持夹持，先到直线入口，再沿 base_link -Y 直线30 mm到记录的六轴目标，到点不自动松爪。不加 `--execute` 时规划预抓取、抓取、抬升、装配入口和装配末段五段。具体测试命令、到位容差及现场验证范围见 [固定装配目标](ASSEMBLY_FIXED_TARGET.md#接入抓取流程操作者测试路径)。

这一层不替代视觉定位，也不用旧照片、旧棋盘平面或像素坐标推断三维目标。目标不合格时直接退出，不调用 MoveIt。

路径以 `graspnet_orbbec` 为工作目录；完整启动步骤见 [真机启动手册](REAL_GRASP_STARTUP.md)。

## 输入门槛

识别结果必须同时满足：

- 指定的 `grasp_plans[]` 状态为 `base_frame_preview_ready`；
- 已人工确认金属/绝缘交界；
- 当前RGB棋盘平面或安装关系未变的固定桌面平面可用，线材中心相对棋盘高度及局部平行关系已重新核实；
- `surface_projection_only=false`；
- `tcp_compensation_applied=true`；
- 具有 `T_base_flange_grasp` 和 `T_base_flange_pregrasp`；
- 预抓取点位于夹持点后方 5–200 mm，且与夹持姿态一致；
- 执行时 `readiness_blockers` 必须为空。

`nominal_finger_tip_reference_not_verified_grasp_center` 默认只允许用于规划预览。用户明确接受 URDF 名义工具参数时，可增加 `--allow-nominal-urdf-tool` 授权真机执行；报告会保留该授权和警告，不会把名义参数伪造成实测 TCP。

## 完整启动流程

实际终端操作、常见错误、保存与退出顺序见：

[Piper X 视觉抓取真机启动手册](REAL_GRASP_STARTUP.md)

日常操作只使用本次新快照及人工确认的候选；启动、环境配置、只规划和可选执行命令统一见上述主手册。旧快照与特定候选的说明移至[历史冻结结果复核](REAL_GRASP_HISTORY.md)。执行许可须查看规划报告的 `motion_spec.execution_allowed` 和 `motion_spec.execution_blockers`，不能仅凭控制器或规划返回成功执行。

报告默认写到新的 `output/wire_grasp/moveit_<timestamp>_<candidate>/report.json`。

## 坐标与工具合同

- 视觉输出是 `base_link -> link6/flange` 位姿。
- 官方当前 launch 的 `tcp_offset` 为零，因此 MoveIt 规划末端 `tcp_link` 与 `link6` 重合。
- 程序启动规划前检查 `link6 -> tcp_link` 的位置和旋转误差；若启动了非零 `tcp_offset`，会拒绝对视觉法兰目标直接规划。
- 工具补偿只消费视觉结果中的一次补偿，不在适配层再次加 141.5/142.5 mm。
- `--base-z-offset-mm` 用于修正机器人 `base_link` 原点相对桌面的固定 Z 差；桌面低于 `base_link` 时填负值，当前现场暂定 `-12 mm`。
- 已保存固定桌面平面 `configs/support_plane_piper_x_02.json`；配合 `--fixed-plane` 时，新采图不要求棋盘入镜。底座、桌面或相机安装移动后必须重新保存。
- 2026-11-03 现场游标卡尺测量：法兰到夹爪末端 `142.5 mm`，预抓取姿态下夹爪末端到桌面约 `88 mm`；与真实法兰反馈对照后，底座 Z 修正取 `-12 mm`，夹持中心采用 `135 mm`。
- 本次对应的重新识别结果：`output/wire_grasp/real_20261103_164546_base-12_grasp135_01`。其夹持中心 Z 为 `30.55 mm`，法兰夹持目标 Z 为 `165.50 mm`。

## 真实执行保护

`--execute` 还会检查：

- 真实关节反馈完整且新鲜；
- 三台控制器和 `/piper_x/move_action` 已就绪；
- `arm_status`、`motor_status`、`gripper_status` 均为 OK；
- 机械臂和夹爪已使能，且不处于拖动记录状态；
- 每个 MoveIt 阶段成功后，`feedback/tcp_pose` 确实到达目标容差；
- 夹爪开口反馈确实达到开/闭目标；
- 夹爪轨迹通过官方 `gripper_controller/follow_joint_trajectory` 执行；期间只通过 `/piper_x/control_enable` 门控短暂开放。

规划目标容差默认仍为 `3 mm / 0.05 rad`；执行后的 TCP 实际到位验收使用独立的 `--arrival-position-tolerance-m 0.01` 和 `--arrival-orientation-tolerance-rad 0.10`。这样不会因为控制器终点存在少量姿态残差，把已经完成的夹持和抬升误报成整次抓取失败。

本程序不会自动使能机械臂、清除急停、回零、关闭 CAN 或调用自研逆解。

## 仍需现场确认

- MoveIt 规划场景目前没有由本适配层自动加入桌面、支架、线材和相机外形；碰撞规划只使用现有 MoveIt scene。
- 手眼 PARK 解的固定标定残差约 5.4 mm，不是独立绝对精度验收。执行容差 3 mm 表示规划/到位门槛，不代表视觉物理精度优于 5.4 mm。
- 当前 2026-11-03 的 `vision_moveit_20261103_recognition_01` 没有 `base_frame_preview_ready` 目标，因此不会产生运动。需要重新布置棋盘、确认唯一端子和测量高度。

## 局部路径门控（2026-11-04）

执行资格增加独立的[视觉路径复核](VISUAL_PATH_VALIDATION.md)。旧JSON可分析，但缺少新证据时只允许规划；名义工具授权不绕过此门控，历史目标资格变化见[兼容性报告](VISUAL_PATH_COMPATIBILITY.md)。

装配入口及最终点采用同时稳定验收：2 mm / 0.01 rad / 各轴最多0.005 rad，持续0.5秒。抓取闭合轴与线材切线范围收紧到85–90°（优先90°），闭合前还会检查真实姿态对应夹角。原抓取阶段到位容差不变。
