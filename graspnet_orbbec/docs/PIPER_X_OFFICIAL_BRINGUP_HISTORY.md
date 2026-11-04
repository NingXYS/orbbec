> 2026-11-04 归位说明：本文为2026-11-03历史部署/交接记录，包含不同时点的状态，不能用于判断当前进程或硬件状态。当前操作以 [真机启动手册](REAL_GRASP_STARTUP.md) 为准。正式入口为 `/home/ning/orbbec/bringup/piper_x/start_real_control.sh`；旧入口仅转发。官方工作空间、视觉代码、标定工程未搬动；历史证据仍在原路径。迁移未启动硬件，也未新增抓取验收。

# Piper X 官方 ROS2 Humble / MoveIt2 真机执行

**用户已明确批准开启真实执行；当前官方 RViz 已连接机械臂。**

## 启动与操作

```bash
bash /home/ning/orbbec/bringup/piper_x/start_real_control.sh
```

当前调用官方 `agx_arm_ctrl/start_single_agx_arm_moveit.launch.py`，关键参数：

- `arm_type=piper_x`、`can_port=can0`、`namespace=piper_x`，ROS domain `93`。
- `follow=true`，真实反馈 `/piper_x/feedback/joint_states`。
- `control_topic=control/joint_states`，已接官方驱动，不再使用隔离话题。
- `allow_trajectory_execution=true`、`speed_percent=5`、`fast_mode=false`。
- `auto_control_gate=true`：空闲时关门，官方 JTC 轨迹执行时开门，结束后关门。
- `auto_enable=false`：本次核对六轴本来均已使能，不额外自动使能或恢复。

在 RViz 的 MotionPlanning 面板：

1. Planning Group 选择 **arm**。
2. Start State 保持 **<current>**，设置一个小范围目标。
3. 点击 **Plan**，查看轨迹。
4. 确认轨迹后，由用户点击 **Execute**。这会通过官方驱动控制真实机械臂。

本次只接通执行链路，没有自动发送测试轨迹、回零或夹爪动作。启动时官方 SDK 下发了 5% 速度设置，接口 TX 从430变为431；没有据此宣称已经完成真实轨迹执行验收。

官方 GenericSystem 的 `initial_positions.yaml` 已用本次真实关节姿态初始化；控制器初值与真机反馈已逐关节比较。此快照只对应本次启动姿态，若断电或人工改变姿态，重新启动前需要重新核对并同步初值。

当前连通核对见 [execution_connection.json](/home/ning/orbbec/agilex_official_ws/execution_connection.json)，本次日志见 [official_execution_enabled.log](/home/ning/orbbec/agilex_official_ws/official_execution_enabled.log)。历史只读验收记录保留在下方，其执行关闭参数已被本次授权配置替代。

## 重复目标与 RViz 崩溃处理（2026-11-03）

现场同时存在 PID 18185、18960 两套同命名空间的官方 launch，各自含 MoveGroup、JTC、驱动和门控。与日志中的双目标接收、PREEMPTED、unknown goal/result response 及重复门控调用相符。已确认关节静止后清理旧会话，重启唯一一套；没有执行测试轨迹。

启动入口现使用 `/tmp/piper_x_official_can0.lock` 的 flock 单实例锁，并检查遗留官方驱动。第二次启动实际返回2并显示“会话已在运行”，不会新增节点。真实反馈、控制输出、arm Action 状态发布者均核对为1，见 `/home/ning/orbbec/agilex_official_ws/single_instance_verification.json`。RViz 若退出，不要直接重复启动；先在原启动终端 Ctrl+C，确认原会话退出再启动。

现存 apport core 对应较早的 RViz PID17385，故障 PC 经 ProcMaps 和 ELF 符号定位到 `rclcpp::CallbackGroup::~CallbackGroup()+81`。core 无法恢复完整调用栈，不能证明用户后来 PID18992 的崩溃也只有同一原因。已沿用本机先前验证过的局部驻留规避：官方 `move_group.launch.py` 只对 MoveGroup 预加载 `libmoveit_simple_controller_manager.so`，官方 `moveit_rviz.launch.py` 只对 RViz 预加载 `libmoveit_motion_planning_rviz_plugin_core.so`。控制算法、规划器和模型限位未因此改变。当前已重新启动，长期运行及用户后续执行/退出是否仍崩溃有待复测，不以只读核验代替真实执行验证。

日志保留：`official_single_instance_launch.log`、`rviz_crash_backtrace_host.log`。这些是官方启动兼容修改，不是自研控制后端。

## 官方版本与部署

| 内容 | 版本/位置 |
|---|---|
| ROS2 分支 | `ros2`，远端核对提交 `286633e0d2edb038ee3a1047d9feb7f614a1c076` |
| 官方工作空间 | `/home/ning/orbbec/agilex_official_ws` |
| 源码 | `src/agx_arm_ros` |
| 官方 URDF 子模块 | `680fa925cd00db7e2628563d2edc5540963045de` |
| 官方 SDK | `pyAgxArm 1.0.0`，源码提交 `d90e234ba8e408cfa964b38aa9120b3131fc9f85` |
| 隔离 Python 依赖 | `official_ws/python`，使用系统 Python 3.10 |
| ROS 安装产物 | `official_ws/install_humble` |

`official_ws` 在本目录中是指向英文实际路径的符号链接。首次中文路径编译触发 Humble rosidl 消息生成错误，旧 `build/`、`install/`、`log/` 保留；实际运行使用英文路径下重新编译的 `build_humble/` 和 `install_humble/`。

用户下载的 wheels 已安装到隔离目录，未安装到 Conda 或系统 site-packages。新版 setuptools 84 仅保留在 `python_build_tools`，ROS/colcon 编译继续使用系统 setuptools 59.6，避免 colcon 的 `<80` 版本冲突。未新增系统 apt 安装。

## 本轮验收（官方链路）

详见 [readonly_acceptance.json](/home/ning/orbbec/agilex_official_ws/readonly_acceptance.json) 和 [官方启动日志](/home/ning/orbbec/agilex_official_ws/official_readonly_launch.log)。

- 官方驱动读取的六轴角度约 `[0.411, 0, 0, 36.128, -0.602, 0]°`。
- `base_link → flange_link` TF 已读取；规划组 `arm` 已由 RViz 加载。
- `arm_controller`、`gripper_controller`、`joint_state_broadcaster` 全部 `active`。
- `/piper_x/arm_controller/follow_joint_trajectory` 为官方 JTC Action。
- 使用真实当前状态请求 J1 +0.5° 的 **GetMotionPlan**，返回 `SUCCESS(1)`，5 个轨迹点；没有调用 Execute 或发送轨迹 Action。
- `auto_enable=false`、`control_enabled=false`、`allow_trajectory_execution=false` 已通过 ROS 参数服务核对。
- 本轮启动前后接口 TX 包计数均为 `430`，没有新增应用发送；不据此宣称物理层无 ACK。

官方当前 ros2_control 使用 `mock_components/GenericSystem` 生成控制话题，再由官方驱动接收；`follow=true` 让 MoveIt/RViz 使用真实反馈。官方 Mock JTC 的成功不能单独证明机械臂真实到位。下一阶段获准后才恢复官方控制话题并验收真实执行。

## 与官方源码的差异

官方 MoveIt launch、模型、JTC 配置和门控算法均保留。有以下可审阅差异：

1. `src/agx_arm_ctrl/agx_arm_ctrl/agx_arm_ctrl_single_node.py`：反馈恢复分支从 `if self.enable_flag` 改为 `if self.auto_enable and self.enable_flag`，附一行中文说明。离线复现证明原版即使 `auto_enable=false` 仍会调用恢复使能；此修改使只读参数在断流恢复后仍有效，没有替换驱动架构。
2. `src/agx_arm_moveit/scripts/agx_arm_control_gate`：仅补可执行位。官方 Git 文件模式为 `100644`，symlink-install 原样保留后，启动报 `executable ... not found on the libexec directory`；内容未改。

3. `src/agx_arm_moveit/config/initial_positions.yaml`：按本次真机当前关节初始化官方 GenericSystem，避免以全零作为控制器启动姿态。

可运行 `git -C /home/ning/orbbec/agilex_official_ws/src/agx_arm_ros diff` 查看。没有 Git 提交或推送。首次启动错误保留在 `official_readonly_launch_attempt01.log`。

历史自研的 Action、CAN 编解码、反馈发布和启动配置与官方现成功能重复，已退出当前链路。原仿真、相机碰撞盒/工装场景、手眼数据、自研代码和历史证据均保留；本轮没有将名义工装场景冒充真机实测环境，也没有删除事故锁。

---

<details>
<summary>历史自研/只读阶段记录（已停用，不作为当前启动说明）</summary>

# Piper X 真机反馈接入与规划诊断

**已接收到真机自然反馈；尚未实现可验收的真机轨迹执行。当前不可使能或运动。**

2026-11-03，用户确认真机已连接、上次异常未处理，并在本轮手动启用了 `can0`。本轮随后只接收已经启用接口的自然反馈，没有改接口、查询参数、启动 SDK/官方驱动、使能、复位、清急停或控制夹爪。

## 已确认的配置

本机已经有官方 `agx_arm_moveit`，含 `piper_x` 支持，安装于：

`/home/ning/文档/ChatGPT/视觉/piper_x_phase1_20261031/offline_ws/install/agx_arm_moveit`

厂家源码参考位于 `piper_x_migration_audit_20261031/vendor_reference/agx_arm_ros`，固定提交 `286633e0d2edb038ee3a1047d9feb7f614a1c076`。

- [agx_arm_sim/Moveit2](https://github.com/agilexrobotics/agx_arm_sim/tree/master/Moveit2)：各型号仿真规划配置。
- [agx_arm_ros 真机 MoveIt 文档](https://github.com/agilexrobotics/agx_arm_ros/blob/ros2/src/agx_arm_moveit/README.md)：真机接口已存在，官方入口为 `start_single_agx_arm_moveit.launch.py`。

官方当前链路为 `MoveIt → Mock JTC → control/joint_states → agx_arm_ctrl → SDK → CAN`，MoveIt 另订阅真实 `feedback/joint_states`。这说明并非一定要编写新的 ros2_control C++ 硬件插件；但 Mock JTC 成功不等于真实到位。现有官方驱动的自动使能、反馈补零、恢复、停止与真实完成判据风险仍未修复。本项目不把该入口直接接到真机。

## 本轮真实观察

接收窗口约为本地时间 **2026-11-03 13:50:03**，实际收集完整所需分组后因无效状态提前结束。

| 项目 | 本轮自然反馈 |
|---|---|
| 六轴角度（度） | `[-1.953, -2.326, +2.570, +36.128, 0.000, +2.507]` |
| J2 厂家 URDF 范围 | `[0, 约180]°`，当前低于下限 |
| J3 厂家 URDF 范围 | `[约−170, 0]°`，当前高于上限 |
| 控制模式 | `ctrl_mode=0`，不满足本接口要求的 CAN 控制模式 `1` |
| 状态/错误 | `arm_status=0, teach_status=0, motion_status=0, err_code=0` |
| 六轴驱动使能位 | 六组均为 `false`；本轮没有更改 |
| 夹爪 | `mode=0`（行程），宽度约 `0.101 m`，力反馈 `0 N`，状态 `0` |
| 自然反馈帧数 | `113` |
| 接口 TX packet 计数 | 接收前 `0`，接收后 `0` |
| 处理结果 | `FEEDBACK_FAULT`，无效状态未发布到规划话题 |

原始记录：

- [本轮观察摘要](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/evidence/live_existing_bus_observation.json)
- [原始 CAN 帧](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/runs/20261103T055003Z_33d93f5a/raw_rx.jsonl)
- [逐帧解码和时间戳](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/runs/20261103T055003Z_33d93f5a/decoded.jsonl)
- [厂家核对材料](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/VENDOR_ZERO_REFERENCE_CHECK.md)

这些字段来自接收窗口中的独立报文，不是控制器同一采样时刻的完整有效快照。`err_code=0` 不表示符合 MoveIt 起点限位。夹爪 101 mm 是 SDK 协议的行程值，不是指垫内距或夹持证据。

接口由用户置为普通模式，本轮没有改变模式。接口 TX packet 未增加不能证明物理层没有 ACK，也不等于独立旁路/系统调用审计。观察文件里的 `application_can_send_calls=0` 是当时程序的静态声明，不是系统调用实测；新的程序字段已改为 `implementation_can_send_api=NONE`，避免与实测计数混淆。

## 新入口的用途

```bash
bash /home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/start_feedback_planning.sh
```

此入口启动 RViz、MoveIt 和真实反馈状态标记：

- 使用官方 Piper X + AgileX 夹爪模型，ROS domain `92`、命名空间 `/piper_x_real`；原仿真继续 domain `91`，入口未改。
- 不加载 FakeSystem，不启动 `ros2_control_node`、JTC、SDK 或 `agx_arm_ctrl`；没有 `/execute_trajectory`、`follow_joint_trajectory` Action。
- `allow_trajectory_execution=false`、`moveit_manage_controllers=false`，保留 `/move_action` 用于规划请求。
- 所有关节状态只从 `/piper_x_real/feedback/joint_states` 接收，不发布零位、仿真名义姿态或保存状态。
- 目标参考为 `flange_link`，不把名义 141.5 mm 当作已标定的真实 TCP。
- 保留当前 `60×60×38 mm` 相机碰撞盒，明确标记为名义几何。没有把仿真的桌面、支架或线材位置当作实测环境加载。
- 禁止自动修补越界/碰撞起点。缺反馈时显示 `WAITING_FOR_REAL_FEEDBACK`，过期显示 `FEEDBACK_STALE`，始终显示 `MOTION BLOCKED`。

当前反馈不满足合同，因此不会把完整机器人姿态作为有效真机状态注入 MoveIt；不能在此入口验收实物碰撞净空或执行抓取。

## 自然反馈诊断接口

在用户已启用且确定没有其他控制程序争用的 `can0` 上，最多接收 10 秒：

```bash
bash /home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/diagnose_can.sh \
  --existing-active-bus --seconds 10
```

该模式不改接口、不承诺电气无 ACK；仅允许当前已 UP、ERROR-ACTIVE、1 Mbps、无自动重启、身份匹配的接口。若要把**通过完整合同**的反馈送到上述规划入口，可加 `--publish-ros`；此参数不会强行发布越界、缺轴或过期状态。采样结束后不会持续伪造最后状态。

输出每次新建 `runs/<UTC时间_随机ID>/`，包含 `summary.json`、`raw_rx.jsonl` 和 `decoded.jsonl`。检查拒绝时可能只有摘要。遇到模式/设备变化、TX packet 计数增加、解析错误即退出。普通模式的无效状态只额外保存至多一秒所需原始分组，绝不发布无效状态；这不授权运动。

另有**尚未做真机测试**的 `observe_once.sh --site-ready`：供未来从 DOWN 接口配置 listen-only 后做一次十秒诊断，要求操作者确认现场并在自己的终端输入 sudo 密码；固件拒绝或模式读回不符则不置 UP，结束尝试恢复 DOWN。当前 can0 已由用户启用，该脚本会拒绝更改，不能用来中断现有会话。电气 ACK 抑制仍需独立验证。

## 验证与未完成项

已实际运行：14 项合同/解码边界测试通过；无硬件的 MoveIt headless 启动通过；ROS 图实测执行参数关闭、执行 Action 不存在、无反馈发布者，状态为等待真实反馈。见 [graph_verification.json](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/evidence/graph_verification.json) 和 [unit_tests.log](/home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/evidence/unit_tests.log)。RViz GUI 的本轮视觉验收尚未执行。

```bash
bash /home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/scripts/ros_env.sh \
  /usr/bin/python3 -m unittest discover \
  -s /home/ning/文档/ChatGPT/视觉/piper_x_real_bringup_20261103/tests -v
```

原事故锁 SHA256 仍为 `16b9208ffc759fc4f6032cb3d530f103e83758bfdde9987cf1229b1859548e88`，与既有审计一致。没有修改原 SDK、厂家源码、仿真入口或事故锁，没有提交或推送。

后续真机执行还需要：厂家认可的零位/编码器参考核查与当前越界解释、夹爪行程映射核对、驱动风险修复、真实反馈闭环 Action、已验证的停止路径及首次受限运动。几何/TCP 未验收继续阻止自动线材抓取。**当前交付仅打通了真实接收诊断和规划配置，不能称为已实现真机控制。**

</details>
