# 厂家控制器环境侦察报告

> 侦察时间：2026-09-13 18:46–18:55 CST  
> 目标：`nvidia@10.101.67.213`（无线网卡 `wlan0`）  
> 原则：**只读**。未安装软件、未改工作区、未启动真机 MoveIt/EtherCAT。  
> 本机唯一写入：按任务要求执行 `ssh-copy-id`，向远端 `~/.ssh/authorized_keys` 追加本机 `id_ed25519` 公钥。密码未写入任何文件。

登录后免密校验：`ssh -o BatchMode=yes nvidia@10.101.67.213 'echo KEY_OK'` 成功。

---

## 三个必须回答的问题

| 问题 | 结论 |
|------|------|
| **MoveIt 编程入口是什么？** | **标准 MoveGroup action**：前端 `/move_action`（`moveit_msgs/action/MoveGroup`）。厂家用 `ar6e_r700_moveit_runtime` 的 `move_group_plan_guard` 把 RViz 目标转到 `/move_action_backend`，再进 OMPL。执行链是 **Plan → Execute → JTC `FollowJointTrajectory` → `tc600e_ethercat_driver` Hardware → EtherCAT**。另有厂家笛卡尔服务 `PlanLine`/`PlanCircle` 和 `execute_trajectory` action，**不是**替代 MoveGroup 的私有运动总线。 |
| **基座坐标系 frame 名？** | **当前致行（Zhixing）模型：规划链基座 `pillar_link`，世界系 `world`（SRDF `virtual_joint` 固定连接）。** 旧模型 `ar6e_r700_description_latest` 用 `world` ≡ `base_link`。手眼标定应对齐 **当前栈的 `world` / `pillar_link`**，不要再用旧 `base_link` 当致行 URDF 根。 |
| **rmw 是 Fast DDS 还是 CycloneDDS？** | **Fast DDS（`rmw_fastrtps_cpp`）。** 未安装 CycloneDDS。`RMW_IMPLEMENTATION` 未显式 export，Jazzy 默认即 Fast RTPS。`ROS_DOMAIN_ID` 未设置，**默认 0**。 |

---

## 1. 系统与硬件

| 项 | 实测 |
|----|------|
| 主机名 | `localhost.localdomain` |
| 用户 | `nvidia`（uid 现场会话） |
| OS | Ubuntu **24.04.2 LTS**（noble），`lsb_release` 确认 |
| 内核 | `6.8.12-rt-tegra` **PREEMPT_RT**，aarch64 |
| 板型 | **NVIDIA Jetson AGX Thor Developer Kit**（device-tree model 同此） |
| L4T / Jetpack | L4T **38.2.0**（`/etc/nv_tegra_release`：R38 REVISION 2.0）；`jetson_release` 报 Jetpack missing |
| 模组 | P-Number `p3834-0008` |
| CPU | ARM，**14 核**，最大 2601 MHz |
| 内存 | **122 GiB** 总量（`MemTotal` 128789284 kB ≈ 122.8 GiB）；侦察时 used 7.5 GiB，available ≈ 115 GiB；无 swap |
| 磁盘 | NVMe `/` 437G，已用 47G（12%） |
| GPU | `nvidia-smi`：GPU 名 **NVIDIA Thor**，驱动 **580.00**，CUDA **13.0** |
| CUDA 栈（jtop） | CUDA 13.0.48，cuDNN 9.12.0.46，TensorRT 10.13.2.6，VPI 4.0，Vulkan 1.4.304，OpenCV 4.8.0（**无 CUDA**） |
| 功耗模式 | NV Power Mode[0]: **MAXN** |

网络（与后续跨机 ROS 有关）：

| 接口 | 地址 | 备注 |
|------|------|------|
| `wlan0` | **10.101.67.213/24** | 本次 SSH 入口 |
| `eth0` | 192.168.1.100/24 | 真机脚本检查 carrier；左臂 EtherCAT Master0 |
| `eth1` | 链路 UP，无 IPv4 | 右臂 EtherCAT Master1 |
| `wlan0-ap` | 192.168.88.88/24 | 板载 AP |
| `eth7` | 192.168.7.1/24 | — |

---

## 2. ROS 2 环境

| 项 | 实测 |
|----|------|
| 发行版 | **ROS 2 Jazzy**（`/opt/ros/jazzy`） |
| `ROS_DISTRO` | `jazzy`（source 后） |
| `ROS_DOMAIN_ID` | **未设置 → 默认 0**。`~/.bashrc` **没有** source ROS / 工作区 |
| `RMW_IMPLEMENTATION` | **未设置**；`ros2 doctor`：**middleware name = `rmw_fastrtps_cpp`** |
| 已装 rmw | `rmw_fastrtps_cpp`、`fastrtps` 2.14.6；**无** `rmw_cyclonedds_cpp` / Cyclone 包 |
| 工作区 overlay | `source /home/nvidia/robot_project_ws/ros2_ws/install/setup.bash`（`--merge-install`） |

跨机（本开发机 ↔ 控制器）时必须对齐：**Jazzy + Fast DDS + DOMAIN 0**（或双方显式设同一 `ROS_DOMAIN_ID`）。本开发机若是 Humble/Cyclone，将无法直接 `ros2 topic echo`。

---

## 3. `~/robot_project_ws` 包清单

工作区根：`/home/nvidia/robot_project_ws`（Git 仓库）。README 写明这是 **AR6E-R700 双臂唯一活动工作区**。

### 3.1 顶层目录

| 路径 | 用途 |
|------|------|
| `ros2_ws/src/` | 当前 ROS 2 Jazzy 源码 |
| `ros2_ws/install/` | merge-install 产物，启动依赖 |
| `ros2_ws/build/`、`ros2_ws/log/` | 构建中间物 / 日志 |
| `ros2_ws/tools/` | MoveIt、监测、CSV、VR、标定入口脚本 |
| `ros2_ws/docs/` | 设计与交接记录 |
| `使用说明/` | 操作顺序（含 MoveIt/RViz、EtherCAT、CSV） |
| `222/` | 2026-07-17 权威原始 URDF/网格参考 |
| `ar6e_r700_dual_arm_zhixing_description/` | 工作区根下另有一份描述包拷贝 |
| `third_party/oculus_reader/`、`unity_quest_pose_client/` | VR |
| `archive/` | 只读归档 |

`colcon list`（在 `ros2_ws` 下）**16 个包**：

| 包名 | 角色 |
|------|------|
| **`ar6e_r700_dual_arm_zhixing_description`** | **当前真机 URDF**（立柱双臂+夹爪） |
| **`ar6e_r700_dual_arm_zhixing_moveit_config`** | **当前 MoveIt2 配置 / 真机 launch** |
| `ar6e_r700_description_latest` | 旧描述（`world`/`base_link`） |
| `ar6e_r700_moveit_config_latest` | 旧 MoveIt 配置 |
| `ar6e_r700_moveit_runtime` | 加固版 `move_group` + `/move_action` 守卫 |
| `ar6e_r700_hardware_visualization` | 硬件可视化 |
| **`tc600e_ethercat_driver`** | **厂家 ros2_control Hardware + EtherCAT/TC600E** |
| `robot_arm` | 旧节点/消息；**不得与当前 Hardware 同时启动** |
| `robot_arm_motion_moveit` | 笛卡尔 `PlanLine`/`PlanCircle` + `execute_trajectory` |
| `robot_arm_motion_algorithms`、`robot_arm_dls_kinematics_plugin` | 算法 / IK 插件 |
| `ar6e_r700_teach_replay`、`ar6e_r700_vr_servo`、`ar6e_r700_vr_teleop` | 示教 / VR |
| `ar6e_r700_drag_controller` | 拖动 |
| `ar6e_r700_robot_sdk` | 离线优先静态 C++ SDK（`robot.h`），**不是**在线 MoveGroup 入口 |

### 3.2 URDF / 描述文件位置

**当前（致行 Zhixing，真机入口 `run_zhixing_moveit_demo.sh`）：**

- URDF：`ros2_ws/src/ar6e_r700_dual_arm_zhixing_description/urdf/ar6e_r700_dual_arm_zhixing.urdf`  
  robot name：`ar6e_r700_dual_arm_pillar`  
  根 link：**`pillar_link`**（立柱底面中心 z=0）
- MoveIt xacro/srdf：`ros2_ws/src/ar6e_r700_dual_arm_zhixing_moveit_config/config/`
  - `ar6e_r700_dual_arm_pillar.urdf.xacro`
  - `ar6e_r700_dual_arm_pillar.srdf`（`virtual_joint`：`world` → `pillar_link`）
  - 规划组：`arm_left` / `arm_right` / `dual_arms` / `gripper_*`  
    chain：`pillar_link` → `left_gripper_base_link` / `right_gripper_base_link`
- 关节名：左 `J1-1`…`J7-1`，右 `J1-2`…`J7-2`

**旧模型（仍构建，回退入口 `run_moveit_demo.sh`）：**

- `ros2_ws/src/ar6e_r700_description_latest/urdf/ar6e_r700_v2.urdf`  
  `world` --fixed `world_to_base`→ **`base_link`**（二者重合）
- SRDF chain：`base_link` → `L7_L` / `L7_R` 或 C4 tip

参考网格：`~/robot_project_ws/222/AR6E-R700-双臂_URDF简化_20260717`

### 3.3 MoveIt 配置包

- **现行：** `ar6e_r700_dual_arm_zhixing_moveit_config`  
  真机 launch：`launch/hardware_csp_c5_zero_then_rviz.launch.py`  
  另有 `move_group.launch.py`（`generate_move_group_launch`）、`demo.launch.py`
- **旧：** `ar6e_r700_moveit_config_latest`（C4 单臂 / C5 双臂 CSP launch）
- **运行时加固：** `ar6e_r700_moveit_runtime`  
  - `move_group_safe.cpp`：加载标准 capability，含 `MoveGroupMoveAction`、`MoveGroupExecuteTrajectoryAction`  
  - `scripts/move_group_plan_guard.py`：对外 **`/move_action`**，对内 **`/move_action_backend`**  
  - `scripts/move_both_to_zero_once.py`：`ActionClient(..., MoveGroup, "/move_action")`

### 3.4 厂家驱动包

**`tc600e_ethercat_driver`**：IgH EtherCAT + ros2_control `SystemInterface`。

- 电机：TC600E，左 Master0/eth0 P0–P6，右 Master1/eth1 P0–P6
- 右夹爪：Master1/P7 网关 `LH-RCB-A1`，CAN ID 9
- 反馈经 **`/joint_states`** 与 **`/dynamic_joint_states`**
- 真机 CSP 控制器配置 **`update_rate: 1000`**（1 ms）
- README 强调默认只读 Hardware；全运动路径需显式后端名 + 现场确认短语
- **禁止**与旧 `dual_arm_node` 同时跑

---

## 4. 运行中的 ROS 图（侦察时刻）

`source /opt/ros/jazzy/setup.bash && source .../install/setup.bash` 后：

| 命令 | 结果 |
|------|------|
| `ros2 node list` | **空**（无业务节点） |
| `ros2 topic list` | 仅 `/parameter_events`、`/rosout` |
| `ros2 action list` | **空** → **此时不存在 `/move_action` / MoveGroup** |
| `ros2 topic hz /joint_states` | **`WARNING: topic [/joint_states] does not appear to be published yet`** |
| `ros2 topic info /joint_states` | **Unknown topic** |

原因：真机 MoveIt 栈**未启动**（`run_zhixing_moveit_demo.sh` 需交互确认 `START ZHIXING CURRENT POSE`，会申请 EtherCAT、使能电机）。本次按只读原则**未启动**。

现场另有 **Codex/sandbox 在 `ros2_ws` 里 `colcon build` + `colcon test --packages-select tc600e_ethercat_driver`**（网络隔离的 bwrap）。该测试不向本机默认 ROS domain 发布 `/joint_states`。

### 配置中的 `/joint_states` 频率（非实测）

| 配置 | `controller_manager.update_rate` | 含义 |
|------|----------------------------------|------|
| `ros2_controllers_csp_c5_dual_arm.yaml` 等 CSP 真机 | **1000 Hz** | joint_state_broadcaster 随 1 ms 控制周期 |
| `ros2_controllers.yaml`（默认/仿真倾向） | **100 Hz** | 非 CSP 真机默认 |

CSV 测试说明写目标采样 1000 Hz。要拿到 `ros2 topic hz` 实测，需在监护下启动真机或仿真 `joint_state_broadcaster`，本次未做。

栈起来后预期节点/话题（来自 launch/文档，非本次 list）：`ros2_control_node`、`move_group`、`robot_state_publisher`、`joint_state_broadcaster`、左右 `*_joint_trajectory_controller`、`/joint_states`、`/tf`、`/move_action`。

---

## 5. 接入建议（给后续任务）

1. **编程：** `rclcpp_action` / `rclpy` 连 **`/move_action`**（`moveit_msgs/action/MoveGroup`），group 用 `arm_left` / `arm_right` / `dual_arms`。笛卡尔直线可用厂家 `PlanLine`。不要走未完成的 `ar6e_r700_robot_sdk` 真机路径。
2. **坐标系：** 抓取位姿请变到 **`world` 或 `pillar_link`**（二者固定）。末端用 `left_gripper_base_link` / `right_gripper_base_link`（夹爪基座），旧 tip `L7_*_C4_tip` 仅旧模型。
3. **DDS：** 开发机需 Jazzy + Fast DDS，或显式 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` 且 `ROS_DOMAIN_ID=0`。
4. **网络：** SSH 走 Wi‑Fi `10.101.67.213`；EtherCAT 在 `eth0`/`eth1`，不要把 ROS 组播绑错网卡（可能需 `ROS_LOCALHOST_ONLY` 或 FastDDS 指定 interface）。
5. **安全：** 真机脚本有确认短语和 flock；后续任务不得在无人监护时启动 Hardware。

---

## 6. 写入核验

| 动作 | 是否发生 |
|------|----------|
| `apt` / `pip` / `colcon build`（本任务） | 否 |
| 修改 URDF / yaml / launch | 否 |
| 启动 MoveIt / EtherCAT / 使能电机 | 否 |
| `ssh-copy-id` → 远端 `authorized_keys` | **是（任务要求）** |
| 密码写入仓库/配置/报告 | 否 |

---

## 7. 缺口与下一步

- `/joint_states` **实测 Hz** 需在仿真或授权真机会话中补测（配置目标 1000 Hz）。
- MoveGroup action **存活证据**需在 `run_zhixing_moveit_demo.sh` 起来后执行 `ros2 action list | grep move`。
- 跨机 topic 是否通，取决于防火墙、Wi‑Fi 组播和 FastDDS 网卡选择。
