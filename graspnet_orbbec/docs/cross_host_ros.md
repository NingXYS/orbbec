# 跨机 ROS 2 连通性实测（选项 B）

> 测试时间：2026-09-14  
> 笔记本：Ubuntu **22.04.5**，ROS 2 **Humble**（`/opt/ros/humble`），网卡 `wlp4s0` = **10.101.67.111/24**  
> 控制器：Ubuntu 24.04，ROS 2 **Jazzy**，`wlan0` = **10.101.67.213/24**（见 [controller_recon.md](controller_recon.md)）  
> 原始产物：`output/cross_host/`  
> RViz 截图：[cross_host_rviz_follow.png](cross_host_rviz_follow.png)

---

## 结论（先看这里）

| 项 | 结果 |
|----|------|
| IP 互通 | **通过**。`ping 10.101.67.213` 3/3，RTT 39–75 ms（同一 Wi‑Fi `/24`） |
| 基础话题互通 | **通过**。Humble 能 `ros2 topic list` 看到控制器 Jazzy 的 `/joint_states`、`/cross_host_heartbeat`、`/planning_scene_compat_probe` |
| `/joint_states` echo | **数据正常**。14 轴 `J1-1…J7-1`、`J1-2…J7-2`，position/velocity 完整 |
| `/joint_states` 频率 | **通过**。轻订阅 12 s 窗口 **平均 81 Hz**（≥50 Hz）。RViz+RSP 同时开时 Wi‑Fi 上会抖到 30–70 Hz |
| `moveit_msgs` | **话题 echo 成功**（`PlanningScene`）。包版本不同：Humble **2.2.1** vs Jazzy **2.6.0**。每条消息伴随 FastDDS 警告（见下）。**MoveGroup action 未在真机栈上实测**（控制器未启动 `move_group`） |
| RViz 跟随 | **TF 正常，双臂模型随 `/joint_states` 摆动**（截图）。RobotModel 面板报 Error（mesh/`package://` 解析不完全），但 3D 里模型可见且关节非零 |
| 真臂 Hardware | **本次未上线**。`eth0`/`eth1` **NO-CARRIER**，EtherCAT 网线未接，不能也不该启动真机 MoveIt |

### 下一步走哪条路

**推荐：直连维持（仅限 `sensor_msgs` / TF / 可视化）+ MoveIt 客户端放在控制器 Jazzy 上。不要把笔记本整机升级成 Jazzy，也不要先上 topic 桥。**

理由：

1. **基础消息已经跨发行版打通**，对齐 `ROS_DOMAIN_ID=0`、`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`，并设置 `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` + `ROS_STATIC_PEERS=<对端 IP>` 后，Humble 能稳定订阅 Jazzy 的 `/joint_states`。
2. **笔记本升 Jazzy 等于升 Ubuntu 24.04**。当前 GraspNet / PyOrbbec / conda 栈在 22.04+CUDA 12.6 上已跑通，换发行版成本高、收益只在 MoveIt 类型哈希。
3. **`moveit_msgs` 2.2 vs 2.6** 虽能 echo 简单 `PlanningScene`，但每条都有 `sequence size exceeds remaining buffer`。把 `/move_action` 从 Humble 打到 Jazzy 仍有 CDR/字段风险。规划与执行应留在控制器。
4. **桥接**（`topic_tools` / `ros1_bridge` 类）在基础话题已通的情况下是多余复杂度。

笔记本继续跑 Gemini + GraspNet + VLM；要动臂时在控制器上调 `/move_action`，或笔记本只发 `geometry_msgs/PoseStamped` 薄目标，由 Jazzy 节点转 MoveGroup。

---

## 1. 对齐参数（两边一致）

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS=<对端IPv4>
```

| 主机 | `ROS_STATIC_PEERS` |
|------|-------------------|
| 笔记本 | `10.101.67.213` |
| 控制器 | `10.101.67.111` |

Wi‑Fi 上组播发现不可靠，**必须**加 `ROS_STATIC_PEERS`。两边默认 `RMW` 都是 Fast DDS，无需 Cyclone。

---

## 2. 实测证据

### 2.1 发现与 echo

控制器 Jazzy 按真机关节名 **100 Hz** 发布 `/joint_states`（EtherCAT 未接，用同名正弦探测，**不是**驱动反馈）。笔记本 Humble：

```text
$ ros2 topic list
/cross_host_heartbeat
/joint_states
/parameter_events
/planning_scene_compat_probe
/rosout
```

`ros2 topic echo /joint_states --once` 原文（节选）：

```text
sequence size exceeds remaining buffer
header:
  stamp:
    sec: 1789364981
    nanosec: 810875323
  frame_id: world
name:
- J1-1
- J2-1
...
- J7-2
position:
- 0.11250695284458315
...
```

类型：`sensor_msgs/msg/JointState`。**没有** “type does not match / checksum” 类错误；警告是 FastDDS 反序列化缓冲，字段仍完整。

`/planning_scene_compat_probe`（`moveit_msgs/msg/PlanningScene`）同样 echo 成功，`robot_model_name: ar6e_r700_dual_arm_pillar`。

### 2.2 频率（验收 ≥50 Hz）

轻订阅、12 s 窗口（`output/cross_host/joint_states_hz_12s.txt`）：

```text
average rate: 81.363
min: 0.000s max: 0.114s std dev: 0.02504s window: 748
average rate: 80.924
min: 0.000s max: 0.197s std dev: 0.02558s window: 827
```

**约 81 Hz ≥ 50 Hz。** 同时开 RViz 时 Wi‑Fi 会抖，曾落到 ~32 Hz，属无线链路而非类型不兼容。

### 2.3 RViz

加载任务 1 的致行 URDF：`ar6e_r700_dual_arm_zhixing`（根 `pillar_link`）。笔记本 `robot_state_publisher` 订控制器 `/joint_states` 发 TF。

截图见 [cross_host_rviz_follow.png](cross_host_rviz_follow.png)：

- Displays → **TF: Status Ok**
- 3D 中双臂白色模型可见，关节非零（随正弦 `/joint_states` 变化）
- RobotModel 行显示 **Status: Error**（部分 `package://` mesh 未完全解析），但不影响 TF 跟随结论

**未做到「与真臂姿态一致」的原因：** `eth0`/`eth1` carrier=0，厂家 Hardware 未启动，没有真实 `/joint_states`。接好 EtherCAT 并只读启动后，用同一套 Humble RViz 即可复测真臂。

### 2.4 `moveit_msgs` 版本

| 端 | 包 |
|----|----|
| 笔记本 Humble | `ros-humble-moveit-msgs` **2.2.1** |
| 控制器 Jazzy | `ros-jazzy-moveit-msgs` **2.6.0** |

简单 `PlanningScene` 能跨版本 echo。`MoveGroup` action 因控制器未起 `move_group`，**没有** `ros2 action send_goal` 成功/失败原文。跨小版本 action 仍视为风险，不作为直连运动接口。

---

## 3. 未做 / 限制

- 未在控制器启动 `run_zhixing_moveit_demo.sh`（会申请 Master、使能电机）。
- 未改控制器磁盘；探测发布器用 SSH stdin 跑 Python，测完已杀掉。
- FastDDS 警告 `sequence size exceeds remaining buffer` 每条都有，生产可视化可忽略，**不要**据此在 Humble 里发复杂 MoveIt goal。

---

## 4. 复现命令

控制器：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET ROS_STATIC_PEERS=10.101.67.111
# 发布 /joint_states 或启动只读 Hardware
```

笔记本：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET ROS_STATIC_PEERS=10.101.67.213
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states
```
