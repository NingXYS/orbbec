# MoveIt 编程接口首次执行（安全）

> 客户端跑在 **控制器 Jazzy 本机**（任务 2：不要从 Humble 打 `/move_action`）。  
> 速度缩放默认 **10%**。默认 **只规划不执行**；真动臂必须加 `--execute`，并保证人手在急停上。

节点：

- `arm/record_joint_pose.py` — 记录当前真实 `/joint_states`
- `arm/moveit_two_joint_goals.py` — 对 `/move_action` 下发两个关节目标

---

## 你需要先做的现场准备

1. **松开急停**，确认驱动有电。  
2. 插上 EtherCAT：**左臂 → eth0，右臂 → eth1**。  
3. 在 Thor 上确认：

```bash
cat /sys/class/net/eth0/carrier   # 1
cat /sys/class/net/eth1/carrier   # 1
```

4. 周围清空，人手不离开急停。  
5. **交互终端**启动真机（会保持当前姿态，不自动回零）：

```bash
cd /home/nvidia/robot_project_ws/ros2_ws
./tools/run_zhixing_moveit_demo.sh
```

确认句必须一字不差：

```text
START ZHIXING CURRENT POSE
```

记住启动日志里的 **`ROS_DOMAIN_ID=...`**（随机 210–229，且 `LOCALHOST`）。

RViz：**白色 = 真机，橙色 = Goal。Plan 不动臂，Execute 才动。** 第一次不要随便 Execute。

---

## 记录两个安全姿态

新开一个 Thor 终端（DOMAIN 与上面一致）：

```bash
cd /home/nvidia/robot_project_ws/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=<启动时打印的数字>
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# 姿态 A = 当前真实位姿（启动后的保持位，通常最安全）
python3 /tmp/arm_safe/record_joint_pose.py --name pose_a --out /tmp/arm_safe/poses.yaml
```

然后在 RViz 把橙色 Goal **小幅**拖到第二个安全位（桌面上方、无碰撞），点 **Plan**。确认轨迹合理后再点 **Execute**（人手在急停上）。到达后：

```bash
python3 /tmp/arm_safe/record_joint_pose.py --name pose_b --out /tmp/arm_safe/poses.yaml
```

---

## 程序下发（先只规划）

```bash
python3 /tmp/arm_safe/moveit_two_joint_goals.py \
  --poses /tmp/arm_safe/poses.yaml --order pose_a,pose_b
```

看到两个 `ok: True` 且 `plan_only` 成功后，才加 `--execute`（≤10% 速度）：

```bash
python3 /tmp/arm_safe/moveit_two_joint_goals.py \
  --poses /tmp/arm_safe/poses.yaml --order pose_a,pose_b --execute --vel 0.10
```

规划失败 / 超时会打 `PLAN_*` / `EXEC_TIMEOUT` 并跳过，不会死循环重试。

验收：到达误差 ≤0.5°；发目标到开始执行 ≤2 s；`JOINT_JUMP` 日志应为空。

**任务 3 已收口**（真机双位姿 2026-09-14 通过）。结项说明：[task3_closeout.md](task3_closeout.md)。  
越限目标的现场演练因控制器占用而推迟，不是硬门槛；本机 `pytest tests/test_moveit_fail_paths.py` 已覆盖失败日志。控制器空闲后若要补演练，只用 `arm/poses_overlimit_plan_only.yaml` 且 **不要** `--execute`。

---

## 不要做的事

- 不要从笔记本 Humble 发 `/move_action`。  
- 不要在 `eth0/eth1 carrier=0` 时强行启动。  
- 不要用文档里的 `MOVE ZHIXING BOTH TO ZERO`（那会真的规划到 q=0）。当前入口是 **保持当前位置**。
