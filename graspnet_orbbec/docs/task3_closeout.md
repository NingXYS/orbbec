# 任务 3 结项：MoveIt 编程接口首次执行

> 结项日期：2026-09-14。真机验收见 [MoveIt双位姿验证记录_20260914.md](MoveIt双位姿验证记录_20260914.md)。  
> **控制器当前在做其他工作：越限目标的现场演练跳过，不作为硬门槛。** 失败日志已在本机单测覆盖。

## 结论

**任务 3 收口完成。** 程序下发 → MoveIt 规划 → 执行 → `/joint_states` 回传已在 Thor 本机打通。不必再启栈、不必再录姿态、不必再 `--execute`。

| 验收项 | 实测 | 判定 |
|--------|------|------|
| 两个目标，关节误差 ≤0.5° | pose_a **0.278°**，pose_b **0.315°** | 通过 |
| 发目标到开始执行 ≤2 s | 规划约 **10 ms**；整段运动约 2.19 s（10% 速度、约 31°） | 通过 |
| `/joint_states` 连续、无跳变 | 记录无 `JOINT_JUMP` | 通过 |
| 规划失败有日志并跳过 | 代码打 `PLAN_REJECTED` / `PLAN_OR_EXEC_FAIL` / `EXEC_TIMEOUT` + `SKIP_NEXT_ON_FAIL`；本机单测覆盖 | 通过（现场演练推迟） |
| 无碰撞告警 | 真机两段均 `error_code=1` | 通过 |
| 速度 ≤10%、急停在手 | `--vel 0.10` | 通过 |

MoveGroup 入口仍是 Thor 上的 `/move_action`。笔记本 Humble **不要**跨机打该 action。

## 已归档产物

| 文件 | 用途 |
|------|------|
| `arm/poses_20260914.yaml` | 真机 pose_a / pose_b（不要依赖 `/tmp`） |
| `arm/moveit_two_joint_goals.py` | 双目标客户端，默认 plan_only |
| `arm/record_joint_pose.py` | 从 `/joint_states` 记姿态 |
| `arm/move_errors.py` | 失败分类（无 ROS 依赖） |
| `arm/poses_overlimit_plan_only.yaml` | 以后现场演练用，**禁止 --execute** |
| `tests/test_moveit_fail_paths.py` | 失败路径单测 |

复现这两次动作（控制器空闲且栈已起、人手在急停上）：

```bash
# Thor 本机，DOMAIN 与 run_zhixing_moveit_demo.sh 打印的一致
python3 arm/moveit_two_joint_goals.py \
  --poses arm/poses_20260914.yaml --order pose_a,pose_b --execute --vel 0.10
```

## 推迟项（控制器空闲后再做，非阻塞）

故意给越限目标、只规划、看日志。**不要加 `--execute`。**

```bash
python3 arm/moveit_two_joint_goals.py \
  --poses arm/poses_overlimit_plan_only.yaml --order pose_overlimit
```

预期：`PLAN_REJECTED` 或 `PLAN_OR_EXEC_FAIL`，打印 `SKIP_NEXT_ON_FAIL`，机械臂不动。  
脚本对名字含 `overlimit` 或关节绝对值 >2.8 rad 的目标会拒绝 `--execute`。

## 本机核验（不碰控制器）

```bash
cd /home/ning/orbbec/graspnet_orbbec
source scripts/activate.sh
pytest tests/test_moveit_fail_paths.py -q
```

覆盖：拒绝目标、执行超时并应取消、规划 error_code 失败、失败后继续下一个、已在目标内跳过、速度 >10% 拒绝、越限 YAML 禁止执行。

## 下一步（不是任务 3）

手眼标定、把笔记本 GraspNet/VLM 的数字递到 Thor 再调 `/move_action`。感知仍在笔记本，运动仍在控制器。
