"""致行 URDF 关节名（与 /joint_states、MoveIt dual_arms 一致）。"""

LEFT_JOINTS = [f"J{i}-1" for i in range(1, 8)]
RIGHT_JOINTS = [f"J{i}-2" for i in range(1, 8)]
ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS
GROUP_DUAL = "dual_arms"

# 验收：到达误差 0.5°；相邻反馈跳变告警阈值
ARRIVE_TOL_DEG = 0.5
JUMP_WARN_DEG = 8.0
