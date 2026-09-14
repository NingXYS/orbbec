#!/usr/bin/env python3
"""向控制器 /move_action 依次下发两个关节空间目标（Jazzy 本机运行）。

安全默认：plan_only=True（只规划不执行）。真动臂必须加 --execute，
且 max_velocity_scaling ≤ 0.10。规划失败或执行超时：打日志并跳过下一目标。

用法：
  python3 moveit_two_joint_goals.py --poses /tmp/arm_safe/poses.yaml
  python3 moveit_two_joint_goals.py --poses /tmp/arm_safe/poses.yaml --execute
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from move_errors import (
        NO_ACTION,
        NO_JOINTS_PREFIX,
        PLAN_REJECTED,
        already_there,
        arrive_timeout_msg,
        classify_after_send,
        execute_forbidden_reason,
        joint_jumped,
        skip_line,
        velocity_too_fast,
    )
    from poses_io import load_poses
    from zhixing_joints import ARRIVE_TOL_DEG, JUMP_WARN_DEG
except ImportError:  # 工程根在 sys.path 时
    from arm.move_errors import (  # type: ignore[no-redef]
        NO_ACTION,
        NO_JOINTS_PREFIX,
        PLAN_REJECTED,
        already_there,
        arrive_timeout_msg,
        classify_after_send,
        execute_forbidden_reason,
        joint_jumped,
        skip_line,
        velocity_too_fast,
    )
    from arm.poses_io import load_poses  # type: ignore[no-redef]
    from arm.zhixing_joints import ARRIVE_TOL_DEG, JUMP_WARN_DEG  # type: ignore[no-redef]

import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def _deg(rad: float) -> float:
    return math.degrees(rad)


class SafeTwoPose(Node):
    def __init__(self, joint_names: list[str], group: str, vel: float, acc: float) -> None:
        super().__init__("moveit_two_joint_goals")
        self.joint_names = joint_names
        self.group = group
        self.vel = vel
        self.acc = acc
        self._pos: dict[str, float] = {}
        self._stamp_ns: int = 0
        self._hist: list[tuple[float, dict[str, float]]] = []
        self._move = ActionClient(self, MoveGroup, "/move_action")
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)

    def _on_js(self, msg: JointState) -> None:
        now = time.monotonic()
        cur = dict(zip(msg.name, msg.position))
        if self._pos:
            for n in self.joint_names:
                if n in cur and n in self._pos:
                    d = abs(_deg(cur[n] - self._pos[n]))
                    if joint_jumped(d, JUMP_WARN_DEG):
                        self.get_logger().error(
                            f"JOINT_JUMP {n} Δ={d:.2f}° 超过 {JUMP_WARN_DEG}°，可能丢包或驱动异常"
                        )
        self._pos.update(cur)
        self._stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._hist.append((now, {n: self._pos[n] for n in self.joint_names if n in self._pos}))
        if len(self._hist) > 4000:
            self._hist = self._hist[-2000:]

    def wait_joints(self, timeout: float = 10.0) -> list[float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(n in self._pos for n in self.joint_names):
                return [self._pos[n] for n in self.joint_names]
        missing = [n for n in self.joint_names if n not in self._pos]
        raise RuntimeError(f"等待 /joint_states 超时，缺 {missing}")

    def max_err_deg(self, target: list[float]) -> float:
        cur = [self._pos[n] for n in self.joint_names]
        return max(abs(_deg(c - t)) for c, t in zip(cur, target))

    def _goal(self, start: list[float], target: list[float], plan_only: bool) -> MoveGroup.Goal:
        constraints = Constraints()
        for name, q in zip(self.joint_names, target):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = q
            jc.tolerance_above = math.radians(0.3)
            jc.tolerance_below = math.radians(0.3)
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group
        goal.request.pipeline_id = "ompl"
        goal.request.start_state = RobotState()
        goal.request.start_state.joint_state.name = list(self.joint_names)
        goal.request.start_state.joint_state.position = list(start)
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.request.num_planning_attempts = 4
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = self.vel
        goal.request.max_acceleration_scaling_factor = self.acc
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return goal

    def send_one(self, name: str, target: list[float], plan_only: bool, exec_timeout: float) -> dict:
        """下发一个目标。规划失败/超时：记录并返回 ok=False，不抛到主循环外。"""
        rec: dict = {"name": name, "ok": False, "plan_only": plan_only}
        try:
            start = self.wait_joints()
        except Exception as e:
            rec["error"] = f"{NO_JOINTS_PREFIX}{e}"
            self.get_logger().error(rec["error"])
            return rec
        rec["start_err_deg"] = max(abs(_deg(s - t)) for s, t in zip(start, target))
        if already_there(rec["start_err_deg"], ARRIVE_TOL_DEG):
            rec["ok"] = True
            rec["skipped"] = "already_there"
            rec["final_err_deg"] = rec["start_err_deg"]
            self.get_logger().info(f"{name} 已在目标内（{rec['start_err_deg']:.3f}°），跳过")
            return rec

        if not self._move.wait_for_server(timeout_sec=10.0):
            rec["error"] = NO_ACTION
            self.get_logger().error(rec["error"])
            return rec

        goal = self._goal(start, target, plan_only)
        t_send = time.monotonic()
        rec["t_send"] = t_send
        exec_started_at = None

        def _fb(fb) -> None:
            nonlocal exec_started_at
            st = str(getattr(fb, "state", "") or "")
            if exec_started_at is None and st and st.upper() not in ("IDLE", "PLANNING", ""):
                exec_started_at = time.monotonic()

        send_fut = self._move.send_goal_async(goal, feedback_callback=_fb)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=15.0)
        handle = send_fut.result()
        accepted = handle is not None and bool(getattr(handle, "accepted", False))
        wrapped = None
        if accepted:
            result_fut = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_fut, timeout_sec=exec_timeout)
            wrapped = result_fut.result()

        outcome = classify_after_send(
            accepted=accepted,
            result_arrived=wrapped is not None,
            status=int(wrapped.status) if wrapped is not None else None,
            error_code=int(wrapped.result.error_code.val) if wrapped is not None else None,
            error_message=(
                getattr(wrapped.result.error_code, "message", "") or ""
                if wrapped is not None
                else ""
            ),
            exec_timeout_s=exec_timeout,
            plan_only=plan_only,
            success_status=GoalStatus.STATUS_SUCCEEDED,
            success_code=MoveItErrorCodes.SUCCESS,
        )
        rec["latency_send_to_result_s"] = time.monotonic() - t_send
        rec["latency_send_to_exec_s"] = (
            (exec_started_at - t_send) if exec_started_at is not None else None
        )
        if wrapped is not None:
            rec["action_status"] = int(wrapped.status)
            rec["error_code"] = int(wrapped.result.error_code.val)
            rec["planning_time_s"] = float(getattr(wrapped.result, "planning_time", 0.0) or 0.0)

        if outcome.should_cancel and handle is not None:
            rec["error"] = outcome.error
            self.get_logger().error(rec["error"])
            try:
                cancel_fut = handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_fut, timeout_sec=5.0)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"取消 goal 失败: {e}")
            return rec
        if not outcome.ok:
            rec["error"] = outcome.error or PLAN_REJECTED
            self.get_logger().error(rec["error"])
            return rec
        if plan_only:
            rec["ok"] = True
            rec["note"] = outcome.note
            self.get_logger().info(f"{name} 规划成功 planning_time={rec.get('planning_time_s', 0):.3f}s")
            return rec

        # 等真实反馈贴近目标
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            err = self.max_err_deg(target)
            if err <= ARRIVE_TOL_DEG:
                rec["ok"] = True
                rec["final_err_deg"] = err
                rec["latency_send_to_exec_s"] = rec["latency_send_to_exec_s"] or rec["planning_time_s"]
                self.get_logger().info(f"{name} 到达 max_err={err:.3f}°")
                return rec
        rec["final_err_deg"] = self.max_err_deg(target)
        rec["error"] = arrive_timeout_msg(rec["final_err_deg"], ARRIVE_TOL_DEG)
        self.get_logger().error(rec["error"])
        return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--execute", action="store_true", help="真正执行；默认只规划")
    p.add_argument("--vel", type=float, default=0.10, help="速度缩放，验收要求 ≤0.10")
    p.add_argument("--acc", type=float, default=0.10)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--order", default="pose_a,pose_b", help="逗号分隔的姿态名")
    args = p.parse_args()
    if velocity_too_fast(args.vel):
        print("拒绝：速度缩放必须 ≤0.10", file=sys.stderr)
        return 2

    names, group, poses = load_poses(Path(args.poses))
    order = [s.strip() for s in args.order.split(",") if s.strip()]
    for k in order:
        if k not in poses:
            print(f"缺少姿态 {k}", file=sys.stderr)
            return 2
    blocked = execute_forbidden_reason(args.execute, poses, names, order)
    if blocked:
        print(blocked, file=sys.stderr)
        return 2

    rclpy.init()
    node = SafeTwoPose(names, group, args.vel, args.acc)
    results = []
    try:
        node.wait_joints()
        for key in order:
            print(f"==== {key} plan_only={not args.execute} ====", flush=True)
            rec = node.send_one(key, poses[key], plan_only=not args.execute, exec_timeout=args.timeout)
            results.append(rec)
            print(rec, flush=True)
            line = skip_line(rec)
            if line:
                print(line, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {e}", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 2

    ok_all = all(r.get("ok") for r in results)
    lat = [r.get("latency_send_to_exec_s") for r in results if r.get("latency_send_to_exec_s") is not None]
    print("SUMMARY", {"ok_all": ok_all, "n": len(results), "latencies_s": lat, "execute": args.execute})
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok_all else 3


if __name__ == "__main__":
    raise SystemExit(main())
