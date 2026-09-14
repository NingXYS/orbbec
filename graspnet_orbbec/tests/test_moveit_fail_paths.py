"""任务 3 失败路径：不连控制器，核验日志文案与跳过逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arm.move_errors import (
    GOAL_STATUS_SUCCEEDED,
    MOVEIT_SUCCESS,
    NO_ACTION,
    PLAN_REJECTED,
    SKIP_NEXT,
    already_there,
    arrive_timeout_msg,
    classify_after_send,
    exec_timeout_msg,
    execute_forbidden_reason,
    find_overlimit_joint,
    joint_jumped,
    skip_line,
    velocity_too_fast,
)
from arm.poses_io import load_poses
from arm.zhixing_joints import ALL_JOINTS, ARRIVE_TOL_DEG, JUMP_WARN_DEG


def test_plan_rejected_and_timeout_and_planning_fail():
    rejected = classify_after_send(accepted=False, result_arrived=False)
    assert rejected.ok is False
    assert rejected.error == PLAN_REJECTED
    assert rejected.should_cancel is False

    timed = classify_after_send(accepted=True, result_arrived=False, exec_timeout_s=10)
    assert timed.ok is False
    assert timed.should_cancel is True
    assert timed.error == exec_timeout_msg(10)
    assert "EXEC_TIMEOUT" in timed.error

    failed = classify_after_send(
        accepted=True,
        result_arrived=True,
        status=GOAL_STATUS_SUCCEEDED,
        error_code=-1,
        error_message="joint out of bounds",
        plan_only=True,
    )
    assert failed.ok is False
    assert failed.error is not None
    assert failed.error.startswith("PLAN_OR_EXEC_FAIL")
    assert "error_code=-1" in failed.error
    assert "joint out of bounds" in failed.error


def test_plan_only_success_and_execute_waits():
    planned = classify_after_send(
        accepted=True,
        result_arrived=True,
        status=GOAL_STATUS_SUCCEEDED,
        error_code=MOVEIT_SUCCESS,
        plan_only=True,
    )
    assert planned.ok is True
    assert planned.note == "plan_only 成功，未下发执行"
    assert planned.wait_arrive is False

    exec_ok = classify_after_send(
        accepted=True,
        result_arrived=True,
        status=4,
        error_code=1,
        plan_only=False,
    )
    assert exec_ok.ok is True
    assert exec_ok.wait_arrive is True


def test_skip_next_on_fail_continues():
    recs = [
        {"name": "pose_overlimit", "ok": False, "error": PLAN_REJECTED},
        {"name": "pose_a", "ok": True},
    ]
    lines = [skip_line(r) for r in recs]
    assert lines[0] == SKIP_NEXT
    assert lines[1] is None
    # 主循环不因第一段失败而中断
    assert recs[1]["ok"] is True


def test_already_there_and_jump_and_velocity():
    assert already_there(0.278, ARRIVE_TOL_DEG) is True
    assert already_there(0.51, ARRIVE_TOL_DEG) is False
    assert joint_jumped(8.0, JUMP_WARN_DEG) is True
    assert joint_jumped(7.9, JUMP_WARN_DEG) is False
    assert velocity_too_fast(0.10) is False
    assert velocity_too_fast(0.11) is True
    msg = arrive_timeout_msg(1.2, ARRIVE_TOL_DEG)
    assert "到达超时" in msg
    assert NO_ACTION.startswith("/move_action")


def test_overlimit_yaml_plan_only_blocks_execute():
    names, group, poses = load_poses(ROOT / "arm" / "poses_overlimit_plan_only.yaml")
    assert group == "dual_arms"
    assert names == ALL_JOINTS
    assert "pose_overlimit" in poses
    hit = find_overlimit_joint(names, poses["pose_overlimit"])
    assert hit is not None
    assert hit[0] == "J2-2"
    assert hit[1] == pytest.approx(3.5)

    assert execute_forbidden_reason(False, poses, names, ["pose_overlimit"]) is None
    reason = execute_forbidden_reason(True, poses, names, ["pose_overlimit"])
    assert reason is not None
    assert "禁止 --execute" in reason

    safe_names, _, safe = load_poses(ROOT / "arm" / "poses_20260914.yaml")
    assert execute_forbidden_reason(True, safe, safe_names, ["pose_a", "pose_b"]) is None
