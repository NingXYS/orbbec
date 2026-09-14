"""MoveIt 下发结果分类：规划失败 / 超时打明确日志并跳过。不依赖 rclpy。"""

from __future__ import annotations

from dataclasses import dataclass

# 与现场日志一致，便于验收对照
PLAN_REJECTED = "PLAN_REJECTED：MoveGroup 拒绝目标（可能碰撞/越限）"
PLAN_OR_EXEC_FAIL_FMT = "PLAN_OR_EXEC_FAIL status={status} error_code={code} {message}"
NO_ACTION = "/move_action 不可用（真机 MoveIt 未启动或 DOMAIN 不一致）"
SKIP_NEXT = "SKIP_NEXT_ON_FAIL 已记录，继续下一个（若有）"
NO_JOINTS_PREFIX = "无关节反馈: "
# 超过该绝对值视为越限演练目标，禁止 --execute
OVERLIMIT_ABS_RAD = 2.8
# MoveIt 惯例：SUCCESS=1，action STATUS_SUCCEEDED=4
MOVEIT_SUCCESS = 1
GOAL_STATUS_SUCCEEDED = 4


def exec_timeout_msg(timeout_s: float) -> str:
    return f"EXEC_TIMEOUT：{timeout_s:.0f}s 内无结果，尝试取消"


def arrive_timeout_msg(err_deg: float, tol_deg: float) -> str:
    return f"到达超时，max_err={err_deg:.3f}° > {tol_deg}°"


def already_there(err_deg: float, tol_deg: float) -> bool:
    return err_deg <= tol_deg


def velocity_too_fast(vel: float, cap: float = 0.10) -> bool:
    return vel > cap + 1e-9


def joint_jumped(delta_deg: float, warn_deg: float) -> bool:
    return delta_deg >= warn_deg


def find_overlimit_joint(names: list[str], q: list[float]) -> tuple[str, float] | None:
    for n, v in zip(names, q):
        if abs(v) > OVERLIMIT_ABS_RAD:
            return n, v
    return None


def execute_forbidden_reason(
    execute: bool,
    poses: dict[str, list[float]],
    names: list[str],
    order: list[str],
) -> str | None:
    """越限演练文件只允许 plan_only，避免误加 --execute 真动臂。"""
    if not execute:
        return None
    for key in order:
        if "overlimit" in key.lower() or "越限" in key:
            return f"拒绝：姿态 {key} 标记为越限演练，禁止 --execute"
        q = poses.get(key)
        if q is None:
            continue
        hit = find_overlimit_joint(names, q)
        if hit is not None:
            jn, val = hit
            return (
                f"拒绝：{key} 的 {jn}={val:.3f} rad 超过安全阈值 "
                f"{OVERLIMIT_ABS_RAD} rad，禁止 --execute（越限演练只允许 plan_only）"
            )
    return None


@dataclass
class MoveOutcome:
    ok: bool
    error: str | None = None
    note: str | None = None
    skipped: str | None = None
    should_cancel: bool = False
    wait_arrive: bool = False


def classify_after_send(
    *,
    accepted: bool,
    result_arrived: bool,
    status: int | None = None,
    error_code: int | None = None,
    error_message: str = "",
    exec_timeout_s: float = 90.0,
    plan_only: bool = True,
    success_status: int = GOAL_STATUS_SUCCEEDED,
    success_code: int = MOVEIT_SUCCESS,
) -> MoveOutcome:
    """根据 MoveGroup 回包判定：失败则带 PLAN_* / EXEC_TIMEOUT，成功则按 plan_only 分流。"""
    if not accepted:
        return MoveOutcome(ok=False, error=PLAN_REJECTED)
    if not result_arrived:
        return MoveOutcome(
            ok=False,
            error=exec_timeout_msg(exec_timeout_s),
            should_cancel=True,
        )
    if status != success_status or error_code != success_code:
        return MoveOutcome(
            ok=False,
            error=PLAN_OR_EXEC_FAIL_FMT.format(
                status=status,
                code=error_code,
                message=error_message or "",
            ).rstrip(),
        )
    if plan_only:
        return MoveOutcome(ok=True, note="plan_only 成功，未下发执行")
    return MoveOutcome(ok=True, wait_arrive=True)


def skip_line(rec: dict) -> str | None:
    if rec.get("ok"):
        return None
    return SKIP_NEXT
