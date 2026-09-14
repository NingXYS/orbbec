"""把目标 3D 质心匹配到最近的 GraspNet 候选。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


class NoGraspMatchError(ValueError):
    """没有合格的最近抓取（空、过远或分数过低）。"""


@dataclass
class GraspMatch:
    index: int
    distance_m: float
    score: float
    translation_m: list[float]
    width_m: float
    rotation: list

    def to_dict(self) -> dict:
        return asdict(self)


def match_nearest_grasp(
    gg,
    xyz_m,
    *,
    min_score: float = 0.08,
    max_dist_m: float = 0.20,
) -> GraspMatch:
    """在合格候选里选抓取中心距质心最近的一个。

    先丢掉异常低分，再按欧氏距离取最近，避免高亮到远处垃圾抓取。
    """
    if gg is None or len(gg) == 0:
        raise NoGraspMatchError("无 GraspNet 候选，无法匹配")
    xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    if not np.isfinite(xyz).all():
        raise NoGraspMatchError(f"质心非法 {xyz}")
    T = np.asarray(gg.translations, dtype=np.float64).reshape(-1, 3)
    S = np.asarray(gg.scores, dtype=np.float64).reshape(-1)
    n = min(len(T), len(S), len(gg))
    T, S = T[:n], S[:n]
    dist = np.linalg.norm(T - xyz[None, :], axis=1)
    ok = np.isfinite(dist) & np.isfinite(S) & (S >= float(min_score))
    if not bool(ok.any()):
        top = float(np.nanmax(S)) if S.size else 0.0
        raise NoGraspMatchError(
            f"无合格抓取：{n} 个候选均低于分数阈值 {min_score:.2f}（最高 {top:.2f}）"
        )
    masked = np.where(ok, dist, np.inf)
    idx = int(np.argmin(masked))
    d = float(masked[idx])
    if d > float(max_dist_m):
        raise NoGraspMatchError(
            f"最近抓取距质心 {d * 1000:.0f} mm > 阈值 {float(max_dist_m) * 1000:.0f} mm，不高亮"
        )
    W = np.asarray(getattr(gg, "widths", np.zeros(n)), dtype=np.float64).reshape(-1)
    R = np.asarray(getattr(gg, "rotation_matrices", np.zeros((n, 3, 3))), dtype=np.float64)
    return GraspMatch(
        index=idx,
        distance_m=d,
        score=float(S[idx]),
        translation_m=T[idx].astype(float).tolist(),
        width_m=float(W[idx]) if idx < len(W) else 0.0,
        rotation=R[idx].astype(float).tolist() if R.ndim == 3 and idx < len(R) else np.eye(3).tolist(),
    )
