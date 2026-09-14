"""阶段一规则过滤：倾角 / 离桌高度 / 夹爪宽度 / 分数。

阶段二会再加上可达性与 MoveIt 无碰撞；这里先把几何规则做扎实，方便肉眼验收。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from common.config import Config, load_config
from pointcloud.workspace import TablePlane


def _as_arrays(gg):
    return (
        np.asarray(gg.translations, dtype=np.float64),
        np.asarray(gg.rotation_matrices, dtype=np.float64),
        np.asarray(gg.widths, dtype=np.float64),
        np.asarray(gg.scores, dtype=np.float64),
    )


def filter_grasps(gg, plane: Optional[TablePlane] = None, cfg: Optional[Config] = None):
    """返回过滤后的 GraspGroup。空输入原样返回。"""
    cfg = cfg or load_config()
    if gg is None or len(gg) == 0:
        return gg
    T, R, widths, scores = _as_arrays(gg)
    keep = np.ones(len(gg), dtype=bool)

    min_score = float(cfg.get_path("filter.min_score", 0.05))
    keep &= scores >= min_score

    min_w = float(cfg.get_path("filter.min_width_m", 0.01))
    max_w = float(cfg.get_path("filter.max_width_m", 0.10))
    keep &= (widths >= min_w) & (widths <= max_w)

    # 接近方向：GraspNet 中 rotation 的第三列是接近方向（夹爪坐标系 z）
    approach = R[:, :, 2]
    approach = approach / (np.linalg.norm(approach, axis=1, keepdims=True) + 1e-9)

    if plane is not None:
        n = plane.normal.copy()
        if n[2] > 0:
            n = -n
        # 桌面法向指向相机，期望抓取接近方向与 -n（朝下指向桌面）夹角不要太大
        table_down = -n
        table_down = table_down / (np.linalg.norm(table_down) + 1e-9)
        cosang = np.clip(approach @ table_down, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        max_ang = float(cfg.get_path("filter.max_approach_angle_deg", 70))
        keep &= ang <= max_ang

        h = plane.height(T)
        hmin = float(cfg.get_path("filter.min_height_above_table_m", 0.008))
        hmax = float(cfg.get_path("filter.max_height_above_table_m", 0.25))
        keep &= (h >= hmin) & (h <= hmax)
    else:
        # 无桌面时：接近方向应大致朝 +Z（相机看向桌面，抓取从上方）
        # 相机系 Z 向前，桌面抓取 approach 常接近 +Z
        cosang = np.clip(approach[:, 2], -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        max_ang = float(cfg.get_path("filter.max_approach_angle_deg", 70))
        keep &= ang <= max_ang

    return gg[keep]


def grasp_to_dicts(gg, top_k: int = 50) -> list[dict]:
    """序列化前 top_k 个抓取，便于写 JSON 报告。"""
    if gg is None or len(gg) == 0:
        return []
    gg = gg.sort_by_score()
    n = min(int(top_k), len(gg))
    out = []
    T = np.asarray(gg.translations)
    R = np.asarray(gg.rotation_matrices)
    W = np.asarray(gg.widths)
    S = np.asarray(gg.scores)
    D = np.asarray(gg.depths) if hasattr(gg, "depths") else np.zeros(len(gg))
    for i in range(n):
        out.append(
            {
                "score": float(S[i]),
                "width_m": float(W[i]),
                "depth_m": float(D[i]),
                "translation_m": T[i].astype(float).tolist(),
                "rotation": R[i].astype(float).tolist(),
            }
        )
    return out
