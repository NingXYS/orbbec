"""把实际点云整理成 GraspNet 需要的 end_points。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from common.config import Config, load_config
from pointcloud.backproject import OrganizedCloud
from pointcloud.workspace import TablePlane, workspace_mask


def sample_points(
    cloud: OrganizedCloud,
    cfg: Optional[Config] = None,
    num_point: Optional[int] = None,
    seed: int = 42,
    plane: Optional[TablePlane] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[TablePlane]]:
    """采样 num_point 个工作空间点，返回 (xyz_sampled, rgb_sampled, workspace_mask, plane)。"""
    cfg = cfg or load_config()
    n = int(num_point or cfg.get_path("graspnet.num_point", 20000))
    seed = int(cfg.get_path("graspnet.seed", seed))
    mask, plane = workspace_mask(cloud, cfg, plane=plane)
    pts = cloud.xyz[mask]
    cols = cloud.rgb[mask]
    if pts.shape[0] == 0:
        n_cloud = int(cloud.valid.sum())
        raise RuntimeError(
            f"工作空间内没有有效点（全图有效点 {n_cloud}）。"
            "请检查深度范围、相机到桌面距离（336L 建议约 0.5 m）与桌面平面。"
        )
    rng = np.random.default_rng(seed)
    if pts.shape[0] >= n:
        idx = rng.choice(pts.shape[0], n, replace=False)
    else:
        extra = rng.choice(pts.shape[0], n - pts.shape[0], replace=True)
        idx = np.concatenate([np.arange(pts.shape[0]), extra])
    return pts[idx].astype(np.float32), cols[idx].astype(np.float32), mask, plane
