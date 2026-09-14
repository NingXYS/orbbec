"""工作空间掩码：深度范围 + 桌面平面 + 图像边缘。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from common.config import Config, load_config
from pointcloud.backproject import OrganizedCloud


@dataclass
class TablePlane:
    """平面方程 n·x + d = 0，法向指向相机（通常接近 -Z）。"""

    normal: np.ndarray  # (3,)
    offset: float
    inlier_count: int

    def height(self, xyz: np.ndarray) -> np.ndarray:
        """点到平面的有符号距离，朝向相机为正（物体在桌面上为正）。"""
        n = self.normal
        # 让法向指向相机：n_z 应为负
        # 法向指向相机（n_z < 0）。点到平面、朝向相机一侧为正，即桌面上方。
        if n[2] > 0:
            n = -n
            d = -self.offset
        else:
            d = self.offset
        return xyz @ n + d


def fit_table_plane(
    xyz: np.ndarray,
    valid: np.ndarray,
    dist_th: float = 0.012,
    n_iter: int = 80,
    max_fit_points: int = 12000,
) -> Optional[TablePlane]:
    """RANSAC 拟合桌面。优先采样图像下半部分。

    1280x800 有效点可达数十万，RANSAC 打分只在降采样子集上进行，平面模型不变。
    """
    h, w = valid.shape
    pts = xyz.reshape(-1, 3)
    mask = valid.reshape(-1)
    # 只在有效像素上拟合，避免 1280x800 全图 mgrid
    ys, xs = np.where(valid)
    if ys.size < 50:
        return None
    lin = ys * w + xs
    prior = ys > h * 0.35
    idx = lin[prior] if int(prior.sum()) >= 200 else lin
    if idx.size < 50:
        return None

    rng = np.random.default_rng(0)
    sample_pool = idx
    if idx.size > max_fit_points:
        sample_pool = rng.choice(idx, max_fit_points, replace=False)
    best_n = None
    best_d = 0.0
    best_cnt = 0
    fit_pts = pts[sample_pool]
    for _ in range(n_iter):
        choice = rng.choice(sample_pool, size=3, replace=False)
        p0, p1, p2 = pts[choice]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(fit_pts @ n + d)
        cnt = int((dist < dist_th).sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_d = cnt, n, d
    if best_n is None:
        return None
    # 用内点 SVD 精修
    dist = np.abs(pts[idx] @ best_n + best_d)
    inliers = pts[idx][dist < dist_th]
    if inliers.shape[0] >= 10:
        c = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - c, full_matrices=False)
        best_n = vh[-1]
        best_n = best_n / (np.linalg.norm(best_n) + 1e-12)
        best_d = -float(best_n @ c)
    return TablePlane(normal=best_n.astype(np.float64), offset=best_d, inlier_count=best_cnt)


def workspace_mask(cloud: OrganizedCloud, cfg: Optional[Config] = None, plane: Optional[TablePlane] = None) -> tuple[np.ndarray, Optional[TablePlane]]:
    """返回 HxW bool 掩码，以及拟合的桌面平面。"""
    cfg = cfg or load_config()
    z = cloud.xyz[:, :, 2]
    z_min = float(cfg.get_path("workspace.z_min", 0.30))
    z_max = float(cfg.get_path("workspace.z_max", 1.20))
    margin = int(cfg.get_path("workspace.roi_margin_px", 20))
    mask = cloud.valid & (z >= z_min) & (z <= z_max)
    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    if bool(cfg.get_path("workspace.use_table_plane", True)):
        if plane is None:
            plane = fit_table_plane(
                cloud.xyz,
                mask,
                dist_th=float(cfg.get_path("workspace.table_inlier_dist", 0.012)),
                max_fit_points=int(cfg.get_path("workspace.plane_sample_n", 12000)),
            )
        if plane is not None:
            hmin = float(cfg.get_path("workspace.height_min_m", 0.008))
            hmax = float(cfg.get_path("workspace.height_max_m", 0.28))
            yy, xx = np.where(mask)
            if yy.size:
                hgt = plane.height(cloud.xyz[yy, xx])
                keep = (hgt >= hmin) & (hgt <= hmax)
                above = np.zeros_like(mask)
                above[yy, xx] = keep
                if int(keep.sum()) >= 500:
                    mask = above
            # 空桌面/纯平面时保留深度范围内的点，避免推理直接失败
    return mask, plane
