"""点云质量评估：尺寸测量、RGB-D 边缘对齐、空洞率。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from capture.types import RGBDFrame
from pointcloud.backproject import OrganizedCloud, frame_to_cloud
from pointcloud.workspace import TablePlane, fit_table_plane


@dataclass
class QualityReport:
    valid_ratio: float
    hole_ratio: float
    mean_depth_m: float
    std_depth_m: float
    edge_align_px: Optional[float]
    block_size_m: Optional[np.ndarray]  # (dx,dy,dz) AABB
    notes: list


def hole_ratio(depth_m: np.ndarray, zmin: float, zmax: float) -> tuple[float, float]:
    roi = (depth_m > 0) | True
    valid = (depth_m >= zmin) & (depth_m <= zmax)
    total = depth_m.size
    v = int(valid.sum())
    return v / max(total, 1), 1.0 - v / max(total, 1)


def measure_edge_alignment(color_bgr: np.ndarray, depth_m: np.ndarray) -> Optional[float]:
    """用 Canny 边缘互相关估计 RGB 与深度边缘的像素偏差。"""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    depth_u8 = np.zeros_like(gray)
    valid = depth_m > 0
    if valid.sum() < 1000:
        return None
    d = depth_m.copy()
    d[~valid] = np.nan
    finite = np.nan_to_num(d, nan=0.0)
    finite = cv2.normalize(finite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    e_rgb = cv2.Canny(gray, 50, 150)
    e_d = cv2.Canny(finite, 50, 150)
    # 只在深度有效区域比较
    kernel = np.ones((5, 5), np.uint8)
    e_d = cv2.dilate(e_d, kernel, iterations=1)
    ys, xs = np.where(e_rgb > 0)
    if ys.size < 200:
        return None
    # 对 RGB 边缘点，在深度边缘图上找最近距离
    dist = cv2.distanceTransform(255 - e_d, cv2.DIST_L2, 3)
    sample = dist[ys, xs]
    # 取中位数，抗噪
    return float(np.median(sample))


def estimate_object_aabb(cloud: OrganizedCloud, plane: Optional[TablePlane] = None) -> Optional[np.ndarray]:
    """对桌面上方最大簇做 AABB，返回边长 (x,y,z) 米。"""
    pts = cloud.xyz.reshape(-1, 3)
    mask = cloud.valid.reshape(-1)
    if plane is None:
        plane = fit_table_plane(cloud.xyz, cloud.valid)
    if plane is None:
        cand = pts[mask]
    else:
        h = plane.height(pts)
        cand = pts[mask & (h > 0.01) & (h < 0.25)]
    if cand.shape[0] < 50:
        return None
    # 简单距离聚类：取距质心最近的主簇
    c = np.median(cand, axis=0)
    d = np.linalg.norm(cand - c, axis=1)
    keep = cand[d < np.percentile(d, 80)]
    if keep.shape[0] < 20:
        keep = cand
    aabb = keep.max(axis=0) - keep.min(axis=0)
    return aabb.astype(np.float64)


def evaluate_frame(frame: RGBDFrame, expected_block_m: Optional[float] = None) -> QualityReport:
    cloud = frame_to_cloud(frame)
    zmin, zmax = 0.35, 1.5
    valid_r, hole_r = hole_ratio(frame.depth_m, zmin, zmax)
    valid = frame.depth_m[(frame.depth_m >= zmin) & (frame.depth_m <= zmax)]
    mean_d = float(valid.mean()) if valid.size else 0.0
    std_d = float(valid.std()) if valid.size else 0.0
    edge = measure_edge_alignment(frame.color_bgr, frame.depth_m)
    plane = fit_table_plane(cloud.xyz, cloud.valid)
    aabb = estimate_object_aabb(cloud, plane)
    notes = []
    if hole_r > 0.4:
        notes.append("有效深度比例偏低，可能超出 Astra Pro Plus 近距（约 0.6 m）或物体反光/透明")
    if edge is not None and edge > 10:
        notes.append(f"RGB-D 边缘对齐中位偏差 {edge:.1f} px，超过 10 px 验收阈值")
    if expected_block_m is not None and aabb is not None:
        # 取与期望最接近的两条边（桌面 XY）
        xy = np.sort(aabb[:2])
        err = abs(float(np.median(aabb)) - expected_block_m)
        notes.append(f"AABB 边长 {aabb*1000} mm，相对 {expected_block_m*1000:.0f} mm 块的粗误差约 {err*1000:.1f} mm")
    return QualityReport(
        valid_ratio=valid_r,
        hole_ratio=hole_r,
        mean_depth_m=mean_d,
        std_depth_m=std_d,
        edge_align_px=edge,
        block_size_m=aabb,
        notes=notes,
    )
