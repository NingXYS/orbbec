"""深度图反投影为有组织点云（相机坐标系，单位米）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capture.types import Intrinsics, RGBDFrame


@dataclass
class OrganizedCloud:
    """与图像同形状的点云。"""

    xyz: np.ndarray  # HxWx3 float32，米，相机系
    rgb: np.ndarray  # HxWx3 float32，0-1
    valid: np.ndarray  # HxW bool

    def flatten(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self.valid.reshape(-1)
        return self.xyz.reshape(-1, 3)[mask], self.rgb.reshape(-1, 3)[mask]


def backproject(depth_m: np.ndarray, K: Intrinsics, organized: bool = True) -> np.ndarray:
    """针孔反投影。depth_m 单位米，与 GraspNet create_point_cloud_from_depth_image 一致。"""
    h, w = depth_m.shape
    if h != K.height or w != K.width:
        raise ValueError(f"深度图尺寸 {w}x{h} 与内参 {K.width}x{K.height} 不一致")
    xmap, ymap = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (xmap - np.float32(K.cx)) * z / np.float32(K.fx)
    y = (ymap - np.float32(K.cy)) * z / np.float32(K.fy)
    cloud = np.stack([x, y, z], axis=-1)
    if not organized:
        cloud = cloud.reshape(-1, 3)
    return cloud


def frame_to_cloud(frame: RGBDFrame, min_depth: float = 0.05, max_depth: float = 2.0) -> OrganizedCloud:
    xyz = backproject(frame.depth_m, frame.color_intrinsics, organized=True)
    rgb = frame.color_rgb.astype(np.float32) / 255.0
    valid = (frame.depth_m > min_depth) & (frame.depth_m < max_depth) & np.isfinite(frame.depth_m)
    xyz = np.where(valid[:, :, None], xyz, 0.0).astype(np.float32)
    return OrganizedCloud(xyz=xyz, rgb=rgb, valid=valid)


def display_cloud(
    cloud: OrganizedCloud,
    zmin: float = 0.30,
    zmax: float = 1.20,
    voxel: float = 0.005,
    estimate_normals: bool = False,
):
    """给可视化用的稠密着色点云：保留桌面，去掉过远背景，体素均匀化。

    推理仍用 workspace 掩码；展示用这张云，才会接近官方文档里的场景点云。
    默认不算法向：离屏渲染走 unlit，连续推理里算法向会多几十毫秒。
    """
    import open3d as o3d

    z = cloud.xyz[:, :, 2]
    valid = cloud.valid & (z >= zmin) & (z <= zmax) & np.isfinite(z)
    pts = cloud.xyz[valid].astype(np.float64)
    cols = np.clip(cloud.rgb[valid].astype(np.float64), 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    if voxel > 0 and pts.shape[0] > 2000:
        pcd = pcd.voxel_down_sample(voxel)
    if estimate_normals and len(pcd.points) >= 30:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 4, 0.015), max_nn=20))
    return pcd


def to_open3d(cloud: OrganizedCloud):
    import open3d as o3d

    pts, cols = cloud.flatten()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd
