"""GraspNet-baseline 推理封装。只加载预训练权重，不训练。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch

from capture.types import RGBDFrame
from common.config import Config, load_config
from common.paths import GRASPNET_ROOT, WEIGHTS_DIR, ensure_sys_path
from infer.filters import filter_grasps
from infer.preprocess import sample_points
from pointcloud.backproject import OrganizedCloud, display_cloud, frame_to_cloud
from pointcloud.workspace import TablePlane

LOGGER = logging.getLogger("graspnet_orbbec.infer")


def _import_graspnet():
    ensure_sys_path()
    from graspnet import GraspNet, pred_decode  # type: ignore
    from collision_detector import ModelFreeCollisionDetector  # type: ignore
    from graspnetAPI import GraspGroup  # type: ignore

    return GraspNet, pred_decode, ModelFreeCollisionDetector, GraspGroup


class GraspNetInfer:
    def __init__(self, cfg: Optional[Config] = None, checkpoint: Optional[str | Path] = None, device: Optional[str] = None):
        self.cfg = cfg or load_config()
        ckpt = Path(checkpoint or self.cfg.get_path("graspnet.checkpoint", WEIGHTS_DIR / "checkpoint-rs.tar"))
        if not ckpt.is_absolute():
            from common.paths import ROOT

            ckpt = ROOT / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(
                f"找不到权重 {ckpt}。请运行 bash scripts/download_weights.sh"
            )
        self.checkpoint = ckpt
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("需要 NVIDIA GPU 才能跑 GraspNet CUDA 算子")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        GraspNet, pred_decode, ModelFreeCollisionDetector, GraspGroup = _import_graspnet()
        self._GraspNet = GraspNet
        self._pred_decode = pred_decode
        self._Collision = ModelFreeCollisionDetector
        self._GraspGroup = GraspGroup
        num_view = int(self.cfg.get_path("graspnet.num_view", 300))
        net = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self.device)
        # PyTorch 2.6 默认 weights_only=True，旧 checkpoint 必须关
        try:
            ckpt_obj = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        except TypeError:
            ckpt_obj = torch.load(str(ckpt), map_location=self.device)
        net.load_state_dict(ckpt_obj["model_state_dict"])
        net.eval()
        self.net = net
        self._cached_plane: Optional[TablePlane] = None
        self._plane_ttl = 0
        LOGGER.info("已加载 GraspNet 权重 %s (epoch %s)", ckpt, ckpt_obj.get("epoch", "?"))

    def infer_points(self, cloud_sampled: np.ndarray, color_sampled: np.ndarray, scene_points: np.ndarray):
        """对已采样点云推理，返回 GraspGroup。"""
        end_points = {}
        tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device)
        end_points["point_clouds"] = tensor
        end_points["cloud_colors"] = color_sampled
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self._pred_decode(end_points)
        gg = self._GraspGroup(grasp_preds[0].detach().cpu().numpy())
        collision_thresh = float(self.cfg.get_path("graspnet.collision_thresh", 0.01))
        voxel = float(self.cfg.get_path("graspnet.voxel_size", 0.01))
        if collision_thresh > 0 and len(gg) > 0:
            detector = self._Collision(scene_points.astype(np.float32), voxel_size=voxel)
            collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]
        if len(gg) > 0:
            gg.nms()
            gg.sort_by_score()
        return gg

    def infer_frame(self, frame: RGBDFrame) -> dict:
        # 深度范围与可视化体素走配置，网络与过滤逻辑不变
        min_z = float(self.cfg.get_path("camera.min_depth_m", 0.05))
        max_z = float(self.cfg.get_path("workspace.z_max", 2.0))
        z_min_ws = float(self.cfg.get_path("workspace.z_min", 0.15))
        disp_voxel = float(self.cfg.get_path("graspnet.display_voxel", 0.004))
        cloud = frame_to_cloud(frame, min_depth=min_z, max_depth=max_z)
        try:
            reuse = self._cached_plane if self._plane_ttl > 0 else None
            sampled_xyz, sampled_rgb, mask, plane = sample_points(cloud, self.cfg, plane=reuse)
            reuse_n = int(self.cfg.get_path("workspace.plane_reuse_frames", 8))
            if plane is not None:
                self._cached_plane = plane
                self._plane_ttl = reuse_n if self._plane_ttl <= 0 else self._plane_ttl - 1
            else:
                self._cached_plane = None
                self._plane_ttl = 0
        except RuntimeError:
            pcd = display_cloud(cloud, zmin=z_min_ws, zmax=max_z, voxel=disp_voxel)
            empty = self._GraspGroup()
            return {
                "grasps": empty,
                "grasps_filtered": empty,
                "grasps_all": empty,
                "cloud": pcd,
                "organized": cloud,
                "workspace_mask": cloud.valid,
                "table_plane": None,
                "frame": frame,
            }
        scene_xyz = cloud.xyz[mask]
        # 1280x800 工作区点远多于旧 640x480，碰撞检测前把规模收到旧相机量级
        max_col = int(self.cfg.get_path("graspnet.collision_max_points", 40000))
        if scene_xyz.shape[0] > max_col:
            rng = np.random.default_rng(int(self.cfg.get_path("graspnet.seed", 42)))
            scene_xyz = scene_xyz[rng.choice(scene_xyz.shape[0], max_col, replace=False)]
        gg = self.infer_points(sampled_xyz, sampled_rgb, scene_xyz)
        gg_f = filter_grasps(gg, plane=plane, cfg=self.cfg)
        top_k = int(self.cfg.get_path("graspnet.top_k", 50))
        if len(gg_f) > top_k:
            gg_show = gg_f[:top_k]
        else:
            gg_show = gg_f
        # 可视化只喂工作区点，避免 1280x800 全图体素化拖慢循环
        vis_cloud = OrganizedCloud(xyz=cloud.xyz, rgb=cloud.rgb, valid=mask)
        pcd = display_cloud(vis_cloud, zmin=z_min_ws, zmax=max_z, voxel=disp_voxel)
        return {
            "grasps": gg_show,
            "grasps_filtered": gg_f,
            "grasps_all": gg,
            "cloud": pcd,
            "organized": cloud,
            "workspace_mask": mask,
            "table_plane": plane,
            "frame": frame,
        }


def default_checkpoint() -> Path:
    return WEIGHTS_DIR / "checkpoint-rs.tar"
