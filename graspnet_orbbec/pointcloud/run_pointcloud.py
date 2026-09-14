#!/usr/bin/env python3
"""独立入口：从相机或目录生成点云，评估尺寸与对齐。

示例：
  python pointcloud/run_pointcloud.py --from-dir output/capture/<ts>
  python pointcloud/run_pointcloud.py --live --save
  python pointcloud/run_pointcloud.py --from-dir ... --expect-block-mm 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d

from capture.camera import open_camera, save_rgbd_dir
from common.config import load_config
from common.paths import OUTPUT_DIR, timestamp_dir
from pointcloud.backproject import frame_to_cloud, to_open3d
from pointcloud.quality import evaluate_frame
from pointcloud.workspace import workspace_mask
from viz.visualize import render_pointcloud, save_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB-D 反投影点云")
    p.add_argument("--config", default=None)
    p.add_argument("--from-dir", default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--expect-block-mm", type=float, default=None, help="已知标定块边长，毫米")
    p.add_argument("--show", action="store_true", help="弹出 Open3D 窗口")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    cam = open_camera(cfg, data_dir=args.from_dir)
    cam.open()
    try:
        frame = cam.read()
    finally:
        cam.close()

    cloud = frame_to_cloud(frame)
    mask, plane = workspace_mask(cloud, cfg)
    report = evaluate_frame(
        frame,
        expected_block_m=None if args.expect_block_mm is None else args.expect_block_mm / 1000.0,
    )
    ws_cloud = type(cloud)(xyz=cloud.xyz, rgb=cloud.rgb, valid=mask)
    pcd = to_open3d(ws_cloud if mask.any() else cloud)
    print("==== 点云质量 ====")
    print(f"有效比例 {report.valid_ratio*100:.1f}%  空洞率 {report.hole_ratio*100:.1f}%")
    print(f"工作距离均值 {report.mean_depth_m*1000:.1f} mm  std {report.std_depth_m*1000:.1f} mm")
    print(f"点数 {np.asarray(pcd.points).shape[0]}")
    if report.edge_align_px is not None:
        print(f"RGB-D 边缘对齐中位偏差 {report.edge_align_px:.2f} px  (验收 ≤10 px @0.5m)")
    if plane is not None:
        print(f"桌面法向 {plane.normal}  内点数 {plane.inlier_count}")
    if report.block_size_m is not None:
        print(f"最大物体 AABB (mm) = {(report.block_size_m*1000).round(1)}")
    for n in report.notes:
        print("注:", n)

    out = timestamp_dir(OUTPUT_DIR / "pointcloud") if args.save or args.from_dir or args.live else None
    if out is not None:
        save_rgbd_dir(frame, out)
        o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)
        save_rgb(frame.color_bgr, out / "01_rgb.png")
        render_pointcloud(pcd, out / "02_cloud.png")
        # 工作空间着色
        vis = frame.color_bgr.copy()
        vis[~mask] = (vis[~mask] * 0.25).astype(np.uint8)
        cv2.imwrite(str(out / "workspace_overlay.png"), vis)
        meta = {
            "valid_ratio": report.valid_ratio,
            "hole_ratio": report.hole_ratio,
            "mean_depth_m": report.mean_depth_m,
            "edge_align_px": report.edge_align_px,
            "aabb_m": None if report.block_size_m is None else report.block_size_m.tolist(),
            "intrinsics": frame.color_intrinsics.to_dict(),
            "notes": report.notes,
        }
        (out / "quality.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已保存 {out}")

    if args.show:
        o3d.visualization.draw_geometries([pcd])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
