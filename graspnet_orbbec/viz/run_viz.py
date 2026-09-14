#!/usr/bin/env python3
"""独立入口：对已保存的 RGB-D / 抓取结果做可视化。

示例：
  python viz/run_viz.py --from-dir output/infer/<ts>
  python viz/run_viz.py --from-dir output/capture/<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d

from capture.camera import load_rgbd_dir
from common.paths import OUTPUT_DIR, timestamp_dir, ensure_sys_path
from pointcloud.backproject import frame_to_cloud, to_open3d
from viz.visualize import save_triplet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化 RGB / 点云 / 抓取")
    p.add_argument("--from-dir", required=True, help="含 color.png/depth.png 的目录")
    p.add_argument("--out", default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _load_grasps(data_dir: Path):
    json_path = data_dir / "grasps.json"
    if not json_path.exists():
        return None
    ensure_sys_path()
    from graspnetAPI import GraspGroup  # type: ignore

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    items = payload.get("top") or []
    if not items:
        return GraspGroup()
    rows = []
    for g in items:
        score = g["score"]
        width = g["width_m"]
        height = 0.02
        depth = g.get("depth_m", 0.02)
        R = np.array(g["rotation"], dtype=np.float32).reshape(9)
        t = np.array(g["translation_m"], dtype=np.float32)
        obj = -1.0
        rows.append(np.concatenate([[score, width, height, depth], R, t, [obj]]))
    return GraspGroup(np.stack(rows, axis=0))


def main() -> int:
    args = parse_args()
    data_dir = Path(args.from_dir)
    frame = load_rgbd_dir(data_dir)
    cloud = frame_to_cloud(frame)
    pcd = to_open3d(cloud)
    gg = _load_grasps(data_dir)
    out = Path(args.out) if args.out else timestamp_dir(OUTPUT_DIR / "viz")
    save_triplet(frame, pcd, gg, out)
    print(f"已保存 {out}")
    if args.show:
        geoms = [pcd]
        if gg is not None and len(gg) > 0:
            geoms.extend(gg.to_open3d_geometry_list())
        o3d.visualization.draw_geometries(geoms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
