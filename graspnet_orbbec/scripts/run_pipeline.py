#!/usr/bin/env python3
"""阶段一端到端：取流 → 点云 → GraspNet → 可视化。

示例：
  python scripts/run_pipeline.py --live              # 连续规划 GUI
  python scripts/run_pipeline.py --live --once       # 只拍一帧
  python scripts/run_pipeline.py --official-demo
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

from capture.camera import open_camera, save_rgbd_dir
from common.config import load_config
from common.paths import GRASPNET_ROOT, OUTPUT_DIR, timestamp_dir
from infer.filters import grasp_to_dicts
from infer.graspnet_infer import GraspNetInfer
from pointcloud.quality import evaluate_frame
from viz.visualize import save_triplet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraspNet + Orbbec 阶段一流水线")
    p.add_argument("--config", default=None)
    p.add_argument("--from-dir", default=None)
    p.add_argument("--live", action="store_true", help="打开连续规划 GUI（相机常开）")
    p.add_argument("--once", action="store_true", help="只拍一帧并保存，不打开 GUI")
    p.add_argument("--official-demo", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    # --live 默认进入连续 GUI；--once 才是单次拍照
    if args.live and not args.once:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_live_gui import main as gui_main  # noqa: WPS433

        sys.argv = [sys.argv[0]]
        if args.config:
            sys.argv += ["--config", args.config]
        if args.from_dir:
            sys.argv += ["--from-dir", args.from_dir]
        if args.official_demo:
            sys.argv.append("--official-demo")
        if args.checkpoint:
            sys.argv += ["--checkpoint", args.checkpoint]
        return gui_main()

    cfg = load_config(args.config)
    data_dir = args.from_dir
    if args.official_demo:
        data_dir = str(GRASPNET_ROOT / "doc" / "example_data")
    cam = open_camera(cfg, data_dir=data_dir)
    cam.open()
    try:
        frame = cam.read()
    finally:
        cam.close()

    quality = evaluate_frame(frame)
    engine = GraspNetInfer(cfg, checkpoint=args.checkpoint)
    result = engine.infer_frame(frame)
    gg = result["grasps"]

    out = Path(args.out) if args.out else timestamp_dir(OUTPUT_DIR)
    save_rgbd_dir(frame, out)
    paths = save_triplet(frame, result["cloud"], gg, out)
    report = {
        "backend": frame.backend,
        "aligned": frame.aligned,
        "resolution": [int(frame.color_bgr.shape[1]), int(frame.color_bgr.shape[0])],
        "intrinsics": frame.color_intrinsics.to_dict(),
        "quality": {
            "valid_ratio": quality.valid_ratio,
            "hole_ratio": quality.hole_ratio,
            "mean_depth_m": quality.mean_depth_m,
            "edge_align_px": quality.edge_align_px,
            "aabb_m": None if quality.block_size_m is None else quality.block_size_m.tolist(),
            "notes": quality.notes,
        },
        "num_grasps": int(len(gg)),
        "num_grasps_before_filter": int(len(result["grasps_all"])),
        "top": grasp_to_dicts(gg, top_k=20),
        "images": {k: str(v) for k, v in paths.items()},
        "checkpoint": str(engine.checkpoint),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("==== 阶段一流水线 ====")
    print(f"输出目录: {out}")
    print(f"RGB: {paths['rgb']}")
    print(f"点云: {paths['cloud']}")
    print(f"抓取: {paths['grasps']}")
    print(f"抓取数: {len(gg)}  有效深度 {quality.valid_ratio*100:.1f}%")
    if quality.edge_align_px is not None:
        print(f"边缘对齐偏差: {quality.edge_align_px:.2f} px")
    if args.show:
        import open3d as o3d

        geoms = [result["cloud"]]
        if len(gg) > 0:
            geoms.extend(gg.to_open3d_geometry_list())
        o3d.visualization.draw_geometries(geoms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
