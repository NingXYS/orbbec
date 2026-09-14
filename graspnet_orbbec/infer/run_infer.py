#!/usr/bin/env python3
"""独立入口：对一帧 RGB-D 跑 GraspNet，保存抓取位姿。

示例：
  python infer/run_infer.py --from-dir output/capture/<ts>
  python infer/run_infer.py --live
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
from common.paths import OUTPUT_DIR, timestamp_dir
from infer.filters import grasp_to_dicts
from infer.graspnet_infer import GraspNetInfer
from viz.visualize import save_triplet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraspNet 推理")
    p.add_argument("--config", default=None)
    p.add_argument("--from-dir", default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--show", action="store_true")
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

    engine = GraspNetInfer(cfg, checkpoint=args.checkpoint)
    result = engine.infer_frame(frame)
    gg = result["grasps"]
    print(f"碰撞+规则过滤后抓取数: {len(gg)}  (过滤前 {len(result['grasps_all'])})")
    if len(gg) > 0:
        print("Top-5 score/width/translation:")
        for i, g in enumerate(grasp_to_dicts(gg, top_k=5)):
            print(f"  [{i}] score={g['score']:.3f} width={g['width_m']*1000:.1f}mm t={np_round(g['translation_m'])}")

    out = Path(args.out) if args.out else timestamp_dir(OUTPUT_DIR / "infer")
    save_rgbd_dir(frame, out)
    save_triplet(frame, result["cloud"], gg, out)
    payload = {
        "num_grasps": int(len(gg)),
        "num_grasps_before_filter": int(len(result["grasps_all"])),
        "top": grasp_to_dicts(gg, top_k=int(cfg.get_path("graspnet.top_k", 50))),
        "intrinsics": frame.color_intrinsics.to_dict(),
        "checkpoint": str(engine.checkpoint),
    }
    (out / "grasps.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 {out}")
    if args.show:
        import open3d as o3d

        geoms = [result["cloud"]]
        if len(gg) > 0:
            geoms.extend(gg.to_open3d_geometry_list())
        o3d.visualization.draw_geometries(geoms)
    return 0


def np_round(v):
    return [round(float(x), 4) for x in v]


if __name__ == "__main__":
    raise SystemExit(main())
