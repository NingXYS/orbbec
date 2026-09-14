#!/usr/bin/env python3
"""标定物尺寸与 RGB-D 对齐检查。

把 50mm 标准块放到画面中央桌面上。Gemini 336L 建议约 0.5 m（验收 ≤3 mm / ≤3 px）；Astra Pro Plus 约 0.6–0.8 m。
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

from capture.align_vis import make_align_panel
from capture.camera import open_camera, save_rgbd_dir
from capture.gemini336l import ALIGN_PASS_PX, BLOCK_PASS_MM
from common.config import load_config
from common.paths import OUTPUT_DIR, timestamp_dir
from pointcloud.quality import evaluate_frame
from viz.visualize import save_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="点云尺寸 / 对齐验收")
    p.add_argument("--from-dir", default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--expect-block-mm", type=float, default=50.0)
    p.add_argument("--max-align-px", type=float, default=ALIGN_PASS_PX, help="边缘对齐验收阈值（336L ≤3 px）")
    p.add_argument("--max-size-err-mm", type=float, default=BLOCK_PASS_MM, help="50mm 块尺寸误差阈值（336L ≤3 mm）")
    p.add_argument("--config", default=None)
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
    report = evaluate_frame(frame, expected_block_m=args.expect_block_mm / 1000.0)
    out = timestamp_dir(OUTPUT_DIR / "calib")
    save_rgbd_dir(frame, out)
    save_rgb(frame.color_bgr, out / "01_rgb.png")
    panel, _ = make_align_panel(frame.color_bgr, frame.depth_m)
    cv2.imwrite(str(out / "align_panel.png"), panel)
    payload = {
        "valid_ratio": report.valid_ratio,
        "hole_ratio": report.hole_ratio,
        "mean_depth_m": report.mean_depth_m,
        "edge_align_px": report.edge_align_px,
        "aabb_mm": None if report.block_size_m is None else (report.block_size_m * 1000).tolist(),
        "expect_block_mm": args.expect_block_mm,
        "notes": report.notes,
        "pass_align": report.edge_align_px is not None and report.edge_align_px <= args.max_align_px,
        "max_align_px": args.max_align_px,
    }
    if report.block_size_m is not None:
        # 用与 50mm 最接近的边评估
        sides = sorted(report.block_size_m.tolist())
        err_mm = min(abs(s * 1000 - args.expect_block_mm) for s in sides)
        payload["size_error_mm"] = err_mm
        payload["pass_size"] = err_mm <= args.max_size_err_mm
        payload["max_size_err_mm"] = args.max_size_err_mm
    (out / "calib.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
