#!/usr/bin/env python3
"""独立入口：RGB-D 取流预览 / 录制 / 测 FPS / 对齐检验。

相机由 configs/default.yaml 的 camera.backend 决定（gemini336l | astra）。

示例：
  python capture/run_capture.py --preview
  python capture/run_capture.py --save-dir output/capture_raw --n 20
  python capture/run_capture.py --fps-seconds 30
  python capture/run_capture.py --align-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from capture.align_vis import edge_overlay, make_align_panel, measure_edge_alignment_px, overlay_rgbd
from capture.camera import open_camera, save_rgbd_dir
from capture.gemini336l import ALIGN_PASS_PX
from common.config import load_config
from common.paths import OUTPUT_DIR, timestamp_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB-D 取流（Gemini 336L / Astra）")
    p.add_argument("--config", default=None, help="YAML 配置路径")
    p.add_argument("--from-dir", default=None, help="离线目录，不接相机")
    p.add_argument("--preview", action="store_true", help="打开预览窗口")
    p.add_argument("--save-dir", default=None, help="保存帧的目录")
    p.add_argument("--n", type=int, default=1, help="保存帧数")
    p.add_argument("--fps-seconds", type=float, default=0, help="测 FPS 的时长（秒）")
    p.add_argument("--align-check", action="store_true", help="保存边缘叠加图并报告对齐偏差")
    p.add_argument("--align-frames", type=int, default=8, help="对齐检验平均帧数")
    p.add_argument("--min-fps", type=float, default=15.0, help="--fps-seconds 验收帧率下限")
    p.add_argument("--no-window", action="store_true", help="不弹窗")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    cam = open_camera(cfg, data_dir=args.from_dir)
    cam.open()
    try:
        if args.fps_seconds > 0:
            n = 0
            t0 = time.time()
            depths = []
            while time.time() - t0 < args.fps_seconds:
                frame = cam.read()
                n += 1
                valid = frame.depth_m[frame.depth_m > 0]
                depths.append(float(valid.mean()) if valid.size else 0.0)
            dt = time.time() - t0
            fps = n / max(dt, 1e-6)
            drift_mm = (max(depths) - min(depths)) * 1000 if depths else 0.0
            print(f"帧数={n} 时长={dt:.1f}s FPS={fps:.2f} 深度均值漂移={drift_mm:.2f} mm")
            return 0 if fps >= args.min_fps else 2

        save_root = Path(args.save_dir) if args.save_dir else None
        if save_root is None and (not args.preview or args.align_check):
            save_root = timestamp_dir(OUTPUT_DIR / "capture")

        saved = 0
        while True:
            frame = cam.read()
            vis = overlay_rgbd(frame.color_bgr, frame.depth_m)
            valid = int((frame.depth_m > 0).sum())
            text = (
                f"{frame.backend} {frame.color_bgr.shape[1]}x{frame.color_bgr.shape[0]} "
                f"valid={valid} aligned={frame.aligned}"
            )
            cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if args.align_check and save_root is not None and saved < args.n:
                errs = []
                last = frame
                n_avg = max(1, int(args.align_frames))
                for i in range(n_avg):
                    if i > 0:
                        last = cam.read()
                    e = measure_edge_alignment_px(last.color_bgr, last.depth_m)
                    if e is not None:
                        errs.append(e)
                vis = overlay_rgbd(last.color_bgr, last.depth_m)
                valid = int((last.depth_m > 0).sum())
                panel, err_px = make_align_panel(last.color_bgr, last.depth_m)
                edges, _ = edge_overlay(last.color_bgr, last.depth_m)
                err_med = float(np.median(errs)) if errs else err_px
                out = save_root if args.n == 1 else save_root / f"{saved:04d}"
                save_rgbd_dir(last, out)
                cv2.imwrite(str(Path(out) / "overlay.png"), vis)
                cv2.imwrite(str(Path(out) / "align_panel.png"), panel)
                cv2.imwrite(str(Path(out) / "edge_overlay.png"), edges)
                report = {
                    "backend": last.backend,
                    "aligned": last.aligned,
                    "edge_align_px": err_med,
                    "edge_align_px_frames": errs,
                    "n_frames": n_avg,
                    "pass_align_3px": err_med is not None and err_med <= ALIGN_PASS_PX,
                    "valid_depth": valid,
                    "mean_depth_m": float(last.depth_m[last.depth_m > 0].mean()) if valid else 0.0,
                    "shape": [int(last.color_bgr.shape[1]), int(last.color_bgr.shape[0])],
                    "device": last.extra.get("device", {}),
                    "profile": last.extra.get("profile", {}),
                    "note": "验收标准为 0.5 m 处 ≤3 px；当前值为几何边缘互相对应中位数",
                }
                (Path(out) / "align.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(json.dumps(report, indent=2, ensure_ascii=False))
                print(f"已保存对齐检验 {out}")
                saved += 1
                if saved >= args.n and not args.preview:
                    break
            elif save_root is not None and saved < args.n:
                out = save_root if args.n == 1 else save_root / f"{saved:04d}"
                save_rgbd_dir(frame, out)
                cv2.imwrite(str(Path(out) / "overlay.png"), vis)
                print(f"已保存 {out}")
                saved += 1
                if saved >= args.n and not args.preview:
                    break
            if args.preview and not args.no_window:
                cv2.imshow("RGB-D", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    out = timestamp_dir(OUTPUT_DIR / "capture")
                    save_rgbd_dir(frame, out)
                    print(f"已保存 {out}")
            if not args.preview and saved >= args.n:
                break
        cv2.destroyAllWindows()
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
