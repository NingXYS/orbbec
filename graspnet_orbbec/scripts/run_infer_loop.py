#!/usr/bin/env python3
"""336L → GraspNet 连续推理（无 GUI），测 FPS 并落盘对比图。

示例：
  python scripts/run_infer_loop.py --seconds 20
  python scripts/run_infer_loop.py --trials 10 --save-each
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

from capture.align_vis import hole_overlay, overlay_rgbd
from capture.camera import open_camera, save_rgbd_dir
from common.config import load_config
from common.paths import OUTPUT_DIR, timestamp_dir
from infer.filters import grasp_to_dicts
from infer.graspnet_infer import GraspNetInfer
from viz.visualize import overlay_grasps_on_rgb, save_triplet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemini 336L GraspNet 连续推理")
    p.add_argument("--config", default=None)
    p.add_argument("--from-dir", default=None)
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--trials", type=int, default=0, help=">0 时改为固定次数（用于 10 次摆放验收）")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--min-fps", type=float, default=5.0)
    p.add_argument("--save-each", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _one_summary(result, infer_ms: float) -> dict:
    gg = result["grasps"]
    n = int(len(gg)) if gg is not None else 0
    top = float(np.asarray(gg.scores)[0]) if n else 0.0
    frame = result["frame"]
    valid = float((frame.depth_m > 0).mean())
    return {
        "infer_ms": infer_ms,
        "num_grasps": n,
        "top_score": top,
        "valid_ratio": valid,
        "shape": [int(frame.color_bgr.shape[1]), int(frame.color_bgr.shape[0])],
        "aligned": bool(frame.aligned),
        "backend": frame.backend,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    cam = open_camera(cfg, data_dir=args.from_dir)
    cam.open()
    engine = GraspNetInfer(cfg)
    out = Path(args.out) if args.out else timestamp_dir(OUTPUT_DIR / "gemini_grasp")
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    last = None
    n_ok = 0
    try:
        for i in range(max(0, args.warmup)):
            frame = cam.read()
            t0 = time.time()
            last = engine.infer_frame(frame)
            logging.info("warmup %d  %.0f ms  grasps=%d", i, (time.time() - t0) * 1000, len(last["grasps"]))

        t_all = time.time()
        if args.trials > 0:
            n_target = args.trials
            timed = False
        else:
            n_target = 10**9
            timed = True
        i = 0
        while i < n_target:
            if timed and (time.time() - t_all) >= args.seconds:
                break
            frame = cam.read()
            t0 = time.time()
            last = engine.infer_frame(frame)
            dt = (time.time() - t0) * 1000.0
            rec = _one_summary(last, dt)
            summaries.append(rec)
            n_ok += 1
            i += 1
            logging.info("trial %d  %.0f ms  grasps=%d  top=%.3f  depth=%.0f%%", i, dt, rec["num_grasps"], rec["top_score"], rec["valid_ratio"] * 100)
            if args.save_each or args.trials > 0:
                sub = out / f"{i:02d}"
                save_rgbd_dir(frame, sub)
                save_triplet(frame, last["cloud"], last["grasps"], sub, top_k=20)
                holes, _ = hole_overlay(frame.color_bgr, frame.depth_m)
                cv2.imwrite(str(sub / "holes.png"), holes)
                rgb_g = overlay_grasps_on_rgb(frame.color_bgr, last["grasps"], frame.color_intrinsics, top_k=8)
                cv2.imwrite(str(sub / "05_rgb_grasps.png"), rgb_g)
    finally:
        cam.close()

    elapsed = time.time() - t_all if summaries else 1e-6
    mean_ms = float(np.mean([s["infer_ms"] for s in summaries])) if summaries else 0.0
    infer_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    loop_fps = len(summaries) / max(elapsed, 1e-6)
    grasp_ok = [s for s in summaries if s["num_grasps"] > 0]
    report = {
        "n": len(summaries),
        "elapsed_s": elapsed,
        "infer_fps": infer_fps,
        "loop_fps": loop_fps,
        "mean_infer_ms": mean_ms,
        "pass_fps": infer_fps >= args.min_fps,
        "trials_with_grasp": len(grasp_ok),
        "grasp_ratio": len(grasp_ok) / max(len(summaries), 1),
        "summaries": summaries,
        "top": grasp_to_dicts(last["grasps"], top_k=10) if last is not None else [],
    }
    if last is not None:
        frame = last["frame"]
        save_rgbd_dir(frame, out)
        save_triplet(frame, last["cloud"], last["grasps"], out, top_k=20)
        holes, hstats = hole_overlay(frame.color_bgr, frame.depth_m)
        cv2.imwrite(str(out / "holes.png"), holes)
        cv2.imwrite(str(out / "overlay.png"), overlay_rgbd(frame.color_bgr, frame.depth_m))
        rgb_g = overlay_grasps_on_rgb(frame.color_bgr, last["grasps"], frame.color_intrinsics, top_k=8)
        cv2.imwrite(str(out / "05_rgb_grasps.png"), rgb_g)
        report["holes"] = hstats
        report["device"] = frame.extra.get("device", {})
        report["profile"] = frame.extra.get("profile", {})
    (out / "infer_loop.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "elapsed_s", "infer_fps", "loop_fps", "mean_infer_ms", "pass_fps", "trials_with_grasp", "grasp_ratio")}, indent=2, ensure_ascii=False))
    print(f"已保存 {out}")
    return 0 if report["pass_fps"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
