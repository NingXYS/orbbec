#!/usr/bin/env python3
"""取流稳定性：默认 30 分钟，要求 ≥10 FPS、无崩溃、深度均值漂移可记录。

示例：
  python scripts/run_stability.py --seconds 1800
  python scripts/run_stability.py --seconds 60   # 冒烟
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

from capture.camera import open_camera
from common.config import load_config
from common.paths import OUTPUT_DIR, timestamp_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB-D 取流稳定性测试")
    p.add_argument("--seconds", type=float, default=1800.0)
    p.add_argument("--config", default=None)
    p.add_argument("--min-fps", type=float, default=15.0, help="验收帧率下限（336L ≥15）")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    cam = open_camera(cfg)
    cam.open()
    t0 = time.time()
    n = 0
    errors = 0
    means = []
    last_log = t0
    try:
        while time.time() - t0 < args.seconds:
            try:
                frame = cam.read()
                n += 1
                valid = frame.depth_m[frame.depth_m > 0]
                means.append(float(valid.mean()) if valid.size else 0.0)
            except Exception as e:
                errors += 1
                logging.exception("读帧失败: %s", e)
                time.sleep(0.05)
            now = time.time()
            if now - last_log >= 10:
                fps = n / max(now - t0, 1e-6)
                logging.info("已运行 %.0fs 帧=%d FPS=%.2f 错误=%d", now - t0, n, fps, errors)
                last_log = now
    finally:
        cam.close()
    dt = time.time() - t0
    fps = n / max(dt, 1e-6)
    drift_mm = (max(means) - min(means)) * 1000 if means else None
    out = timestamp_dir(OUTPUT_DIR / "stability")
    report = {
        "seconds": dt,
        "frames": n,
        "fps": fps,
        "errors": errors,
        "depth_mean_drift_mm": drift_mm,
        "depth_mean_first_mm": None if not means else means[0] * 1000,
        "depth_mean_last_mm": None if not means else means[-1] * 1000,
        "pass_fps": fps >= args.min_fps and errors == 0,
        "min_fps": args.min_fps,
    }
    (out / "stability.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"报告: {out / 'stability.json'}")
    if fps < args.min_fps or errors > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
