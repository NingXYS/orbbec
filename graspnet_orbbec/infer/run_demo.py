#!/usr/bin/env python3
"""跑通官方 GraspNet example_data demo（不接相机）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.paths import GRASPNET_ROOT, OUTPUT_DIR, timestamp_dir
from infer.run_infer import main as infer_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="官方 example_data 推理")
    p.add_argument(
        "--data-dir",
        default=str(GRASPNET_ROOT / "doc" / "example_data"),
        help="官方 RGB-D 示例目录",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not (data_dir / "color.png").exists():
        raise FileNotFoundError(f"找不到官方示例 {data_dir}")
    # 复用 run_infer 的 CLI
    sys.argv = [
        "run_infer.py",
        "--from-dir",
        str(data_dir),
        "--out",
        str(timestamp_dir(OUTPUT_DIR / "official_demo")),
    ]
    if args.checkpoint:
        sys.argv += ["--checkpoint", args.checkpoint]
    if args.show:
        sys.argv.append("--show")
    return infer_main()


if __name__ == "__main__":
    raise SystemExit(main())
