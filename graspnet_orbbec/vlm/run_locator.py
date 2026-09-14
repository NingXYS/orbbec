#!/usr/bin/env python3
"""离线 bbox→3D 质心。只读已保存的对齐帧，不接相机。

示例：
  python vlm/run_locator.py --frame vlm/locator_data/desk --bbox 550,513,864,651 --target 机械键盘
  python vlm/run_locator.py --eval
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
import yaml

from capture.camera import load_rgbd_dir
from common.config import load_config
from common.paths import OUTPUT_DIR
from vlm.target_locator import (
    InsufficientDepthError,
    centroid_in_bbox,
    draw_depth_error,
    draw_locate_overlay,
    locate_target,
)

DEFAULT_OUT = OUTPUT_DIR / "locator" / "latest"
DEFAULT_EVAL = ROOT / "configs" / "locator_eval.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bbox → 3D 质心（离线对齐帧）")
    p.add_argument("--config", default=None)
    p.add_argument("--frame", default=None, help="对齐帧目录（含 color.png/depth.png/intrinsics.json）")
    p.add_argument("--bbox", default=None, help="x1,y1,x2,y2 像素")
    p.add_argument("--target", default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-ms", type=float, default=50.0)
    return p.parse_args()


def prepare_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.iterdir():
        if old.is_file() and old.suffix.lower() in {".png", ".json"}:
            old.unlink()
    return path


def parse_bbox(text: str) -> list[int]:
    parts = [p.strip() for p in str(text).replace("x", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"bbox 需要 4 个数，得到 {text}")
    return [int(float(p)) for p in parts]


def run_one(frame_dir: Path, bbox: list[int], target: str, cfg, out_dir: Path | None):
    frame = load_rgbd_dir(frame_dir)
    result = locate_target(frame, bbox, cfg=cfg, target=target)
    rec = result.to_dict()
    rec["frame"] = str(frame_dir)
    rec["inside_bbox"] = centroid_in_bbox(result)
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_locate_overlay(frame.color_bgr, result, depth_m=frame.depth_m)
        stem = f"{frame_dir.name}_{target or 'target'}"
        cv2.imwrite(str(out_dir / f"{stem}.png"), vis)
        (out_dir / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, rec


def run_eval(eval_path: Path, cfg, out_dir: Path, max_ms: float) -> int:
    spec = yaml.safe_load(eval_path.read_text(encoding="utf-8")) or {}
    cases = spec.get("cases") or []
    prepare_out(out_dir)
    rows = []
    n_pos = n_pos_ok = 0
    n_neg = n_neg_ok = 0
    times = []
    for i, case in enumerate(cases):
        frame_dir = Path(case["frame"])
        if not frame_dir.is_absolute():
            frame_dir = ROOT / frame_dir
        target = str(case.get("target", ""))
        bbox = [int(v) for v in case["bbox_2d"]]
        expect_ok = bool(case.get("expect_ok", True))
        frame = load_rgbd_dir(frame_dir)
        rec = {
            "frame": str(frame_dir),
            "target": target,
            "bbox_2d": bbox,
            "expect_ok": expect_ok,
        }
        try:
            result = locate_target(frame, bbox, cfg=cfg, target=target)
            vis = draw_locate_overlay(frame.color_bgr, result, depth_m=frame.depth_m)
            cv2.imwrite(str(out_dir / f"{i:02d}_{frame_dir.name}_{target}.png"), vis)
            rec.update(result.to_dict())
            rec["inside_bbox"] = centroid_in_bbox(result)
            rec["error"] = None
            times.append(result.elapsed_ms)
            if expect_ok:
                n_pos += 1
                ok = rec["inside_bbox"]
                if ok:
                    n_pos_ok += 1
            else:
                n_neg += 1
                ok = False
            rec["ok"] = bool(ok) if expect_ok else False
        except InsufficientDepthError as e:
            rec["error"] = str(e)
            rec["inside_bbox"] = False
            rec["elapsed_ms"] = None
            vis = draw_depth_error(frame.color_bgr, bbox, str(e))
            cv2.imwrite(str(out_dir / f"{i:02d}_{frame_dir.name}_{target}.png"), vis)
            if not expect_ok:
                n_neg += 1
                n_neg_ok += 1
                rec["ok"] = True
            else:
                n_pos += 1
                rec["ok"] = False
        logging.info(
            "locate %s target=%s expect_ok=%s ok=%s xyz=%s uv=%s t=%s",
            frame_dir.name,
            target,
            expect_ok,
            rec.get("ok"),
            rec.get("xyz_m"),
            rec.get("uv"),
            rec.get("elapsed_ms"),
        )
        rows.append(rec)
    inside_acc = n_pos_ok / max(n_pos, 1)
    neg_acc = n_neg_ok / max(n_neg, 1)
    mean_ms = sum(times) / max(len(times), 1)
    max_elapsed = max(times) if times else 0.0
    report = {
        "n": len(rows),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "inside_acc": inside_acc,
        "negative_acc": neg_acc,
        "mean_ms": mean_ms,
        "max_ms": max_elapsed,
        "pass_inside": inside_acc >= 0.90,
        "pass_negative": n_neg == 0 or neg_acc == 1.0,
        "pass_latency": (not times) or max_elapsed <= max_ms,
        "cases": rows,
    }
    (out_dir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n", "inside_acc", "negative_acc", "mean_ms", "max_ms",
        "pass_inside", "pass_negative", "pass_latency",
    )}, ensure_ascii=False, indent=2))
    print(f"已保存 {out_dir}")
    ok_all = report["pass_inside"] and report["pass_negative"] and report["pass_latency"]
    return 0 if ok_all else 2


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.eval:
        eval_path = Path(args.eval_file) if args.eval_file else DEFAULT_EVAL
        return run_eval(eval_path, cfg, out, args.max_ms)
    if not args.frame or not args.bbox:
        print("请提供 --frame 与 --bbox，或使用 --eval")
        return 2
    prepare_out(out)
    try:
        run_one(Path(args.frame), parse_bbox(args.bbox), args.target, cfg, out)
        return 0
    except InsufficientDepthError as e:
        logging.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
