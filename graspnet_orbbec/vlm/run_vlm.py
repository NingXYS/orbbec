#!/usr/bin/env python3
"""单图测试入口（离线静态图，不接相机）。

评测只用 vlm/testdata 里的固定 5 张图，不会扫描 output/ 历史采集。
结果默认覆盖写入 output/vlm/latest，避免每次跑出一堆时间戳目录。

示例：
  export DASHSCOPE_API_KEY=...
  python vlm/run_vlm.py --image vlm/testdata/01_floor.png --target 蓝色湿巾包
  python vlm/run_vlm.py --eval
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
import yaml

from common.config import load_config
from common.paths import OUTPUT_DIR
from vlm.vlm_client import VLMClient, iou_xyxy, redact_secrets

TESTDATA_DIR = ROOT / "vlm" / "testdata"
DEFAULT_OUT_DIR = OUTPUT_DIR / "vlm" / "latest"

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _put_text(bgr, text: str, org, color=(0, 255, 0)) -> None:
    """用系统 CJK 字体画中文；OpenCV 默认字体没有汉字。"""
    x, y = org
    try:
        from PIL import Image, ImageDraw, ImageFont

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(_CJK_FONT, 22)
        # PIL 颜色是 RGB
        draw.text((x, max(0, y - 22)), text, font=font, fill=(color[2], color[1], color[0]))
        bgr[:, :, :] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(bgr, text.encode("ascii", "replace").decode("ascii"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-VL 单图检测")
    p.add_argument("--config", default=None)
    p.add_argument("--image", default=None, help="输入图路径")
    p.add_argument("--target", default=None, help="目标词")
    p.add_argument("--eval", action="store_true", help="跑 configs/vlm_eval.yaml 验收集")
    p.add_argument("--eval-file", default=None)
    p.add_argument("--iou", type=float, default=0.5, help="框准 IoU 阈值")
    p.add_argument("--max-latency", type=float, default=3.0)
    p.add_argument("--out", default=None, help="结果目录，默认 output/vlm/latest（覆盖上次）")
    return p.parse_args()


def prepare_out_dir(path: Path) -> Path:
    """评测结果只留一份最新，避免 output/vlm 堆历史图。"""
    path.mkdir(parents=True, exist_ok=True)
    for old in path.iterdir():
        if old.is_file() and old.suffix.lower() in {".png", ".json"}:
            old.unlink()
    return path


def resolve_eval_image(raw: str) -> Path:
    """评测图必须落在 vlm/testdata，禁止再去 output/ 或官方示例里捞图。"""
    image = Path(raw)
    if not image.is_absolute():
        image = (ROOT / image).resolve()
    testdata = TESTDATA_DIR.resolve()
    try:
        image.relative_to(testdata)
    except ValueError as exc:
        raise ValueError(f"评测图必须在 {testdata} 内，拒绝 {image}") from exc
    if not image.is_file():
        raise FileNotFoundError(image)
    return image


def draw_result(bgr, result, gt=None):
    vis = bgr.copy()
    if gt is not None:
        gx1, gy1, gx2, gy2 = [int(v) for v in gt]
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)
    if result.found:
        x1, y1, x2, y2 = result.bbox_2d
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _put_text(vis, result.target, (x1, max(24, y1 - 4)), (0, 255, 0))
    else:
        _put_text(vis, f"未找到: {result.target}", (12, 32), (0, 0, 255))
    return vis


def run_one(client: VLMClient, image: Path, target: str, out_dir: Path | None = None):
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image)
    result = client.detect(bgr, target)
    rec = {
        "image": str(image),
        "target": target,
        "found": result.found,
        "bbox_2d": result.bbox_2d,
        "latency_s": result.latency_s,
        "retries": result.retries,
        "model": result.model,
        "image_size": list(result.image_size),
    }
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_result(bgr, result)
        stem = f"{image.stem}_{target}"
        cv2.imwrite(str(out_dir / f"{stem}.png"), vis)
        (out_dir / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, rec


def run_eval(client: VLMClient, eval_path: Path, iou_th: float, max_latency: float, out_dir: Path) -> int:
    spec = yaml.safe_load(eval_path.read_text(encoding="utf-8")) or {}
    cases = spec.get("cases") or []
    rows = []
    n_pos = n_pos_ok = 0
    n_neg = n_neg_ok = 0
    latencies = []
    prepare_out_dir(out_dir)
    unique_images = []
    for case in cases:
        img = resolve_eval_image(str(case["image"]))
        if img not in unique_images:
            unique_images.append(img)
    logging.info("评测只用 testdata %d 张：%s", len(unique_images), ", ".join(p.name for p in unique_images))
    logging.info("结果覆盖写入 %s", out_dir)
    for i, case in enumerate(cases):
        image = resolve_eval_image(str(case["image"]))
        target = str(case["target"])
        expect_found = bool(case.get("found", True))
        gt = case.get("bbox_2d")
        result = client.detect(image, target)
        latencies.append(result.latency_s)
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        vis = draw_result(bgr, result, gt=gt if expect_found else None)
        name = f"{i:02d}_{image.stem}_{target}"
        cv2.imwrite(str(out_dir / f"{name}.png"), vis)
        iou = None
        ok = False
        if not expect_found:
            n_neg += 1
            ok = result.found is False
            if ok:
                n_neg_ok += 1
        else:
            n_pos += 1
            if result.found and gt is not None:
                iou = iou_xyxy(result.bbox_2d, [int(v) for v in gt])
                ok = iou >= iou_th
            if ok:
                n_pos_ok += 1
        row = {
            "image": str(image),
            "target": target,
            "expect_found": expect_found,
            "pred_found": result.found,
            "pred_bbox": result.bbox_2d,
            "gt_bbox": gt,
            "iou": iou,
            "ok": ok,
            "latency_s": result.latency_s,
            "retries": result.retries,
        }
        rows.append(row)
        logging.info(
            "eval %s target=%s expect=%s pred=%s iou=%s ok=%s t=%.3fs",
            image.name,
            target,
            expect_found,
            result.found,
            None if iou is None else f"{iou:.2f}",
            ok,
            result.latency_s,
        )
    pos_acc = n_pos_ok / max(n_pos, 1)
    neg_acc = n_neg_ok / max(n_neg, 1)
    mean_lat = sum(latencies) / max(len(latencies), 1)
    p95 = sorted(latencies)[max(0, int(0.95 * (len(latencies) - 1)))] if latencies else 0.0
    report = {
        "n": len(rows),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_hit": n_pos_ok,
        "negative_hit": n_neg_ok,
        "bbox_acc": pos_acc,
        "negative_acc": neg_acc,
        "mean_latency_s": mean_lat,
        "p95_latency_s": p95,
        "max_latency_s": max(latencies) if latencies else 0.0,
        "pass_bbox": pos_acc >= 0.90,
        "pass_negative": n_neg == 0 or neg_acc == 1.0,
        "pass_latency": bool(latencies) and max(latencies) <= max_latency,
        "iou_threshold": iou_th,
        "model": client.model,
        "cases": rows,
    }
    (out_dir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n", "bbox_acc", "negative_acc", "mean_latency_s", "max_latency_s",
        "pass_bbox", "pass_negative", "pass_latency", "model",
    )}, ensure_ascii=False, indent=2))
    print(f"已保存 {out_dir}")
    ok_all = report["pass_bbox"] and report["pass_negative"] and report["pass_latency"]
    return 0 if ok_all else 2


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    try:
        cfg = load_config(args.config)
        client = VLMClient(cfg)
        out = Path(args.out) if args.out else DEFAULT_OUT_DIR
        prepare_out_dir(out)
        if args.eval:
            eval_path = Path(args.eval_file) if args.eval_file else ROOT / "configs" / "vlm_eval.yaml"
            return run_eval(client, eval_path, args.iou, args.max_latency, out)
        if not args.image or not args.target:
            print("请提供 --image 与 --target，或使用 --eval")
            return 2
        run_one(client, Path(args.image), args.target, out)
        return 0
    except Exception as e:
        logging.error("%s", redact_secrets(f"{type(e).__name__}: {e}"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
