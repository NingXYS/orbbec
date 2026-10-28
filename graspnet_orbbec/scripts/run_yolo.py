#!/usr/bin/env python3
"""现有 YOLO 权重效果检验：Orbbec、UVC、图片目录或视频。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".yolo-deps"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "output/yolo_config"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WINDOW = "YOLO segmentation | q: quit | s: save"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT.parent / "best.pt")
    p.add_argument("--label", default="线材金属端子", help="单类别模型显示名称；传空字符串保留权重原名")
    p.add_argument("--source", default="orbbec", help="orbbec、UVC 编号（如 0）、图片、图片目录或视频")
    p.add_argument("--config", help="Orbbec 配置 YAML，默认沿用项目配置")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto", help="auto / cpu / 0（第一块 GPU）")
    p.add_argument("--threads", type=int, default=4, help="CPU 推理线程数")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 表示读至结束或手动退出")
    p.add_argument("--output", type=Path, help="新建结果目录，不覆盖已存在目录")
    p.add_argument("--info", action="store_true", help="只读取模型信息，不打开相机")
    args = p.parse_args()
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        p.error("conf 必须在 [0,1]，iou 必须在 (0,1]")
    if args.imgsz <= 0 or args.threads <= 0 or args.max_frames < 0:
        p.error("imgsz、threads 必须为正数；max-frames 不能为负数")
    if not args.weights.is_file():
        p.error(f"找不到权重：{args.weights}")
    if not args.info and not args.no_window and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        p.error("没有图形显示环境，请加 --no-window")
    return args


def orbbec_frames(config):
    from capture.color_convert import frame_to_bgr
    from common.config import load_config

    cfg = load_config(config)
    backend = str(cfg.get_path("camera.backend", "auto")).lower()
    if backend == "auto":
        from capture.usb_detect import detect_capture_backend

        backend = detect_capture_backend()
    if backend in ("gemini336l", "gemini", "gemini330", "gemini336"):
        from capture.sdk_v2 import import_sdk_v2

        ob = import_sdk_v2()
    elif backend in ("dabai_dc1", "dc1", "astra", "pyorbbec", "astra_pro_plus"):
        from capture.dabai_dc1 import _import_sdk_v1

        ob = _import_sdk_v1()
    else:
        raise ValueError(f"不支持的彩色相机后端：{backend}")
    context = ob.Context()
    devices = context.query_devices()
    serial = str(cfg.get_path("camera.serial", "") or "")
    if devices.get_count() > 1 and not serial:
        raise RuntimeError("检测到多台 Orbbec，请在配置中填写 camera.serial")
    device = None
    for i in range(devices.get_count()):
        candidate = devices.get_device_by_index(i)
        if not serial or candidate.get_device_info().get_serial_number() == serial:
            device = candidate
            break
    if device is None:
        raise RuntimeError(f"找不到 Orbbec 相机（serial={serial or '未指定'}）")
    pipeline = ob.Pipeline(device)
    stream_config = ob.Config()
    profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
        int(cfg.get_path("camera.color_width", 640)), int(cfg.get_path("camera.color_height", 480)),
        getattr(ob.OBFormat, str(cfg.get_path("camera.color_format", "MJPG")).upper()),
        int(cfg.get_path("camera.color_fps", 30)))
    # 二维分割只启用彩色流，不依赖有效深度，也不更改激光或 D2C 设置。
    stream_config.enable_stream(profile)
    try:
        pipeline.start(stream_config)
        info = device.get_device_info()
        print(f"相机：{info.get_name()} serial={info.get_serial_number()}，仅彩色流", flush=True)
        while True:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                pair = pipeline.wait_for_frames(1000)
                color = pair.get_color_frame() if pair is not None else None
                image = frame_to_bgr(color) if color is not None else None
                if image is not None:
                    yield "orbbec", image, False
                    break
            else:
                raise RuntimeError("Orbbec 彩色取流超时，请检查连接、权限和占用")
    finally:
        pipeline.stop()


def frames(source, config):
    if source == "orbbec":
        yield from orbbec_frames(config)
    else:
        path = Path(source).expanduser()
        if not source.isdecimal() and not path.exists():
            raise FileNotFoundError(f"找不到输入：{source}")
        if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
            paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if path.is_dir() else [path]
            if not paths:
                raise ValueError("目录内没有支持的图片（不递归子目录）")
            for p in paths:
                image = cv2.imread(str(p))
                if image is None:
                    raise RuntimeError(f"图片无法解码：{p}")
                yield str(p.resolve()), image, True
        else:
            cap = cv2.VideoCapture(int(source) if source.isdecimal() else str(path))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开输入：{source}；请检查设备连接、权限和占用")
                count = 0
                while True:
                    ok, image = cap.read()
                    if not ok:
                        if source.isdecimal() or count == 0:
                            raise RuntimeError(f"读取输入失败：{source}")
                        break
                    count += 1
                    yield source, image, False
            finally:
                cap.release()


def detections(result):
    items = []
    if result.boxes is None:
        return items
    polygons = result.masks.xy if result.masks is not None else []
    for i, box in enumerate(result.boxes.cpu()):
        cls = int(box.cls.item())
        items.append({
            "class_id": cls, "name": result.names[cls],
            "confidence": float(box.conf.item()),
            "xyxy": box.xyxy[0].tolist(),
            "polygon_xy": polygons[i].tolist() if i < len(polygons) else [],
        })
    return items


def render(image, items, fps, conf):
    from PIL import Image, ImageDraw, ImageFont

    palette = [(0, 220, 0), (255, 160, 0), (0, 180, 255), (220, 0, 220)]
    overlay = image.copy()
    for i, item in enumerate(items):
        polygon = np.asarray(item["polygon_xy"], dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(overlay, [polygon], palette[i % len(palette)])
    vis = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    labels = []
    for i, item in enumerate(items):
        color = palette[i % len(palette)]
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        labels.append((x1, max(26, y1 - 24), f'{item["name"]} {item["confidence"]:.2f}', color))
    # 使用本地字体，避免中文类别触发 Ultralytics 在线字体下载。
    canvas = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for x, y, label, color in labels:
        draw.text((x, y), label, font=font, fill=tuple(reversed(color)), stroke_width=1, stroke_fill=(0, 0, 0))
    vis = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    cv2.putText(vis, f"objects={len(items)} | infer FPS={fps:.1f} | conf={conf:.2f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if float(image.mean()) < 5:
        cv2.putText(vis, "Image very dark: check lens / lighting", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)
    return vis


def save_frame(out, index, image, vis):
    for suffix, data in (("raw", image), ("result", vis)):
        target = out / f"{index:06d}_{suffix}.jpg"
        if not cv2.imwrite(str(target), data):
            raise OSError(f"保存失败：{target}")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = YOLO(str(args.weights.resolve()))
    original_names = dict(model.names)
    if args.label and len(model.names) == 1:
        # 类别语义由用户确认，只改运行时标签，不修改权重文件。
        model.model.names = {next(iter(model.names)): args.label}
    info = {"weights": str(args.weights.resolve()), "task": model.task, "names": model.names,
            "original_names": original_names,
            "device": device, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "checkpoint_version": model.ckpt.get("version"),
            "original_model": model.ckpt.get("train_args", {}).get("model")}
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.info:
        return 0
    out = args.output or ROOT / "output/yolo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    summary = {**info, "source": args.source, "imgsz": args.imgsz, "iou": args.iou,
               "frames": 0, "frames_with_detections": 0, "detections": 0, "completed": False}
    stream = frames(args.source, args.config)
    durations = []
    last = None
    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Confidence %", WINDOW, round(args.conf * 100), 100, lambda _: None)
    started = time.perf_counter()
    try:
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as log:
            for source, image, is_image in stream:
                conf = args.conf if args.no_window else cv2.getTrackbarPos("Confidence %", WINDOW) / 100
                t0 = time.perf_counter()
                result = model.predict(image, conf=conf, iou=args.iou, imgsz=args.imgsz,
                                       device=device, retina_masks=True, verbose=False)[0]
                items = detections(result)
                elapsed = time.perf_counter() - t0
                durations.append(elapsed)
                index = summary["frames"]
                log.write(json.dumps({"frame": index, "source": source, "conf": conf,
                                      "image_size_wh": [image.shape[1], image.shape[0]],
                                      "inference_ms": elapsed * 1000, "detections": items}, ensure_ascii=False) + "\n")
                log.flush()
                summary["frames"] += 1
                summary["frames_with_detections"] += bool(items)
                summary["detections"] += len(items)
                vis = render(image, items, 1 / max(elapsed, 1e-6), conf)
                last = index, image, vis
                if index == 0 or is_image:
                    save_frame(out, index, image, vis)
                print(f"frame={index} objects={len(items)} infer={elapsed*1000:.0f}ms", flush=True)
                if not args.no_window:
                    cv2.imshow(WINDOW, vis)
                    key = cv2.waitKey(0 if is_image else 1) & 0xFF
                    if key == ord("s"):
                        save_frame(out, index, image, vis)
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if args.max_frames and summary["frames"] >= args.max_frames:
                    break
        summary["completed"] = True
    except KeyboardInterrupt:
        summary["stopped_by_user"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        stream.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        if last is not None:
            save_frame(out, *last)
        summary["elapsed_s"] = time.perf_counter() - started
        summary["mean_inference_ms"] = float(np.mean(durations) * 1000) if durations else None
        summary["mean_inference_ms_after_first"] = float(np.mean(durations[1:]) * 1000) if len(durations) > 1 else None
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存：{out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"YOLO 检测失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
