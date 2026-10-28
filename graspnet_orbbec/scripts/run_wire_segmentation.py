#!/usr/bin/env python3
"""离线线材分割入口；模型在本机运行，不发送图像至外部服务。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from common.config import load_config
from segmentation.wire import Sam2WireSegmenter, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--auto", action="store_true", help="本地自动搜索线材和端子，不需要框或点")
    parser.add_argument("--box", nargs=4, type=float)
    parser.add_argument("--positive", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--negative", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.auto and (args.box is not None or args.positive or args.negative):
        parser.error('--auto 不与手工框/点提示同时使用')
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取原图：{args.image}")
    points = args.positive + args.negative
    labels = [1] * len(args.positive) + [0] * len(args.negative)
    if args.auto:
        from segmentation.automatic import AutomaticWireRecognizer
        result = AutomaticWireRecognizer(load_config(args.config)).recognize(image)
    else:
        result = Sam2WireSegmenter(load_config(args.config)).segment(image, box=args.box, points=points, labels=labels)
    save_result(result, args.out)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"已保存：{args.out.resolve()}")


if __name__ == "__main__":
    main()
