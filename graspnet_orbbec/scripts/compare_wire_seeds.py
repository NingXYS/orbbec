#!/usr/bin/env python3
"""对冻结RGB进行种子A/B比较；不加载SAM2/YOLO、不采图、不连接机械臂。"""

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config, load_config
from segmentation.automatic import automatic_line_seeds, draw_seed_debug


def compare(image, cfg, roi, out):
    """输出两组完整RGB坐标、过滤统计及原图尺寸的可视化；禁止覆盖已有目录。"""
    h, w = image.shape[:2]
    roi = [0, 0, w, h] if roi is None else roi
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('ROI必须在完整RGB图像范围内')
    crop = image[y1:y2, x1:x2]
    reports = {}
    canvases = {}
    for name, enabled in [('before', False), ('after', True)]:
        settings = Config(copy.deepcopy(cfg))
        auto = settings.setdefault('wire', {}).setdefault('auto', {})
        auto.setdefault('seed_filter', {})['enabled'] = enabled
        report = {}
        automatic_line_seeds(crop, settings, roi_xyxy=roi, diagnostics=report)
        reports[name] = report
        canvas = image.copy()
        canvas[y1:y2, x1:x2] = draw_seed_debug(crop, report)
        canvases[name] = canvas
    summary = {
        'image_size_wh': [w, h], 'roi_xyxy_original_rgb': roi,
        'before_count': reports['before']['seed_count'],
        'after_count': reports['after']['seed_count'],
        'after_allowed_pixels': reports['after']['allowed_pixels'],
        'after_filtered_contrast_pixels': reports['after']['contrast_pixels_filtered'],
        'sam_inference_performed': False, 'robot_commands_sent': 0,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name in reports:
        (out/f'{name}.json').write_text(json.dumps(reports[name], ensure_ascii=False, indent=2))
        if not cv2.imwrite(str(out/f'{name}.png'), canvases[name]):
            raise OSError('种子可视化保存失败')
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('无法读取RGB图像')
    try:
        summary = compare(image, load_config(args.config), args.roi, args.out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
