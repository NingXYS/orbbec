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
from segmentation.grasp_policy import apply_grasp_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--auto", action="store_true", help="本地自动搜索线材和端子，不需要框或点")
    parser.add_argument("--box", nargs=4, type=float)
    parser.add_argument("--positive", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--negative", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary", nargs=3, action="append", default=[], metavar=('ID','X','Y'),
                        help="人工确认候选的金属/绝缘交界；原图像素，可重复")
    parser.add_argument('--offline-scene',type=Path,help='冻结坐标上下文；启用14mm/TCP/完整夹爪投影复核')
    parser.add_argument('--terminal-annotation',type=Path,help='独立保存并绑定原图/实例的人工端子多边形')
    args = parser.parse_args()
    if args.auto and (args.box is not None or args.positive or args.negative):
        parser.error('--auto 不与手工框/点提示同时使用')
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"无法读取原图：{args.image}")
    points = args.positive + args.negative
    labels = [1] * len(args.positive) + [0] * len(args.negative)
    cfg = load_config(args.config)
    if args.auto:
        from segmentation.automatic import AutomaticWireRecognizer
        result = AutomaticWireRecognizer(cfg).recognize(image)
    else:
        result = Sam2WireSegmenter(cfg).segment(image, box=args.box, points=points, labels=labels)
    boundaries = {name: [float(x), float(y)] for name,x,y in args.boundary}
    apply_grasp_policy(result, cfg, boundaries)
    from segmentation.grasp_v2 import FrozenScene,apply_preview
    scene=FrozenScene.load(args.offline_scene) if args.offline_scene else None
    annotation=None
    if args.terminal_annotation:
        from segmentation.terminal_annotation import load_annotation
        plans=[p for p in result.grasp_plans if p.get('boundary_confirmed')]
        if scene is None or len(plans)!=1:raise ValueError('端子标注需要冻结scene和唯一确认交界')
        annotation=load_annotation(args.terminal_annotation,scene.image_path,result.image_bgr,result.mask,
            candidate_id=plans[0]['candidate_id'],boundary_xy=plans[0]['boundary_input_xy'])
    apply_preview(result,cfg,scene,annotation)
    save_result(result, args.out)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"已保存：{args.out.resolve()}")


if __name__ == "__main__":
    main()
