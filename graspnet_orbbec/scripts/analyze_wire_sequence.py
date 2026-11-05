#!/usr/bin/env python3
"""离线YOLO/SAM2多帧稳定性分析；无相机或机械臂控制入口。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge
from common.config import load_config
from segmentation.grasp_policy import confirm_body_boundary
from segmentation.multiframe import StabilityConfig, analyze_sequence


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', type=Path, required=True, help='任务7序列根目录')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--config', type=Path, help='现有YOLO/SAM2视觉配置')
    p.add_argument('--stability-config', type=Path, default=ROOT/'configs/multiframe_experimental.yaml')
    p.add_argument('--reference-frame', help='manifest中的frame_id；默认第一张')
    p.add_argument('--allow-simulated', action='store_true', help='允许模拟采集数据，仅供离线测试')
    p.add_argument('--geometry-preview', action='store_true', help='仅参考快照的离线几何计算，不规划或执行')
    p.add_argument('--boundary-point', type=float, nargs=2, action='append', default=[], metavar=('X', 'Y'),
                   help='仅这张参考原图的人工交界，不能填其他帧坐标')
    p.add_argument('--center-height-mm', type=float)
    p.add_argument('--support-relation-confirmed', action='store_true')
    p.add_argument('--same-installation-confirmed', action='store_true')
    p.add_argument('--fixed-plane', type=Path)
    p.add_argument('--base-z-offset-mm', type=float, default=0)
    p.add_argument('--tool', type=Path)
    return p


def main():
    args = parser().parse_args()
    cfg = load_config(args.config)
    stability = StabilityConfig(**load_config(args.stability_config))
    if args.boundary_point and not args.geometry_preview:
        raise ValueError('--boundary-point需要--geometry-preview，坐标只属于指定参考帧')

    def geometry(result, reference_snapshot):
        bridge = WireHandeyeBridge(bundle, reference_snapshot,
                                  center_height_m=None if args.center_height_mm is None else args.center_height_mm/1000,
                                  support_relation_confirmed=args.support_relation_confirmed,
                                  installation_confirmed=args.same_installation_confirmed,
                                  tool_file=args.tool, fixed_plane_file=args.fixed_plane,
                                  base_z_offset_m=args.base_z_offset_mm/1000)
        for xy in args.boundary_point:
            confirm_body_boundary(result, xy, cfg)
        bridge.apply(result, cfg)

    bundle = HandeyeBundle(args.binding)
    output = args.out or ROOT/'output/multiframe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report = analyze_sequence(bundle, args.sequence, output, cfg, stability=stability,
                              reference_id=args.reference_frame, allow_simulated=args.allow_simulated,
                              _geometry=geometry if args.geometry_preview else None)
    print(json.dumps({'status': report['status'], 'rejection_reasons': report['rejection_reasons'],
                      'report': str(output/'analysis.json'), 'ab_comparison': report.get('ab_comparison')},
                     ensure_ascii=False, indent=2))
    return 2 if report['status'] == 'manual_confirmation_required' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'离线分析未通过：{exc}', file=sys.stderr)
        sys.exit(2)
