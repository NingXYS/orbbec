#!/usr/bin/env python3
"""037K线材手眼联调：审计、只读配对采图、自动识别与基座坐标预览。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge, read_json
from arm.wire_geometry import board_plane, fit_pivot, flange_transform
from common.config import load_config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--config', help='现有视觉YAML配置')
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('audit', help='检查冻结标定指纹与系统CAN状态，不访问硬件')
    check = commands.add_parser('check-plane', help='只验证相机原图的棋盘平面，不生成抓取或机器人位姿')
    check.add_argument('--capture', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    capture = commands.add_parser('capture', help='停稳后采RGB/被动法兰反馈，不发送机器人命令')
    capture.add_argument('--out', type=Path)
    for command in ('preview', 'recognize'):
        sub = commands.add_parser(command, help='配对快照自动识别；preview打开交界确认窗口')
        sub.add_argument('--snapshot', type=Path, required=True)
        sub.add_argument('--center-height-mm', type=float,
                         help='实测线材夹持中心相对棋盘印刷面的有符号高度，朝相机为正')
        sub.add_argument('--support-relation-confirmed', action='store_true',
                         help='确认线材局部平面平行于棋盘，且已测量高度关系')
        sub.add_argument('--same-installation-confirmed', action='store_true',
                         help='确认同一标定机械臂/相机安装未变')
        sub.add_argument('--tool', type=Path, help='实测夹持中心与工具坐标轴文件')
        sub.add_argument('--boundary', nargs=3, action='append', default=[], metavar=('ID', 'X', 'Y'),
                         help='本快照人工确认的金属/绝缘交界，可重复')
        sub.add_argument('--roi', nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                         help='本地自动识别工作区；输出仍为原图坐标，裁剪截断端不当作线头')
        sub.add_argument('--boundary-point', nargs=2, type=float, action='append', default=[],
                         help='人工确认线身/金属交界，无需把线身掩码断口当作物理端点')
        sub.add_argument('--out', type=Path)
        sub.add_argument('--seconds', type=float, default=0, help='预览定时退出，0为手动退出')
        sub.add_argument('--canvas', type=Path, help='保存识别窗口画布')
    pivot = commands.add_parser('fit-tcp', help='从多份真实停稳反馈报告拟合同一刚性测点的位置')
    pivot.add_argument('--feedback', nargs='+', type=Path, required=True)
    pivot.add_argument('--out', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    bundle = HandeyeBundle(args.binding)
    if args.command == 'audit':
        interface = Path('/sys/class/net')/bundle.settings['channel']
        info = bundle.info()
        try:
            info['can_up'] = bool(int((interface/'flags').read_text(), 16) & 1)
        except OSError:
            info['can_up'] = None
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'check-plane':
        bundle.check_camera(read_json(args.capture/'camera.json'))
        image = cv2.imread(str(args.capture/'rgb.png'))
        if image is None or image.shape[1::-1] != bundle.size:
            raise ValueError('缺少匹配分辨率的原图')
        plane = board_plane(image, bundle.k, bundle.d, bundle.session['board'])
        report = {'camera': bundle.info(), 'plane': plane, 'robot_pose_available': False,
                  'wire_recognition_performed': False, 'grasp_test_performed': False}
        with args.out.open('x') as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'capture':
        from arm.stationary_capture import capture_snapshot
        out = args.out or ROOT/'output/wire_grasp'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print('请保持机械臂、相机和线材静止，约4秒。仅被动读取CAN，不使能或运动。', flush=True)
        print(f'配对快照已保存：{capture_snapshot(bundle, out)}')
        return 0
    if args.command == 'fit-tcp':
        transforms = []
        for path in args.feedback:
            report = read_json(path)
            if (report.get('status') != 'passed' or report.get('application_frames_sent_by_tool') != 0
                    or report.get('source_sha') != bundle.session['robot']['audited_source_sha']
                    or report.get('firmware_version') != bundle.session['robot']['firmware_version']):
                raise ValueError(f'需要真实只读反馈报告：{path}')
            ts = [flange_transform(s['pose_m_rad']) for s in report['samples']]
            if (len(ts) < 10 or max(np.linalg.norm(t[:3, 3]-ts[0][:3, 3]) for t in ts) > .0005
                    or max(np.linalg.norm(cv2.Rodrigues(t[:3, :3] @ ts[0][:3, :3].T)[0]) for t in ts) > np.deg2rad(.1)):
                raise ValueError(f'反馈不足或接触时未停稳：{path}')
            transforms.append(ts[len(ts)//2])
        result = fit_pivot(transforms)
        result['feedback_sources'] = [str(p.resolve()) for p in args.feedback]
        with args.out.open('x') as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    bridge = WireHandeyeBridge(bundle, args.snapshot,
                              center_height_m=None if args.center_height_mm is None else args.center_height_mm/1000,
                              support_relation_confirmed=args.support_relation_confirmed,
                              installation_confirmed=args.same_installation_confirmed, tool_file=args.tool)
    cfg = load_config(args.config)
    cfg['camera']['serial'] = bundle.camera['serial']
    if args.roi is not None:
        cfg['wire']['auto']['roi_xyxy'] = args.roi
    if args.command == 'recognize':
        from segmentation.automatic import AutomaticWireRecognizer
        from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary
        from segmentation.wire import save_result
        result = AutomaticWireRecognizer(cfg).recognize(bridge.image)
        boundaries = {name: [float(x), float(y)] for name, x, y in args.boundary}
        apply_grasp_policy(result, cfg, boundaries)
        for point in args.boundary_point:
            confirm_body_boundary(result, point, cfg)
        bridge.apply(result, cfg)
        out = args.out or ROOT/'output/wire_grasp'/datetime.now().strftime('recognition_%Y%m%d_%H%M%S_%f')
        print(f'已保存：{save_result(result, out)}')
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.boundary or args.boundary_point or args.out:
        raise ValueError('preview使用Shift+左键确认、S保存；--boundary/--out用于recognize')
    from segmentation.wire_gui import run
    class NoInput:
        stop_flag = False
    gui_args = SimpleNamespace(from_dir=str(bridge.snapshot/'capture'), wire_auto=True,
                               wire_box=None, wire_point=[], wire_seconds=args.seconds,
                               wire_preview=str(args.canvas) if args.canvas else None, no_target=True)
    return run(cfg, gui_args, NoInput(), handeye_bridge=bridge)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'抓取联调未通过：{exc}', file=sys.stderr)
        sys.exit(2)
