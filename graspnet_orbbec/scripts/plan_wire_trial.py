#!/usr/bin/env python3
"""冻结真实识别与起始反馈，生成连续逆解检查后的名义试验路径。"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge, image_digest, read_json, sha256
from arm.wire_trial_plan import build_trial_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--recognition', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--contact-setback-mm', type=float, default=5)
    p.add_argument('--prepare-start', action='store_true', help='仅准备J2/J3轻微越出名义范围时的小幅进入区间路径')
    p.add_argument('--control-audit', type=Path, help='本轮人工重启、保持使能和现场稳定观察的证据')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('拒绝覆盖已有试验路径')
    bundle = HandeyeBundle(args.binding)
    recognition, state = read_json(args.recognition), read_json(args.state)
    candidate = next(c for c in recognition['grasp_plans'] if c['candidate_id'] == args.candidate)
    snapshot = Path(recognition['grasp_policy']['capture_snapshot'])
    bridge = WireHandeyeBridge(bundle, snapshot,
                              center_height_m=candidate['center_height_over_board_m'],
                              support_relation_confirmed=True, installation_confirmed=True)
    image = cv2.imread(str(args.recognition.parent/'color.png'))
    if image is None or image_digest(image) != image_digest(bridge.image):
        raise ValueError('识别保存原图与配对快照不一致')
    if (state['sdk_source_sha'] != bundle.session['robot']['audited_source_sha']
            or state['channel'] != bundle.settings['channel']):
        raise ValueError('起始反馈不属于当前SDK/通道')
    plan = build_trial_plan(recognition, state, args.candidate, contact_setback_mm=args.contact_setback_mm,
                            prepare_start=args.prepare_start)
    sources = [args.recognition, args.state, snapshot/'pair.json', snapshot/'capture/rgb.png',
               ROOT/'configs/wire_tool_piper_x_nominal.json']
    if args.control_audit:
        sources.append(args.control_audit)
        plan['control_audit'] = {'path': str(args.control_audit.resolve()), 'sha256': sha256(args.control_audit)}
    plan.update(calibration_sha256=bundle.settings['sha256'], created_wall_time_s=time.time(),
                source_recognition=str(args.recognition.resolve()), snapshot=str(snapshot.resolve()),
                sources=[{'path': str(path.resolve()), 'sha256': sha256(path)} for path in sources])
    with args.out.open('x') as out:
        json.dump(plan, out, ensure_ascii=False, indent=2)
    print(json.dumps({'path': str(args.out.resolve()), 'status': plan['status'],
                      'grasp_base_m': plan['grasp_base_m'], 'contact_setback_mm': plan['contact_setback_mm'],
                      'sample_count': sum(len(s['samples']) for s in plan['stages']),
                      'stages': [{'name': s['name'], 'flange_pose_m_rad': s['target_flange_pose_m_rad']} for s in plan['stages']],
                      'collision_checked': False, 'readiness_blockers': plan['readiness_blockers']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
