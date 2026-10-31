#!/usr/bin/env python3
"""被动核对PIPER X位姿、关节、驱动和原厂夹爪；不创建SDK通信会话。"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.wire_geometry import flange_transform


def load_binding(path):
    settings = json.loads(Path(path).read_text())
    root = (ROOT/settings['handeye_root']).resolve(strict=True)
    report = root/settings['calibration_report']
    for name, filename in [('session', 'config_used.json'), ('camera', 'camera_used.json'), ('result', 'result.json')]:
        if hashlib.sha256((report/filename).read_bytes()).hexdigest() != settings['sha256'][name]:
            raise ValueError('冻结标定文件改变，拒绝沿用机器人合同')
    session = json.loads((report/'config_used.json').read_text())
    return SimpleNamespace(settings=settings, root=root, session=session)


def receive_state(bundle, seconds=3):
    import can
    from pyAgxArm.utiles.mdh_kinematics import get_mdh
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    interface = Path('/sys/class/net')/bundle.settings['channel']
    if int((interface/'type').read_text()) != 280 or not int((interface/'flags').read_text(), 16)&1:
        raise ValueError('需要已启用CAN接口，本工具不配置系统')
    spec = importlib.util.spec_from_file_location('_wire_feedback_decoder',
                                                 bundle.root/'tools/check_flange_feedback.py')
    decoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder_module)
    robot, profile = decoder_module.make_decoder(bundle.session['robot']['firmware_version'],
                                                  bundle.settings['channel'])
    with patch('socket.socket', side_effect=RuntimeError('禁止SDK创建socket')), \
         patch('can.Bus', side_effect=RuntimeError('禁止SDK连接CAN')), \
         patch('threading.Thread.start', side_effect=RuntimeError('禁止SDK启动线程')):
        gripper = robot.init_effector('agx_gripper')
    decoder = decoder_module.FreshPoseDecoder(robot, .020, .020)
    counts, last, pose = Counter(), {}, None
    pose_count = 0
    with can.Bus(interface='socketcan', channel=bundle.settings['channel'], ignore_config=True,
                 receive_own_messages=False, local_loopback=False) as bus:
        start = time.monotonic()
        while time.monotonic()-start < seconds:
            frame = bus.recv(timeout=.02)
            if frame is None:
                continue
            now = time.time()
            if (frame.is_extended_id or frame.is_remote_frame or frame.is_error_frame
                    or frame.dlc != 8 or not -.001 <= now-frame.timestamp <= .020):
                continue
            counts[f'0x{frame.arbitration_id:03X}'] += 1
            last[frame.arbitration_id] = frame.timestamp
            robot._parser.parse_packet(frame)
            gripper._parser.parse_packet(frame)
            complete = decoder.feed(frame, time.monotonic(), now)
            if complete is not None:
                pose, pose_count = complete, pose_count+1
        now = time.time()
        fresh = lambda identifiers, age: (all(i in last for i in identifiers)
                    and now-min(last[i] for i in identifiers) <= age
                    and max(last[i] for i in identifiers)-min(last[i] for i in identifiers) <= age)
        joints = robot.get_joint_angles() if fresh((0x2A5, 0x2A6, 0x2A7), .020) else None
        status = robot.get_arm_status() if fresh((0x2A1,), .050) else None
        gs = gripper.get_gripper_status() if fresh((0x2A8,), .050) else None
        drivers = []
        for i in range(1, 7):
            value = robot.get_driver_states(i) if fresh((0x260+i,), .100) else None
            drivers.append(value.msg.to_dict() if value is not None else None)
    report = {'status': 'feedback_observed', 'channel': bundle.settings['channel'],
              'observed_wall_time_s': now, 'firmware_profile': profile,
              'sdk_source_sha': bundle.session['robot']['audited_source_sha'],
              'application_frames_sent': 0, 'sdk_session_connected': False,
              'complete_pose_count': pose_count, 'counts_by_id': dict(counts),
              'flange_pose_m_rad': pose['pose_m_rad'] if pose and now-pose['host_kernel_timestamp_s'] <= .020 else None,
              'joint_angles_rad': list(joints.msg) if joints is not None else None,
              'arm_status': status.msg.to_dict() if status is not None else None,
              'gripper_status': gs.msg.to_dict() if gs is not None else None,
              'driver_states': drivers,
              'nominal_mdh': get_mdh('piper_x'),
              'joint_limits_rad': list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values()),
              'readiness_blockers': []}
    for key in ('flange_pose_m_rad', 'joint_angles_rad', 'arm_status', 'gripper_status'):
        if report[key] is None:
            report['readiness_blockers'].append(f'missing_fresh_{key}')
    if report['arm_status'] is not None:
        arm = report['arm_status']
        if arm['ctrl_mode'] not in (0, 1):
            report['readiness_blockers'].append('operator_must_exit_teaching_or_other_control_mode')
        if arm['arm_status'] != 0 or any(arm['err_status'].values()):
            report['readiness_blockers'].append('arm_fault_reported')
    if report['gripper_status'] is not None:
        state = report['gripper_status']
        if state['mode'] != 'width':
            report['readiness_blockers'].append('gripper_not_in_verified_width_mode')
        if any(v for k, v in state['foc_status'].items() if k not in ('driver_enable_status', 'homing_status')):
            report['readiness_blockers'].append('gripper_fault_reported')
    if all(d is not None for d in drivers):
        report['joint_enable_status'] = [d['foc_status']['driver_enable_status'] for d in drivers]
        if not all(report['joint_enable_status']):
            report['readiness_blockers'].append('joint_motors_not_all_enabled')
        if any(v for d in drivers for k, v in d['foc_status'].items() if k != 'driver_enable_status'):
            report['readiness_blockers'].append('joint_driver_fault_reported')
    else:
        report['joint_enable_status'] = None
        report['readiness_blockers'].append('missing_fresh_driver_feedback')
    if report['flange_pose_m_rad'] is not None and joints is not None:
        computed = flange_transform(robot.fk(list(joints.msg)))
        feedback = flange_transform(report['flange_pose_m_rad'])
        report['nominal_fk_feedback_translation_difference_m'] = float(np.linalg.norm(computed[:3, 3]-feedback[:3, 3]))
        report['nominal_fk_feedback_rotation_difference_deg'] = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(computed[:3, :3] @ feedback[:3, :3].T)[0])))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('拒绝覆盖已有机器人状态报告')
    report = receive_state(load_binding(args.binding))
    with args.out.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
