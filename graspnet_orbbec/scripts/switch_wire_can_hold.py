#!/usr/bin/env python3
"""仅对已审计v189执行当前关节保持式CAN切换；默认只接收与预览报文。"""

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.wire_trial_runtime import pose_error, require_safe_feedback
from execute_wire_trial import FeedbackMonitor
from read_wire_robot_state import load_binding


def main():
    import can
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--operator-ready', action='store_true', help='现场人员已离开扫掠区并可操作急停/支撑')
    args = p.parse_args()
    if args.apply and (ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json').exists():
        raise ValueError('本轮控制恢复异常尚未审计，禁止再次实际切换')
    if args.out.exists():
        raise FileExistsError('拒绝覆盖已有切换报告')
    if args.apply and not args.operator_ready:
        raise ValueError('实际切换需要现场准备完成标记')
    lock = None
    if args.apply:
        lock = Path('/tmp/wire_grasp_can0.execution.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = load_binding(args.binding)
    spec = importlib.util.spec_from_file_location('_hold_decoder', binding.root/'tools/check_flange_feedback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, profile = module.make_decoder(binding.session['robot']['firmware_version'], binding.settings['channel'])
    if profile != 'v189':
        raise ValueError('本次仅审计v189，禁止套用其他固件切换流程')
    gripper = robot.init_effector('agx_gripper')
    report = {'status': 'not_started', 'sdk_profile': profile, 'application_frames_sent': [],
              'no_reset_or_enable_or_home_commands': True,
              'official_basis': 'agx_arm_ros@286633e0 lines245-252 and1180-1191; v189 seamless position hold branch'}
    try:
        with can.Bus(interface='socketcan', channel=binding.settings['channel'], ignore_config=True,
                     receive_own_messages=False, local_loopback=False) as bus:
            monitor = FeedbackMonitor(bus, robot, gripper, module.FreshPoseDecoder(robot, .020, .020))
            monitor.pump(.5)
            stable_start = monitor.snapshot()
            require_safe_feedback(stable_start, allowed_ctrl_modes=(0, 1, 2))
            stable_deadline = time.monotonic()+.5
            while time.monotonic() < stable_deadline:
                monitor.pump()
                observed = monitor.snapshot()
                require_safe_feedback(observed, allowed_ctrl_modes=(0, 1, 2))
                dp, da = pose_error(stable_start['flange_pose_m_rad'], observed['flange_pose_m_rad'])
                if dp > .0005 or da > .1:
                    raise ValueError('切换前未停稳，未发送指令')
            initial = monitor.snapshot()
            require_safe_feedback(initial, allowed_ctrl_modes=(0, 1, 2))
            report['initial_feedback'] = initial
            q = list(initial['joint_angles_rad'])
            robot._msg_mode.move_spd_rate_ctrl = 5
            robot._msg_mode.move_mode = 1
            robot._msg_mode.mit_mode = 0
            messages = [robot._msg_mode, *robot._deal_move_j_msgs(q)]
            frames = [robot._parser.pack(message) for message in messages]
            expected = [(f.arbitration_id, bytes(f.data)) for f in frames]
            if ([identifier for identifier, _ in expected] != [0x151, 0x155, 0x156, 0x157]
                    or expected[0][1] != bytes([1, 1, 5, 0, 0, 0, 0, 0])):
                raise ValueError('SDK当前关节保持编码与已审计格式不一致')
            report['current_joint_hold_packet_preview'] = [
                {'id': f'0x{i:03X}', 'data_hex': data.hex()} for i, data in expected]
            if not args.apply:
                report['status'] = 'receive_only_packet_preview_no_commands_sent'
                return
            if initial['arm_status']['ctrl_mode'] == 1:
                report['status'] = 'already_can_control_no_commands_sent'
                return
            robot.connect(start_read_thread=False)
            original_send = robot._ctx.comm.send
            def checked_send(frame, *a, **kw):
                packet = (frame.arbitration_id, bytes(frame.data))
                stop = packet == (0x150, bytes([1, 0, 0, 0, 0, 0, 0, 0]))
                index = len(report['application_frames_sent'])
                if not stop and (index >= len(expected) or packet != expected[index]):
                    raise ValueError('拒绝非本次实时关节保持报文')
                original_send(frame, *a, **kw)
                if getattr(robot._ctx.comm, 'last_error', None) is not None:
                    raise RuntimeError('SDK报告CAN发送失败')
                report['application_frames_sent'].append({'id': f'0x{packet[0]:03X}', 'data_hex': packet[1].hex()})
            robot._ctx.comm.send = checked_send
            # 目标仅为刚读取的真实关节角，不发送零点、抓取点或缓存的历史目标。
            robot.move_j(q)
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                monitor.pump()
                current = monitor.snapshot()
                require_safe_feedback(current, allowed_ctrl_modes=(0, 1, 2))
                position, angle = pose_error(initial['flange_pose_m_rad'], current['flange_pose_m_rad'])
                joint_delta = float(np.max(np.abs(np.asarray(current['joint_angles_rad'])-q)))
                if position > .002 or angle > 1 or joint_delta > np.deg2rad(.5):
                    raise ValueError('切换期间实际位姿变化超限，停止')
                report['final_feedback'] = current
                report['translation_change_m'] = position
                report['rotation_change_deg'] = angle
                if current['arm_status']['ctrl_mode'] == 1 and time.monotonic() > deadline-1:
                    report['status'] = 'can_control_observed_current_joint_hold'
                    return
            raise ValueError('2秒内未观察到正常CAN控制模式，禁止自动复位重试')
    except BaseException as error:
        report.update(status='failed', error=str(error))
        if report['application_frames_sent']:
            try:
                robot.electronic_emergency_stop()
                report['damped_stop_sent'] = True
            except Exception as stop_error:
                report['stop_error'] = str(stop_error)
        raise
    finally:
        robot.disconnect()
        if lock is not None:
            lock.close()
        with args.out.open('x') as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
