#!/usr/bin/env python3
"""正常重启后保持实时关节角的受控使能；默认只读预览，不复位或回零。"""

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import signal
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.wire_trial_runtime import pose_error, require_safe_feedback
from execute_wire_trial import FeedbackMonitor
from read_wire_robot_state import load_binding


def validate_hold_packets(expected, q, width):
    ids = [0x151, 0x155, 0x156, 0x157, 0x159, 0x471, 0x155, 0x156, 0x157]
    joints = np.asarray(q, float)
    if joints.shape != (6,) or not np.isfinite(joints).all() or not .095 <= width <= .100:
        raise ValueError('实时保持目标无效')
    millidegrees = [round(float(j)*180/np.pi*1000) for j in joints]
    joint_payloads = [struct.pack('>ii', *millidegrees[i:i+2]) for i in (0, 2, 4)]
    payloads = [bytes([1, 1, 5, 0, 0, 0, 0, 0]), *joint_payloads,
                struct.pack('>iHBB', round(width*1e6), 1000, 1, 0),
                bytes([7, 2, 0, 0, 0, 0, 0, 0]), *joint_payloads]
    if expected != list(zip(ids, payloads)):
        raise ValueError('SDK编码不是实时角度/开口保持，拒绝执行')


def boot_feedback(state):
    require_safe_feedback(state, require_enabled=False)
    if state['arm_status']['ctrl_mode'] != 0 or state['arm_status'].get('teach_status') not in (0, 2):
        raise ValueError('需要正常待机且拖动未活动')
    if any(d['foc_status']['driver_enable_status'] for d in state['driver_states']):
        raise ValueError('需要本轮重启后六轴均未使能的明确起始状态')


def enabled_hold_observation(initial, state, width):
    checked = copy.deepcopy(state)
    # 使能尚未完成时可短暂报告抱闸未释放；不能据此通过最终验收。
    if checked.get('fresh') and checked['arm_status']['arm_status'] == 6:
        checked['arm_status']['arm_status'] = 0
    require_safe_feedback(checked, require_enabled=False)
    position, angle = pose_error(initial['flange_pose_m_rad'], state['flange_pose_m_rad'])
    joint_delta = float(np.rad2deg(np.max(np.abs(np.asarray(state['joint_angles_rad'])-initial['joint_angles_rad']))))
    if position > .002 or angle > 1 or joint_delta > .5:
        raise ValueError('使能保持期间位姿或关节变化超限')
    ready = (state['arm_status']['arm_status'] == 0 and state['arm_status']['ctrl_mode'] == 1
             and all(d['foc_status']['driver_enable_status'] for d in state['driver_states'])
             and state['gripper_status']['foc_status']['driver_enable_status']
             and abs(state['gripper_status']['value']-width) <= .002)
    return {'ready': bool(ready), 'translation_change_m': position,
            'rotation_change_deg': angle, 'joint_change_deg': joint_delta}


def main():
    import can
    from pyAgxArm.protocols.can_protocol.msgs.effector.agx_gripper.default import ArmMsgGripperCtrl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--post-restart-state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--operator-ready', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('拒绝覆盖已有使能报告')
    if args.apply and not args.operator_ready:
        raise ValueError('实际使能需要现场明确准备完成')
    recorded = json.loads(args.post_restart_state.read_text())
    if (recorded.get('application_frames_sent') != 0 or recorded['arm_status']['arm_status'] != 0
            or recorded['arm_status']['ctrl_mode'] != 0 or any(recorded['joint_enable_status'])):
        raise ValueError('需要重启后正常待机、六轴未使能的只读报告')
    block_file = ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json'
    block = json.loads(block_file.read_text())
    if recorded['observed_wall_time_s'] <= block['wall_time_s']:
        raise ValueError('只读报告早于本轮恢复异常，不能作为重启后检查')
    if args.apply and block.get('post_restart_hold_attempted'):
        raise ValueError('本轮重启后的单次保持使能已尝试，禁止自动重复')
    binding = load_binding(args.binding)
    if (recorded['channel'] != binding.settings['channel']
            or recorded['sdk_source_sha'] != binding.session['robot']['audited_source_sha']):
        raise ValueError('重启检查与当前绑定不一致')
    spec = importlib.util.spec_from_file_location('_enable_hold_decoder', binding.root/'tools/check_flange_feedback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, profile = module.make_decoder(binding.session['robot']['firmware_version'], binding.settings['channel'])
    if profile != 'v189':
        raise ValueError('本轮仅审计v189')
    gripper = robot.init_effector('agx_gripper')
    report = {'status': 'not_started', 'application_frames_sent': [], 'reset_commands_sent': 0,
              'home_targets_sent': 0, 'grasp_targets_sent': 0,
              'command_attempts': 0, 'post_restart_state': str(args.post_restart_state.resolve()),
              'post_restart_state_sha256': hashlib.sha256(args.post_restart_state.read_bytes()).hexdigest()}
    def write_report():
        with args.out.open('w') as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.flush()
            import os
            os.fsync(out.fileno())
    # 在任何发送前保证报告可写；异常后也保留失败证据。
    with args.out.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f'收到停止信号{signum}')
    signal.signal(signal.SIGTERM, stop_signal)
    with Path('/tmp/wire_grasp_can0.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with can.Bus(interface='socketcan', channel=binding.settings['channel'], ignore_config=True,
                         receive_own_messages=False, local_loopback=False) as bus:
                monitor = FeedbackMonitor(bus, robot, gripper, module.FreshPoseDecoder(robot, .020, .020))
                monitor.pump(.5)
                baseline = monitor.snapshot()
                boot_feedback(baseline)
                deadline = time.monotonic()+.5
                while time.monotonic() < deadline:
                    monitor.pump()
                    initial = monitor.snapshot()
                    boot_feedback(initial)
                    dp, da = pose_error(baseline['flange_pose_m_rad'], initial['flange_pose_m_rad'])
                    if dp > .0005 or da > .1:
                        raise ValueError('未停稳，不发送使能指令')
                report['initial_feedback'] = initial
                q = list(initial['joint_angles_rad'])
                width = min(float(initial['gripper_status']['value']), .100)
                if not .095 <= width <= .100 or abs(initial['gripper_status']['value']-width) > .002:
                    raise ValueError('夹爪开度已改变，不能套用本轮保持开口方案')
                robot._msg_mode.move_mode = 1
                robot._msg_mode.move_spd_rate_ctrl = 5
                robot._msg_mode.mit_mode = 0
                joint_messages = robot._deal_move_j_msgs(q)
                gm = ArmMsgGripperCtrl(value=round(width*1e6), force=1000, status_code=1)
                enable = robot._MSG_MotorEnableDisableConfig(joint_index=7, enable_flag=2)
                packets = [robot._parser.pack(robot._msg_mode),
                           *[robot._parser.pack(m) for m in joint_messages], gripper._parser.pack(gm),
                           robot._parser.pack(enable), *[robot._parser.pack(m) for m in joint_messages]]
                expected = [(f.arbitration_id, bytes(f.data)) for f in packets]
                validate_hold_packets(expected, q, width)
                report['packet_preview'] = [{'id': f'0x{i:03X}', 'data_hex': d.hex()} for i, d in expected]
                report['gripper_hold_width_m'] = width
                if not args.apply:
                    report['status'] = 'receive_only_enable_hold_preview_no_commands_sent'
                    return
                block.update(post_restart_hold_attempted=True, post_restart_hold_report=str(args.out.resolve()))
                block_file.write_text(json.dumps(block, ensure_ascii=False, indent=2))
                write_report()
                robot.connect(start_read_thread=False)
                original_send = robot._ctx.comm.send
                def checked_send(frame, *a, **kw):
                    packet = (frame.arbitration_id, bytes(frame.data))
                    stop = packet == (0x150, bytes([1, 0, 0, 0, 0, 0, 0, 0]))
                    index = len(report['application_frames_sent'])
                    if not stop and (index >= len(expected) or packet != expected[index]):
                        raise ValueError('拒绝非本轮实时角度/开口保持的指令')
                    if frame.is_extended_id or frame.is_remote_frame or frame.is_error_frame:
                        raise ValueError('拒绝异常CAN指令格式')
                    report['command_attempts'] += 1
                    original_send(frame, *a, **kw)
                    report['application_frames_sent'].append({'id': f'0x{packet[0]:03X}', 'data_hex': packet[1].hex()})
                    if getattr(robot._ctx.comm, 'last_error', None) is not None:
                        raise RuntimeError('SDK报告CAN发送失败')
                    write_report()
                robot._ctx.comm.send = checked_send
                robot.move_j(q)
                robot.set_auto_set_motion_mode_enabled(False)
                gripper.move_gripper_m(width, force=1.)
                robot.enable()
                robot.move_j(q)
                deadline, settled = time.monotonic()+3, None
                while time.monotonic() < deadline:
                    monitor.pump()
                    current = monitor.snapshot()
                    report['final_feedback'] = current
                    assessment = enabled_hold_observation(initial, current, width)
                    report['assessment'] = assessment
                    if assessment['ready']:
                        settled = settled or time.monotonic()
                        if time.monotonic()-settled >= .5:
                            report['status'] = 'current_hold_enabled_feedback_verified'
                            return
                    else:
                        settled = None
                raise ValueError('未在3秒内观察到正常已使能保持，不自动重试')
        except BaseException as error:
            report.update(status='failed', error=str(error))
            if report['command_attempts']:
                try:
                    robot.electronic_emergency_stop()
                    report['damped_stop_sent'] = True
                except Exception as stop_error:
                    report['stop_error'] = str(stop_error)
            raise
        finally:
            robot.disconnect()
            write_report()
            print(json.dumps({'status': report['status'], 'error': report.get('error'),
                              'application_frames_sent': len(report['application_frames_sent']),
                              'report': str(args.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
