#!/usr/bin/env python3
"""恢复本轮已知保护急停；默认只读，明确--apply后只发送一次SDK reset。"""

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.wire_trial_runtime import pose_error, require_known_software_estop, recovery_observation
from execute_wire_trial import FeedbackMonitor
from read_wire_robot_state import load_binding


def main():
    import can
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--operator-ready', action='store_true')
    p.add_argument('--audit-reviewed', action='store_true',
                   help='仅在本轮初始化判据离线审计完成并另获现场许可后验证一次恢复')
    args = p.parse_args()
    if args.apply and (ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json').exists() and not args.audit_reviewed:
        raise ValueError('本轮恢复异常尚未审计，禁止重复复位')
    if args.out.exists():
        raise FileExistsError('拒绝覆盖已有恢复报告')
    if args.apply and not args.operator_ready:
        raise ValueError('恢复急停需要现场明确准备完成')
    evidence = json.loads(args.evidence.read_text())
    binding = load_binding(args.binding)
    spec = importlib.util.spec_from_file_location('_clear_estop_decoder', binding.root/'tools/check_flange_feedback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, profile = module.make_decoder(binding.session['robot']['firmware_version'], binding.settings['channel'])
    if profile != 'v189':
        raise ValueError('本轮仅审计v189')
    gripper = robot.init_effector('agx_gripper')
    report = {'status': 'not_started', 'application_frames_sent': [],
              'cartesian_or_joint_targets_sent': 0, 'enable_commands_sent': 0,
              'gripper_commands_sent': 0, 'evidence': str(args.evidence.resolve())}
    with Path('/tmp/wire_grasp_can0.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with can.Bus(interface='socketcan', channel=binding.settings['channel'], ignore_config=True,
                         receive_own_messages=False, local_loopback=False) as bus:
                monitor = FeedbackMonitor(bus, robot, gripper, module.FreshPoseDecoder(robot, .020, .020))
                monitor.pump(.5)
                initial = monitor.snapshot()
                require_known_software_estop(initial, evidence)
                report['initial_feedback'] = initial
                deadline = time.monotonic()+.5
                while time.monotonic() < deadline:
                    monitor.pump()
                    state = monitor.snapshot()
                    require_known_software_estop(state, evidence)
                    dp, da = pose_error(initial['flange_pose_m_rad'], state['flange_pose_m_rad'])
                    if dp > .0005 or da > .1:
                        raise ValueError('恢复前未停稳，未发送指令')
                report['packet_preview'] = {'id': '0x150', 'data_hex': '0200000000000000'}
                if not args.apply:
                    report['status'] = 'receive_only_reset_preview_no_commands_sent'
                    return
                robot.connect(start_read_thread=False)
                original_send = robot._ctx.comm.send
                def checked_send(frame, *a, **kw):
                    packet = (frame.arbitration_id, bytes(frame.data))
                    reset = packet == (0x150, bytes([2, 0, 0, 0, 0, 0, 0, 0]))
                    stop = packet == (0x150, bytes([1, 0, 0, 0, 0, 0, 0, 0]))
                    if not stop and (not reset or report['application_frames_sent']):
                        raise ValueError('拒绝重复复位或额外控制报文')
                    original_send(frame, *a, **kw)
                    if getattr(robot._ctx.comm, 'last_error', None) is not None:
                        raise RuntimeError('SDK报告CAN发送失败')
                    report['application_frames_sent'].append({'id': f'0x{packet[0]:03X}', 'data_hex': packet[1].hex()})
                robot._ctx.comm.send = checked_send
                audit_lock = ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json'
                if args.audit_reviewed and audit_lock.exists():
                    audit_record = json.loads(audit_lock.read_text())
                    if audit_record.get('audited_retry_consumed'):
                        raise ValueError('已授权的单次审计后恢复验证已使用，禁止再次执行')
                    audit_record.update(audited_retry_consumed=True, retry_report=str(args.out.resolve()))
                    audit_lock.write_text(json.dumps(audit_record, ensure_ascii=False, indent=2)+'\n')
                reset_sent_wall_time = time.time()
                robot.reset()
                deadline = time.monotonic()+2
                settled_since = None
                report['invalid_initialization_observations'] = []
                while time.monotonic() < deadline:
                    monitor.pump()
                    state = monitor.snapshot()
                    report['final_feedback'] = state
                    assessment = recovery_observation(initial, state, reset_sent_wall_time)
                    if not assessment['ready']:
                        settled_since = None
                        if len(report['invalid_initialization_observations']) < 50:
                            report['invalid_initialization_observations'].append({'assessment': assessment, 'feedback': state})
                        continue
                    report.update(translation_change_m=assessment['translation_change_m'],
                                  rotation_change_deg=assessment['rotation_change_deg'])
                    settled_since = settled_since or time.monotonic()
                    if time.monotonic()-settled_since >= .5:
                        report['status'] = 'software_estop_cleared_feedback_verified'
                        report['motion_ready'] = False
                        return
                raise ValueError('未观察到急停解除，禁止自动复位重试')
        except BaseException as error:
            report.update(status='failed', error=str(error))
            if (report['application_frames_sent']
                    and report['application_frames_sent'][-1]['data_hex'] != '0100000000000000'):
                try:
                    robot.electronic_emergency_stop()
                    report['damped_stop_sent'] = True
                except Exception as stop_error:
                    report['stop_error'] = str(stop_error)
            raise
        finally:
            robot.disconnect()
            with args.out.open('x') as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
            print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
