#!/usr/bin/env python3
"""现场分步试验执行器；默认仅检查文件，--execute才允许发送白名单指令。"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.wire_trial_runtime import (pose_error, require_expected_start, require_safe_feedback,
                                    validate_plan, validate_tx, joint_start_packets)
from read_wire_robot_state import load_binding


class FeedbackMonitor:
    def __init__(self, bus, robot, gripper, decoder):
        self.bus, self.robot, self.gripper, self.decoder = bus, robot, gripper, decoder
        self.last, self.pose = {}, None

    def pump(self, seconds=.01):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            frame = self.bus.recv(timeout=min(.005, max(0, end-time.monotonic())))
            if frame is None:
                continue
            if frame.is_error_frame:
                raise ValueError('CAN总线报告错误帧')
            if frame.is_extended_id or frame.is_remote_frame or frame.dlc != 8:
                continue
            now = time.time()
            if not -.001 <= now-frame.timestamp <= .020:
                continue
            self.last[frame.arbitration_id] = frame.timestamp
            self.robot._ctx._run_parser_packet_funs(frame)
            pose = self.decoder.feed(frame, time.monotonic(), now)
            if pose is not None:
                self.pose = pose

    def snapshot(self):
        now = time.time()
        def fresh(ids, age):
            return (all(i in self.last for i in ids)
                    and now-min(self.last[i] for i in ids) <= age
                    and max(self.last[i] for i in ids)-min(self.last[i] for i in ids) <= age)
        ready = (self.pose is not None and now-self.pose['host_kernel_timestamp_s'] <= .020
                 and fresh((0x2A5, 0x2A6, 0x2A7), .020)
                 and fresh((0x2A1, 0x2A8), .050)
                 and fresh(tuple(range(0x261, 0x267)), .100))
        if not ready:
            return {'fresh': False, 'observed_wall_time_s': now}
        return {'fresh': True, 'observed_wall_time_s': now,
                'feedback_stamps_s': {f'0x{i:03X}': stamp for i, stamp in self.last.items()},
                'flange_pose_m_rad': list(self.pose['pose_m_rad']),
                'raw_pose_frames': self.pose['frames'],
                'joint_angles_rad': list(self.robot.get_joint_angles().msg),
                'arm_status': self.robot.get_arm_status().msg.to_dict(),
                'gripper_status': self.gripper.get_gripper_status().msg.to_dict(),
                'driver_states': [self.robot.get_driver_states(i).msg.to_dict() for i in range(1, 7)]}


def require_post_restart_audit(plan, block_file):
    source = plan.get('control_audit')
    if not source:
        raise ValueError('本轮控制异常锁仍在，需要新路径绑定重启后保持使能审计')
    audit_file = Path(source['path'])
    if hashlib.sha256(audit_file.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('保持使能审计文件已改变')
    audit = json.loads(audit_file.read_text())
    if (audit.get('status') != 'post_restart_hold_verified_operator_confirmed'
            or audit.get('physical_observation') != 'operator_confirmed_stable_no_obvious_motion_or_collision'
            or plan.get('created_wall_time_s', 0) <= audit['created_wall_time_s']):
        raise ValueError('需要现场确认稳定后的新规划')
    evidence = {}
    for item in audit['evidence']:
        path = Path(item['path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('原异常锁或保持使能证据已改变')
        evidence[path] = json.loads(path.read_text())
    if Path(block_file).resolve() not in evidence:
        raise ValueError('审计不属于当前控制异常事件')
    hold = next((v for v in evidence.values() if v.get('status') == 'current_hold_enabled_feedback_verified'), None)
    ready = [(p, v) for p, v in evidence.items() if v.get('status') == 'feedback_observed']
    if not hold or len(ready) != 1 or len(hold.get('application_frames_sent', [])) != 9:
        raise ValueError('缺少本轮实际保持使能及其后只读复查')
    require_safe_feedback(hold['final_feedback'])
    path, state = ready[0]
    if (state.get('readiness_blockers') or state['observed_wall_time_s'] <= hold['final_feedback']['observed_wall_time_s']
            or not any(Path(s['path']).resolve() == path for s in plan['sources'])):
        raise ValueError('路径未使用使能成功后的真实起始反馈')


def await_pose(monitor, pose, expected_joints, emit, joint_tolerance_deg=None):
    deadline, settled = time.monotonic()+6, None
    while time.monotonic() < deadline:
        monitor.pump()
        state = monitor.snapshot()
        require_safe_feedback(state)
        if np.max(np.abs(np.asarray(state['joint_angles_rad'])-expected_joints)) > np.deg2rad(15):
            raise ValueError('实际关节偏离已检查的连续路径支路')
        position, angle = pose_error(state['flange_pose_m_rad'], pose)
        joint_reached = (joint_tolerance_deg is None or
                         np.max(np.abs(np.asarray(state['joint_angles_rad'])-expected_joints)) <= np.deg2rad(joint_tolerance_deg))
        if position <= .0007 and angle <= .3 and joint_reached:
            settled = settled or time.monotonic()
            if time.monotonic()-settled >= .10:
                emit({'event': 'pose_reached', 'position_error_m': position,
                      'rotation_error_deg': angle, 'feedback': state})
                return state
        else:
            settled = None
    error = ValueError('实际反馈未在6秒内稳定到达当前小步目标')
    error.diagnostic = {'last_feedback_before_stop': state, 'target_pose_m_rad': list(pose),
                        'target_joint_angles_rad': np.asarray(expected_joints).tolist(),
                        'position_error_m': position, 'rotation_error_deg': angle}
    raise error


def await_gripper(monitor, seconds):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        monitor.pump()
        require_safe_feedback(monitor.snapshot())
    return monitor.snapshot()


def execute(plan, binding, phase, emit):
    import can
    spec = importlib.util.spec_from_file_location('_trial_pose_decoder', binding.root/'tools/check_flange_feedback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, _ = module.make_decoder(binding.session['robot']['firmware_version'], binding.settings['channel'])
    gripper = robot.init_effector('agx_gripper')
    sent, command_attempted, start_packets = [], False, None
    try:
        with can.Bus(interface='socketcan', channel=binding.settings['channel'], ignore_config=True,
                     receive_own_messages=False, local_loopback=False) as bus:
            monitor = FeedbackMonitor(bus, robot, gripper, module.FreshPoseDecoder(robot, .020, .020))
            monitor.pump(1)
            initial = monitor.snapshot()
            split = next(i for i, s in enumerate(plan['stages']) if s['name'] == 'descend') if phase == 'grasp' else len(plan['stages'])
            stages = [s for s in plan['stages'] if s['name'] not in ('descend', 'lift')] if phase == 'hover' else plan['stages'][split:]
            hover = next((s for s in plan['stages'] if s['name'] == 'hover'), None)
            expected_pose = plan['initial_flange_pose_m_rad'] if phase == 'hover' else hover['target_flange_pose_m_rad']
            expected_q = plan['initial_joint_angles_rad'] if phase == 'hover' else hover['samples'][-1]['joint_angles_rad']
            require_expected_start(initial, expected_pose, expected_q)
            has_preparation = stages[0]['name'] == 'prepare_start'
            if has_preparation and np.max(np.abs(np.asarray(initial['joint_angles_rad'])-expected_q)) > np.deg2rad(.1):
                raise ValueError('起始调整要求实时关节与已审查起点相差不超过0.1°')
            emit({'event': 'preflight_passed', 'feedback': initial})
            # 标准SDK只建立通信，接收由本线程显式处理，避免混用旧缓存或隐藏线程。
            robot.connect(start_read_thread=False)
            original_send = robot._ctx.comm.send
            def checked_send(frame, *args, **kwargs):
                nonlocal command_attempted
                packet = (frame.arbitration_id, bytes(frame.data))
                stop = packet == (0x150, bytes([1, 0, 0, 0, 0, 0, 0, 0]))
                if start_packets is not None and not stop:
                    if (frame.is_extended_id or len(frame.data) != 8 or not start_packets
                            or packet != start_packets.pop(0)):
                        raise ValueError('起始关节调整指令不属于当前已审查小步')
                else:
                    validate_tx(frame)
                command_attempted = True
                original_send(frame, *args, **kwargs)
                if getattr(robot._ctx.comm, 'last_error', None) is not None:
                    raise RuntimeError('SDK报告CAN发送失败')
                sent.append(frame.arbitration_id)
                emit({'event': 'tx', 'id': f'0x{frame.arbitration_id:03X}', 'data_hex': bytes(frame.data).hex()})
            robot._ctx.comm.send = checked_send
            if has_preparation:
                preparation = stages.pop(0)
                robot._msg_mode.move_spd_rate_ctrl = 5
                robot._msg_mode.move_mode = 1
                robot._msg_mode.mit_mode = 0
                emit({'event': 'stage_start', 'stage': 'prepare_start'})
                for index, sample in enumerate(preparation['samples']):
                    monitor.pump()
                    require_safe_feedback(monitor.snapshot())
                    start_packets = joint_start_packets(sample['joint_angles_rad'], include_mode=index == 0)
                    robot.move_j(sample['joint_angles_rad'])
                    if start_packets:
                        raise ValueError('SDK未完整发送当前起始调整小步')
                    robot.set_auto_set_motion_mode_enabled(False)
                    await_pose(monitor, sample['flange_pose_m_rad'], np.asarray(sample['joint_angles_rad']),
                               emit, joint_tolerance_deg=.2)
                start_packets = None
            else:
                robot.set_speed_percent(5)
            robot.set_motion_mode('l')
            robot.set_auto_set_motion_mode_enabled(False)
            for stage in stages:
                emit({'event': 'stage_start', 'stage': stage['name']})
                print(f"执行阶段：{stage['name']}，{len(stage['samples'])}个已检查小步", flush=True)
                for sample in stage['samples']:
                    monitor.pump()
                    require_safe_feedback(monitor.snapshot())
                    robot.move_l(sample['flange_pose_m_rad'])
                    await_pose(monitor, sample['flange_pose_m_rad'], np.asarray(sample['joint_angles_rad']), emit)
                if stage['name'] == 'hover':
                    gripper.move_gripper_m(.025, force=1.)
                    opened = await_gripper(monitor, 2)
                    if abs(opened['gripper_status']['value']-.025) > .0015:
                        raise ValueError('夹爪未达到25mm试验开度')
                if stage['name'] == 'descend':
                    gripper.move_gripper_m(0., force=1.)
                    closed = await_gripper(monitor, 2)
                    emit({'event': 'gripper_closed_feedback', 'feedback': closed,
                          'physical_hold_success_verified': False})
            final = monitor.snapshot()
            emit({'event': 'phase_completed', 'phase': phase, 'feedback': final,
                  'application_frames_sent': len(sent), 'physical_hold_success_verified': False})
            return final
    except BaseException as error:
        stop_event = None
        if command_attempted:
            try:
                robot.electronic_emergency_stop()
                stop_event = {'event': 'damped_stop_sent'}
            except Exception as stop_error:
                stop_event = {'event': 'stop_send_failed', 'error': str(stop_error)}
        # 先尝试制动再写失败日志，磁盘/日志异常不能绕过停止动作。
        if stop_event is not None:
            emit(stop_event)
        emit({'event': 'aborted', 'error': str(error), 'application_frames_sent': len(sent),
              'diagnostic': getattr(error, 'diagnostic', None)})
        raise
    finally:
        robot.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--phase', choices=('hover', 'grasp'), required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--operator-reviewed', action='store_true', help='现场已审查机械臂、夹爪与相机扫掠净空')
    p.add_argument('--log', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args.phase)
    binding = load_binding(args.binding)
    if plan.get('calibration_sha256') != binding.settings['sha256']:
        raise ValueError('试验路径缺少当前冻结标定指纹')
    if len(plan.get('sources', [])) < 2:
        raise ValueError('试验路径缺少识别/起始反馈的来源指纹')
    for source in plan.get('sources', []):
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('试验源文件已改变，请重新规划')
    if args.execute and (ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json').exists():
        require_post_restart_audit(plan, ROOT/'output/wire_grasp/CONTROL_RECOVERY_BLOCKED.json')
    if not args.execute:
        print(json.dumps({'status': 'files_checked_no_hardware_access', 'phase': args.phase,
                          'motion_commands_sent': 0, 'plan_readiness_blockers': plan['readiness_blockers']}, ensure_ascii=False, indent=2))
        return
    if not args.operator_reviewed or args.log is None:
        raise ValueError('执行需要现场审查标记和新的日志文件')
    with args.log.open('x') as log, Path('/tmp/wire_grasp_can0.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def emit(value):
            log.write(json.dumps({'wall_time_s': time.time(), **value}, ensure_ascii=False)+'\n')
            log.flush()
        def stopped(signum, frame):
            raise KeyboardInterrupt(f'收到终止信号{signum}')
        signal.signal(signal.SIGTERM, stopped)
        execute(plan, binding, args.phase, emit)


if __name__ == '__main__':
    main()
