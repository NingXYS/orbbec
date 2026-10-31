"""分步试验的反馈门槛和指令白名单；不在导入时访问硬件。"""

import struct
import copy

import cv2
import numpy as np

from arm.wire_geometry import flange_transform, fk


def pose_error(actual, expected):
    a, b = flange_transform(actual), flange_transform(expected)
    return (float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(a[:3, :3] @ b[:3, :3].T)[0]))))


def require_safe_feedback(state, *, allowed_ctrl_modes=(0, 1), require_enabled=True):
    if state.get('fresh') is not True:
        raise ValueError('机器人或夹爪反馈陈旧/不完整')
    arm = state['arm_status']
    if arm.get('teach_status') == 1:
        raise ValueError('拖动/示教记录仍活动，禁止切换或运动')
    if arm['ctrl_mode'] not in allowed_ctrl_modes:
        raise ValueError('请操作者先退出示教/其他控制模式')
    if arm['arm_status'] != 0 or any(arm['err_status'].values()):
        raise ValueError('机械臂报告故障，停止试验')
    for driver in state['driver_states']:
        foc = driver['foc_status']
        if require_enabled and not foc['driver_enable_status']:
            raise ValueError('关节未使能，本程序不会自动使能')
        if any(value for key, value in foc.items() if key != 'driver_enable_status'):
            raise ValueError('关节驱动报告故障')
    gripper = state['gripper_status']
    if gripper['mode'] != 'width' or (require_enabled and not gripper['foc_status']['driver_enable_status']):
        raise ValueError('夹爪不是已使能的宽度模式')
    if any(value for key, value in gripper['foc_status'].items()
           if key not in ('driver_enable_status', 'homing_status')):
        raise ValueError('夹爪报告故障')


def require_known_software_estop(state, evidence):
    """只接受本轮已记录的保护急停，不能借此清除不明故障或仍在拖动的状态。"""
    if (evidence.get('status') != 'failed' or evidence.get('damped_stop_sent') is not True
            or not evidence.get('application_frames_sent')
            or evidence['application_frames_sent'][-1] != {'id': '0x150', 'data_hex': '0100000000000000'}):
        raise ValueError('缺少本轮保护急停的明确发送记录')
    if state.get('fresh') is not True:
        raise ValueError('需要完整新鲜反馈')
    arm = state['arm_status']
    if arm['ctrl_mode'] != 0 or arm['arm_status'] != 1 or arm.get('teach_status') != 2:
        raise ValueError('只允许在拖动已结束的待机/急停状态准备恢复')
    checked = copy.deepcopy(state)
    checked['arm_status']['arm_status'] = 0
    require_safe_feedback(checked, require_enabled=False)


def recovery_observation(initial, state, reset_sent_wall_time):
    """恢复期间先等通信和各反馈重新有效，再判断位姿；不会把未使能当成已可运动。"""
    if not state.get('fresh'):
        return {'ready': False, 'reason': 'waiting_for_complete_feedback'}
    status = state['arm_status']['arm_status']
    if status in (1, 5):
        return {'ready': False, 'reason': 'waiting_for_estop_clear_or_communication_initialization'}
    if status != 0:
        raise ValueError(f'恢复期间出现其他控制器故障：{status}')
    require_safe_feedback(state, require_enabled=False)
    required = (*range(0x2A1, 0x2A9), *range(0x261, 0x267))
    stamps = state.get('feedback_stamps_s', {})
    if any(stamps.get(f'0x{i:03X}', 0) <= reset_sent_wall_time for i in required):
        return {'ready': False, 'reason': 'waiting_for_all_feedback_groups_after_reset'}
    position, angle = pose_error(initial['flange_pose_m_rad'], state['flange_pose_m_rad'])
    if position > .002 or angle > 1:
        raise ValueError('有效反馈确认恢复后位姿变化超限')
    return {'ready': True, 'translation_change_m': position, 'rotation_change_deg': angle}


def require_expected_start(state, pose, joints=None):
    require_safe_feedback(state)
    position, angle = pose_error(state['flange_pose_m_rad'], pose)
    if position > .002 or angle > .5:
        raise ValueError(f'起始位姿已改变：{position*1000:.2f}mm / {angle:.2f}deg，需重新规划')
    if joints is not None:
        actual, expected = np.asarray(state['joint_angles_rad']), np.asarray(joints)
        if actual.shape != (6,) or expected.shape != (6,) or np.max(np.abs(actual-expected)) > np.deg2rad(2):
            raise ValueError('起始关节与已审查路径不一致')


def validate_tx(frame):
    """只允许低速线性位姿、≤1N夹爪命令和减速急停；不允许使能/复位/标零。"""
    if frame.is_extended_id or len(frame.data) != 8:
        raise ValueError('不允许的CAN指令格式')
    identifier, data = frame.arbitration_id, bytes(frame.data)
    if identifier == 0x150:
        allowed = data == bytes([1, 0, 0, 0, 0, 0, 0, 0])
    elif identifier == 0x151:
        allowed = (data[0] == 1 and data[1] in (2, 255) and 1 <= data[2] <= 5
                   and data[3] in (0, 255) and data[4:] == bytes(4))
    elif identifier in (0x152, 0x153, 0x154):
        allowed = True
    elif identifier == 0x159:
        width, force = struct.unpack('>iH', data[:6])
        allowed = 0 <= width <= 25000 and 0 < force <= 1000 and data[6:] == bytes([1, 0])
    else:
        allowed = False
    if not allowed:
        raise ValueError(f'CAN指令被试验白名单拒绝：0x{identifier:03X}')


def validate_plan(plan, phase):
    if plan.get('status') != 'nominal_kinematic_preview_only' or phase not in ('hover', 'grasp'):
        raise ValueError('需要已审查的名义试验路径与明确阶段')
    expected = ['raise', 'translate_above', 'orient_above', 'hover', 'descend', 'lift']
    if plan['stages'][0]['name'] == 'prepare_start':
        expected.insert(0, 'prepare_start')
        validate_start_preparation(plan)
    if [s['name'] for s in plan['stages']] != expected or plan['speed_percent'] != 5:
        raise ValueError('试验阶段或速度配置改变')
    if plan['gripper_open_width_m'] != .025 or plan['gripper_force_n'] != 1. or plan['gripper_close_target_m'] != 0:
        raise ValueError('夹爪试验参数改变')
    previous = plan['initial_flange_pose_m_rad']
    total = 0
    for stage in plan['stages']:
        if not stage['samples']:
            raise ValueError('阶段缺少经过检查的路径点')
        for sample in stage['samples']:
            pose = np.asarray(sample['flange_pose_m_rad'], float)
            joints = np.asarray(sample['joint_angles_rad'], float)
            if pose.shape != (6,) or joints.shape != (6,) or not np.isfinite(pose).all() or not np.isfinite(joints).all():
                raise ValueError('路径包含无效位姿')
            position, angle = pose_error(previous, pose)
            if (position > .00801 or angle > 3.01 or sample['position_error_m'] > .0005
                    or sample['rotation_error_deg'] > .1 or sample['max_joint_step_deg'] > 10):
                raise ValueError('路径点超出离线检查门槛')
            if np.linalg.norm(pose[:3]) > .8 or not .05 <= pose[2] <= .65:
                raise ValueError('法兰位姿超出本轮试验范围')
            previous, total = pose.tolist(), total+1
        end_position, end_angle = pose_error(previous, stage['target_flange_pose_m_rad'])
        if end_position > 1e-9 or end_angle > 1e-5:
            raise ValueError('阶段末点与目标不一致')
    if total > 200:
        raise ValueError('路径点过多，拒绝扩大本轮试验范围')


def joint_start_packets(joints, *, include_mode=False):
    q = np.asarray(joints, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('起始调整需要有限六轴角度')
    mdeg = [round(float(j)*180/np.pi*1000) for j in q]
    packets = [(0x155+i, struct.pack('>ii', *mdeg[2*i:2*i+2])) for i in range(3)]
    return ([(0x151, bytes([1, 1, 5, 0, 0, 0, 0, 0]))] if include_mode else [])+packets


def validate_start_preparation(plan):
    """该例外只允许离开现有边界外位置；不能借此扩展普通路径的关节限位。"""
    q = np.asarray(plan['initial_joint_angles_rad'], float)
    bounds = np.asarray(plan['joint_limits_rad'], float)
    if q.shape != (6,) or bounds.shape != (6, 2) or not np.isfinite(q).all() or not np.isfinite(bounds).all():
        raise ValueError('起始调整缺少有效真实起始角度/名义限位')
    outside = (q < bounds[:, 0]) | (q > bounds[:, 1])
    if (not outside.any() or any(i not in (1, 2) for i in np.flatnonzero(outside))
            or np.max(np.maximum(bounds[:, 0]-q, q-bounds[:, 1])) > np.deg2rad(5)):
        raise ValueError('起始调整仅允许J2/J3轻微越出名义规划边界')
    goal = q.copy()
    for i in np.flatnonzero(outside):
        goal[i] = bounds[i, 0]+np.deg2rad(.5) if q[i] < bounds[i, 0] else bounds[i, 1]-np.deg2rad(.5)
    count = int(np.ceil(np.max(np.abs(goal-q))/np.deg2rad(1)))
    stage = plan['stages'][0]
    if count > 5 or stage.get('motion_mode') != 'j' or len(stage['samples']) != count:
        raise ValueError('起始调整步数/模式改变')
    for fraction, sample in zip(np.linspace(0, 1, count+1)[1:], stage['samples']):
        expected = q+(goal-q)*fraction
        if not np.allclose(sample['joint_angles_rad'], expected, atol=1e-10, rtol=0):
            raise ValueError('起始调整不是已审查的单调小幅关节路径')
        modeled = fk(plan['nominal_mdh'], expected)
        if not np.allclose(flange_transform(sample['flange_pose_m_rad']), modeled, atol=1e-7):
            raise ValueError('起始调整名义FK与关节目标不一致')
