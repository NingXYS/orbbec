"""基于真实起始反馈的离线试验路径检查；不连接或控制机械臂。"""

import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from arm.wire_geometry import flange_transform, rigid_transform, fk


def matrix_pose(transform):
    t = rigid_transform(transform)
    r = t[:3, :3]
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) > 1e-6:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = math.atan2(-r[1, 2], r[1, 1]), 0.0
    pose = [*t[:3, 3].tolist(), roll, pitch, yaw]
    if not np.allclose(flange_transform(pose), t, atol=1e-7):
        raise ValueError('矩阵到SDK固定轴XYZ转换失败')
    return pose


def ik(mdh, limits, target, seed):
    target, bounds = rigid_transform(target), np.asarray(limits, float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('需要已核实的六轴关节限位')
    seed = np.clip(np.asarray(seed, float), bounds[:, 0]+1e-7, bounds[:, 1]-1e-7)
    def residual(q):
        current = fk(mdh, q)
        angle = cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].ravel()
        return np.r_[(current[:3, 3]-target[:3, 3])*1000, angle*100]
    solved = least_squares(residual, seed, bounds=(bounds[:, 0], bounds[:, 1]),
                           max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    current = fk(mdh, solved.x)
    position_error = float(np.linalg.norm(current[:3, 3]-target[:3, 3]))
    rotation_error = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0])))
    if position_error > .0005 or rotation_error > .1:
        raise ValueError(f'离线IK未通过：位置{position_error*1000:.2f}mm，姿态{rotation_error:.2f}deg')
    return solved.x, {'position_error_m': position_error, 'rotation_error_deg': rotation_error}


def sample_segment(start, end, distance_step_m=.008, angle_step_deg=3):
    start, end = rigid_transform(start), rigid_transform(end)
    rotvec = cv2.Rodrigues(end[:3, :3] @ start[:3, :3].T)[0].ravel()
    steps = max(1, math.ceil(np.linalg.norm(end[:3, 3]-start[:3, 3])/distance_step_m),
                math.ceil(np.rad2deg(np.linalg.norm(rotvec))/angle_step_deg))
    for fraction in np.linspace(0, 1, steps+1)[1:]:
        t = np.eye(4)
        t[:3, :3] = cv2.Rodrigues(rotvec*fraction)[0] @ start[:3, :3]
        t[:3, 3] = start[:3, 3]+fraction*(end[:3, 3]-start[:3, 3])
        yield t


def prepare_start_segment(state):
    """只准备J2/J3在名义边界外≤5°时朝区间内部的小幅关节路径，不扩展限位。"""
    q = np.asarray(state['joint_angles_rad'], float)
    limits = np.asarray(state['joint_limits_rad'], float)
    outside = (q < limits[:, 0]) | (q > limits[:, 1])
    if not outside.any() or any(i not in (1, 2) for i in np.flatnonzero(outside)):
        raise ValueError('本轮起始调整仅适用于J2/J3轻微越出名义规划边界')
    if np.max(np.maximum(limits[:, 0]-q, q-limits[:, 1])) > np.deg2rad(5):
        raise ValueError('起始角度超出本轮可审查的小幅调整范围')
    goal = q.copy()
    for i in np.flatnonzero(outside):
        goal[i] = limits[i, 0]+np.deg2rad(.5) if q[i] < limits[i, 0] else limits[i, 1]-np.deg2rad(.5)
    if np.max(np.abs(goal-q)) > np.deg2rad(5):
        raise ValueError('起始调整总幅度超过5°')
    steps = math.ceil(float(np.max(np.abs(goal-q)))/np.deg2rad(1))
    previous, samples = q, []
    for fraction in np.linspace(0, 1, steps+1)[1:]:
        point = q+(goal-q)*fraction
        pose = matrix_pose(fk(state['nominal_mdh'], point))
        samples.append({'joint_angles_rad': point.tolist(), 'flange_pose_m_rad': pose,
                        'max_joint_step_deg': float(np.rad2deg(np.max(np.abs(point-previous)))),
                        'position_error_m': 0., 'rotation_error_deg': 0.,
                        'kinematic_source': 'nominal_fk_of_joint_target_not_measured_feedback'})
        previous = point
    return {'name': 'prepare_start', 'motion_mode': 'j',
            'start_flange_pose_m_rad': state['flange_pose_m_rad'],
            'target_flange_pose_m_rad': samples[-1]['flange_pose_m_rad'], 'samples': samples}


def build_trial_plan(recognition, state, candidate_id, *, contact_setback_mm=5, prepare_start=False):
    """尖端后5mm为试验参考选择，保留名义模型标识，不能伪装成实测TCP。"""
    if not 0 <= contact_setback_mm <= 10:
        raise ValueError('试验夹持参考仅允许位于名义尖端后0至10mm')
    candidate = next((p for p in recognition['grasp_plans'] if p['candidate_id'] == candidate_id), None)
    if candidate is None or candidate.get('status') != 'base_frame_preview_ready' or candidate.get('surface_projection_only'):
        raise ValueError('需要已确认交界和线材中心高度的完整基座坐标预览')
    if set(candidate['readiness_blockers']) - {'nominal_finger_tip_reference_not_verified_grasp_center'}:
        raise ValueError('存在不能用于生成名义试验路径的定位阻塞')
    if state.get('application_frames_sent') != 0 or state.get('flange_pose_m_rad') is None or state.get('joint_angles_rad') is None:
        raise ValueError('需要真实被动起始反馈')
    initial = flange_transform(state['flange_pose_m_rad'])
    q0 = np.asarray(state['joint_angles_rad'])
    nominal_initial = fk(state['nominal_mdh'], q0)
    if (np.linalg.norm(initial[:3, 3]-nominal_initial[:3, 3]) > .002
            or np.linalg.norm(cv2.Rodrigues(initial[:3, :3] @ nominal_initial[:3, :3].T)[0]) > np.deg2rad(.5)):
        raise ValueError('SDK名义运动学与起始反馈不一致，禁止规划')
    preparation = prepare_start_segment(state) if prepare_start else None
    start = flange_transform(preparation['target_flange_pose_m_rad']) if preparation else initial
    start_seed = np.asarray(preparation['samples'][-1]['joint_angles_rad']) if preparation else q0
    base_grasp = rigid_transform(candidate['T_base_grasp'])
    tool = np.eye(4)
    tool[2, 3] = .1425-contact_setback_mm/1000
    errors = []
    for symmetry in (1, -1):
        grasp = base_grasp.copy()
        grasp[:3, :2] *= symmetry
        final = grasp @ np.linalg.inv(tool)
        normal = -grasp[:3, 2]
        hover = final.copy()
        hover[:3, 3] += normal*.030
        transit_height = max(start[:3, 3] @ normal, hover[:3, 3] @ normal)+.080
        raised = start.copy()
        raised[:3, 3] += normal*(transit_height-start[:3, 3] @ normal)
        translated = raised.copy()
        translated[:3, 3] = hover[:3, 3]+normal*(transit_height-hover[:3, 3] @ normal)
        oriented = translated.copy()
        oriented[:3, :3] = final[:3, :3]
        lifted = final.copy()
        lifted[:3, 3] += normal*.030
        goals = [('raise', raised), ('translate_above', translated), ('orient_above', oriented),
                 ('hover', hover), ('descend', final), ('lift', lifted)]
        current, seed, stages = start.copy(), start_seed.copy(), [preparation] if preparation else []
        try:
            for name, goal in goals:
                samples = []
                for t in sample_segment(current, goal):
                    solution, error = ik(state['nominal_mdh'], state['joint_limits_rad'], t, seed)
                    jump = float(np.max(np.abs(solution-seed)))
                    if jump > np.deg2rad(10):
                        raise ValueError(f'{name}关节支路跳变{np.rad2deg(jump):.1f}deg')
                    samples.append({'flange_pose_m_rad': matrix_pose(t), 'joint_angles_rad': solution.tolist(),
                                    'max_joint_step_deg': float(np.rad2deg(jump)), **error})
                    seed = solution
                stages.append({'name': name, 'start_flange_pose_m_rad': matrix_pose(current),
                               'target_flange_pose_m_rad': matrix_pose(goal), 'samples': samples})
                current = goal
            return {'schema_version': 1, 'status': 'nominal_kinematic_preview_only', 'candidate_id': candidate_id,
                    'initial_flange_pose_m_rad': matrix_pose(initial), 'initial_joint_angles_rad': q0.tolist(),
                    'nominal_mdh': state['nominal_mdh'], 'joint_limits_rad': state['joint_limits_rad'],
                    'T_flange_trial_contact': tool.tolist(), 'contact_setback_mm': contact_setback_mm,
                    'tool_reference_verified': False, 'closing_axis_symmetry': symmetry,
                    'grasp_base_m': candidate['grasp_base_m'],
                    'center_height_over_board_m': candidate['center_height_over_board_m'],
                    'gripper_open_width_m': .025, 'gripper_close_target_m': 0., 'gripper_force_n': 1.,
                    'speed_percent': 5, 'stages': stages,
                    'collision_checked': False, 'camera_mount_envelope_verified': False,
                    'motion_commands_enabled': False,
                    'readiness_blockers': ['nominal_tool_contact_reference_requires_operator_review',
                                           'swept_robot_and_camera_collision_review_required',
                                           *(['operator_must_exit_teaching_or_other_control_mode']
                                             if state.get('arm_status', {}).get('ctrl_mode') not in (0, 1) else []),
                                           *state.get('readiness_blockers', [])]}
        except ValueError as exc:
            errors.append(f'symmetry={symmetry}: {exc}')
    raise ValueError('两种夹爪对称朝向均未通过连续IK：'+'; '.join(errors))
