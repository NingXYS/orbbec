"""固定装配关节目标的数据校验；不访问机械臂或 ROS。"""

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from arm.moveit_wire_adapter import rigid_transform, transform_to_pose


JOINT_NAMES = [f'joint{i}' for i in range(1, 7)]


def load_assembly_target(path, *, root, frame_id, joints):
    """显式选择配置才追加装配阶段；历史验证状态不代替本次运行检查。"""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    data = json.loads(raw)
    try:
        if data['schema_version'] != 1 or data['robot_model'] != 'piper_x':
            raise ValueError('装配目标需要 piper_x schema_version=1')
        if frame_id != 'base_link' or data['reference_frame'] != frame_id:
            raise ValueError('装配目标需要 base_link 坐标系')
        if list(joints) != JOINT_NAMES or data['joint_names'] != JOINT_NAMES:
            raise ValueError('装配目标六轴名称或顺序不一致')
        values = np.asarray(data['joint_positions_rad'], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('装配目标需要有限六轴弧度')
        flange = rigid_transform(data['flange_goal']['T_base_frame'], '装配法兰')
        tcp = rigid_transform(data['grasp_tcp_goal']['T_base_frame'], '装配夹持中心')
        tool = rigid_transform(data['tool']['T_flange_grasp'], '装配工具偏置')
        tool_path = (Path(root) / data['tool']['file']).resolve(strict=True)
        tool_raw = tool_path.read_bytes()
        if hashlib.sha256(tool_raw).hexdigest() != data['tool']['sha256']:
            raise ValueError('装配工具配置已变化')
        actual_tool = rigid_transform(json.loads(tool_raw)['T_flange_grasp'])
        if not np.allclose(tool, actual_tool, atol=1e-9, rtol=0):
            raise ValueError('装配记录的工具偏置与配置不一致')
        if not np.allclose(flange @ tool, tcp, atol=1e-9, rtol=0):
            raise ValueError('装配目标工具补偿不一致')
        approach = data.get('final_approach')
        if approach is not None:
            if (approach['frame_id'] != 'base_link' or approach['axis'] != 'y'
                    or approach['direction'] not in (-1, 1)
                    or not 0.005 <= float(approach['distance_m']) <= 0.1
                    or not 0 < float(approach.get('max_speed_m_s', 0.01)) <= 0.01):
                raise ValueError('装配末段需要 base_link Y轴、明确方向和5–100mm长度')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'装配目标缺少有效字段：{exc}') from exc
    return {
        'name': 'assembly', 'kind': 'joint', 'target_id': data.get('target_id'),
        'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'joint_names': list(joints), 'joint_positions_rad': values.tolist(),
        'pose': transform_to_pose(flange), 'T_base_flange': flange.tolist(),
        'tool_file': str(tool_path), 'tool_sha256': data['tool']['sha256'],
        'joint_goal_tolerance_rad': 0.001,
        'final_approach': approach,
        'warnings': ['operator_requested_transfer_path_test',
                     'held_wire_and_physical_obstacles_require_on_site_review',
                     'gripper_closure_is_not_independent_grasp_success_detection'],
    }


def validate_runtime_assembly(stage, urdf, actual_flange):
    """对照运行模型限位和正解，拒绝旧模型或错误关节/法兰配对。"""
    model = ET.fromstring(urdf)
    limits = {joint.attrib['name']: joint.find('limit') for joint in model.findall('joint')}
    for name, value in zip(stage['joint_names'], stage['joint_positions_rad']):
        limit = limits.get(name)
        if limit is None or not float(limit.attrib['lower']) <= value <= float(limit.attrib['upper']):
            raise ValueError(f'装配目标 {name} 超出运行模型限位或缺少限位')
    actual = rigid_transform(actual_flange)
    expected = rigid_transform(stage['T_base_flange'])
    position = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
    angle = math.acos(float(np.clip((np.trace(actual[:3, :3] @ expected[:3, :3].T)-1)/2, -1, 1)))
    if position > 0.001 or angle > math.radians(0.1):
        raise ValueError('装配目标关节正解与保存法兰不一致')
    return {'runtime_model_sha256': hashlib.sha256(urdf.encode()).hexdigest(),
            'position_error_m': position, 'orientation_error_rad': angle}


def validate_y_approach(transforms, direction, distance):
    """对返回轨迹逐点正解复核，拒绝把普通关节路径伪称为Y轴直线。"""
    matrices = np.array([rigid_transform(t) for t in transforms])
    if len(matrices) < 2:
        raise ValueError('装配直线轨迹点不足')
    positions = matrices[:, :3, 3]
    lateral = float(np.max(np.linalg.norm((positions-positions[-1])[:, [0, 2]], axis=1)))
    angles = [math.acos(float(np.clip((np.trace(t[:3,:3] @ matrices[-1,:3,:3].T)-1)/2, -1, 1)))
              for t in matrices]
    travel = float((positions[-1, 1]-positions[0, 1])*direction)
    if (lateral > 0.0002 or max(angles) > math.radians(0.1)
            or abs(travel-distance) > 0.0002
            or np.min(np.diff(positions[:, 1])*direction) < -1e-5):
        raise ValueError('装配末段未满足固定姿态、指定长度和单调Y轴直线')
    return {'axis': 'base_link/y', 'direction': direction, 'travel_m': travel,
            'max_lateral_error_m': lateral, 'max_orientation_error_rad': max(angles),
            'fk_samples': len(matrices)}


def flange_lever_bounds(urdf, joint_names):
    """由运行模型给出各转轴到 link6 的最大力臂，单位米。"""
    model = ET.fromstring(urdf)
    parents = {joint.find('child').get('link'): joint for joint in model.findall('joint')}
    link, distance, radii = 'link6', 0., {}
    visited = set()
    while link != 'base_link':
        if link in visited or link not in parents:
            raise ValueError('运行模型缺少 base_link 到 link6 的有效运动链')
        visited.add(link)
        joint = parents[link]
        if joint.get('type') != 'fixed':
            if joint.get('type') not in ('revolute', 'continuous'):
                raise ValueError('装配限速需要旋转关节模型')
            radii[joint.get('name')] = distance
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.] * 3
        if len(xyz) != 3 or not np.isfinite(xyz).all():
            raise ValueError('运行模型关节平移无效')
        distance += np.linalg.norm(xyz)
        link = joint.find('parent').get('link')
    if set(radii) != set(joint_names):
        raise ValueError('装配轨迹关节与运行模型运动链不一致')
    return np.array([radii[name] for name in joint_names])


def spline_flange_speed_bound(points, times, radii):
    """按 JTC 的线性、三次或五次插值计算整个轨迹的法兰线速度上界。"""
    peak = 0.
    for left, right, dt in zip(points, points[1:], np.diff(times)):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('限速需要严格递增的轨迹时间')
        q0, q1 = np.array(left.positions), np.array(right.positions)
        control = [q0, q1]
        if len(left.velocities) and len(right.velocities):
            v0, v1 = np.array(left.velocities), np.array(right.velocities)
            if len(left.accelerations) and len(right.accelerations):
                a0, a1 = np.array(left.accelerations), np.array(right.accelerations)
                control = [q0, q0+v0*dt/5, q0+2*v0*dt/5+a0*dt**2/20,
                           q1-2*v1*dt/5+a1*dt**2/20, q1-v1*dt/5, q1]
            else:
                control = [q0, q0+v0*dt/3, q1-v1*dt/3, q1]
        # 贝塞尔导数位于控制点凸包内，再用各关节力臂约束法兰线速度。
        velocities = (len(control)-1) * np.diff(control, axis=0) / dt
        peak = max(peak, float(np.max(np.abs(velocities) @ radii)))
    return peak


def limit_flange_speed(joints, radii, limit):
    times_ns = np.array([p.time_from_start.sec * 10**9 + p.time_from_start.nanosec
                         for p in joints.points], dtype=np.int64)
    bound = spline_flange_speed_bound(joints.points, times_ns * 1e-9, radii)
    # 留出纳秒取整余量，并按发送给控制器的最终时间、导数复核。
    slowdown = max(1., bound / limit * 1.000001)
    scaled_ns = np.rint(times_ns * slowdown).astype(np.int64)
    for point, ns in zip(joints.points, scaled_ns):
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(int(ns), 10**9)
        point.velocities = [v/slowdown for v in point.velocities]
        point.accelerations = [a/slowdown**2 for a in point.accelerations]
    actual_bound = spline_flange_speed_bound(joints.points, scaled_ns * 1e-9, radii)
    if not math.isfinite(actual_bound) or actual_bound > limit:
        raise ValueError('装配末段插值速度超过指定上限')
    return actual_bound
