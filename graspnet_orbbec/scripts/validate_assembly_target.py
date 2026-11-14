#!/usr/bin/env python3
"""通过官方 MoveIt 服务验证固定装配目标；没有执行、使能或夹爪接口。"""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from moveit_msgs.msg import Constraints, JointConstraint, PlanningSceneComponents, RobotState
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionFK, GetStateValidity


def call(node, kind, name, request, timeout=20):
    client = node.create_client(kind, '/piper_x/' + name)
    try:
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError(f'服务未就绪：{name}')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'服务超时或失败：{name}')
        return future.result()
    finally:
        node.destroy_client(client)


def state(base, names, values):
    result = copy.deepcopy(base)
    positions = dict(zip(result.joint_state.name, result.joint_state.position))
    positions.update(zip(names, map(float, values)))
    result.joint_state.name = list(positions)
    result.joint_state.position = list(positions.values())
    result.joint_state.velocity = []
    result.joint_state.effort = []
    result.is_diff = False
    return result


def validity(node, robot):
    request = GetStateValidity.Request()
    request.robot_state = robot
    request.group_name = 'arm'
    response = call(node, GetStateValidity, 'check_state_validity', request)
    return {'valid': response.valid, 'contacts': [
        [c.contact_body_1, c.contact_body_2] for c in response.contacts]}


def pose_matrix(pose):
    q = pose.orientation
    x, y, z, w = np.array([q.x, q.y, q.z, q.w]) / np.linalg.norm([q.x, q.y, q.z, q.w])
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return t


def validate(node, target_path, report):
    raw = target_path.read_bytes()
    target = json.loads(raw)
    names = target['joint_names']
    values = np.array(target['joint_positions_rad'], dtype=float)
    if names != [f'joint{i}' for i in range(1, 7)] or values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('需要按官方顺序排列的有限六轴目标')
    if target['reference_frame'] != 'base_link':
        raise ValueError('目标必须在 base_link')
    report.update(target_file=str(target_path), target_sha256=hashlib.sha256(raw).hexdigest(),
                  target_snapshot=target)
    params = GetParameters.Request()
    params.names = ['robot_description']
    urdf = call(node, GetParameters, 'move_group/get_parameters', params).values[0].string_value
    model = ET.fromstring(urdf)
    limits = {j.attrib['name']: j.find('limit').attrib for j in model.findall('joint')
              if j.attrib['name'] in names}
    for name, value in zip(names, values):
        if not float(limits[name]['lower']) <= value <= float(limits[name]['upper']):
            raise ValueError(f'{name}超出运行中模型限位')
    report['runtime_model'] = {'sha256': hashlib.sha256(urdf.encode()).hexdigest(),
                               'joint_limits': limits,
                               'links': [link.attrib['name'] for link in model.findall('link')]}
    request = GetPlanningScene.Request()
    request.components.components = (PlanningSceneComponents.ROBOT_STATE
        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        | PlanningSceneComponents.OCTOMAP)
    scene = call(node, GetPlanningScene, 'get_planning_scene', request).scene
    report['scene'] = message_to_ordereddict(scene)
    latest = {}
    seen = [0.0]

    def feedback(msg):
        if len(msg.name) != len(msg.position):
            return
        # 只接受同一条新反馈中的完整六轴及夹爪，避免混用历史消息。
        positions = dict(zip(msg.name, msg.position))
        if all(n in positions for n in names + ['gripper']):
            latest.clear()
            latest.update(positions)
            seen[0] = time.monotonic()

    sub = node.create_subscription(JointState, '/piper_x/feedback/joint_states', feedback,
                                   qos_profile_sensor_data)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not latest:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not latest or time.monotonic() - seen[0] > 0.5 or not np.isfinite(list(latest.values())).all():
        raise RuntimeError('缺少完整、新鲜的真实关节及夹爪反馈')
    base = state(scene.robot_state, list(latest), list(latest.values()))
    node.destroy_subscription(sub)
    report['start_positions'] = latest
    goal = state(base, names, values)
    report['start_validity'] = validity(node, base)
    report['goal_validity'] = validity(node, goal)
    fk = GetPositionFK.Request()
    fk.header.frame_id = 'base_link'
    fk.fk_link_names = ['link6', 'tcp_link']
    fk.robot_state = goal
    response = call(node, GetPositionFK, 'compute_fk', fk)
    if response.error_code.val != 1 or len(response.pose_stamped) != 2:
        raise RuntimeError('目标正解失败')
    report['goal_fk'] = message_to_ordereddict(response)
    matrices = {name: pose_matrix(p.pose) for name, p in zip(response.fk_link_names, response.pose_stamped)}
    expected = np.array(target['flange_goal']['T_base_frame'])
    actual = matrices['link6']
    position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    angle_error = float(math.acos(np.clip((np.trace(actual[:3, :3] @ expected[:3, :3].T)-1)/2, -1, 1)))
    report['goal_fk_error'] = {'position_m': position_error, 'orientation_rad': angle_error}
    if position_error > 0.001 or angle_error > math.radians(0.1):
        raise ValueError('运行中模型正解与记录法兰位姿不一致')
    if not np.allclose(matrices['tcp_link'], actual, atol=1e-7):
        raise ValueError('运行中 tcp_link 有非零工具偏移')
    tool_path = target_path.parent.parent / target['tool']['file']
    if hashlib.sha256(tool_path.read_bytes()).hexdigest() != target['tool']['sha256']:
        raise ValueError('工具配置已变化')
    if not np.allclose(expected @ np.array(target['tool']['T_flange_grasp']),
                       target['grasp_tcp_goal']['T_base_frame'], atol=1e-9):
        raise ValueError('目标工具补偿不一致')
    if not report['start_validity']['valid'] or not report['goal_validity']['valid']:
        raise ValueError('当前状态或固定目标在运行中场景无效')
    request = GetMotionPlan.Request()
    req = request.motion_plan_request
    req.group_name = 'arm'
    req.pipeline_id = 'ompl'
    req.start_state = base
    req.allowed_planning_time = 10.0
    req.num_planning_attempts = 8
    req.max_velocity_scaling_factor = 0.05
    req.max_acceleration_scaling_factor = 0.05
    constraints = Constraints()
    for name, value in zip(names, values):
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = float(value)
        constraint.tolerance_above = constraint.tolerance_below = 0.001
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)
    req.goal_constraints = [constraints]
    result = call(node, GetMotionPlan, 'plan_kinematic_path', request, timeout=50).motion_plan_response
    report['planning_response'] = message_to_ordereddict(result)
    if result.error_code.val != 1 or not result.trajectory.joint_trajectory.points:
        raise RuntimeError(f'规划失败：{result.error_code.val}')
    trajectory = result.trajectory.joint_trajectory
    if set(trajectory.joint_names) != set(names):
        raise ValueError('规划轨迹关节集合不符')
    ordered_goal = np.array([values[names.index(n)] for n in trajectory.joint_names])
    if np.max(np.abs(np.array(trajectory.points[-1].positions)-ordered_goal)) > 0.0011:
        raise ValueError('轨迹终点不满足固定关节目标')
    previous = np.array([latest[n] for n in trajectory.joint_names])
    count = 0
    # 额外细分检查离散路径；该采样检查不宣称连续扫掠体安全。
    for point in trajectory.points:
        current = np.array(point.positions)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise ValueError('轨迹点无效')
        steps = max(1, math.ceil(float(np.max(np.abs(current-previous))) / 0.02))
        for fraction in np.linspace(0, 1, steps+1)[1:]:
            sample = previous + fraction * (current-previous)
            for name, value in zip(trajectory.joint_names, sample):
                if not float(limits[name]['lower']) <= value <= float(limits[name]['upper']):
                    raise ValueError(f'路径采样 {name} 超出限位')
            checked = validity(node, state(base, trajectory.joint_names, sample))
            if not checked['valid']:
                report['invalid_path_sample'] = checked
                raise ValueError('路径细分采样状态无效')
            count += 1
        previous = current
    report['path_validity_samples'] = count
    report['status'] = 'planned_only'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    # 先以排他模式保留报告位置，失败也留下证据，不覆盖已有验证。
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        report = {'status': 'started', 'physical_motion_performed': False,
                  'motion_commands_sent': 0, 'started_at': datetime.now().astimezone().isoformat(),
                  'scope': 'current live state to recorded joint target; gripper held at current width',
                  'limitations': ['scene geometry only', 'no physical execution',
                                  'not a postgrasp loaded transfer validation',
                                  'discrete path samples; not continuous collision proof']}
        rclpy.init()
        node = Node('validate_assembly_target')
        try:
            validate(node, args.target.resolve(), report)
        except Exception as exc:
            report.update(status='failed', error=str(exc))
        finally:
            node.destroy_node()
            rclpy.shutdown()
            report['completed_at'] = datetime.now().astimezone().isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'error', 'goal_fk_error',
          'start_validity', 'goal_validity', 'path_validity_samples') if k in report}, ensure_ascii=False))
    print(f'报告：{args.report}')
    return 0 if report['status'] == 'planned_only' else 2


if __name__ == '__main__':
    raise SystemExit(main())
