#!/usr/bin/env python3
"""将视觉基座坐标交给官方MoveIt规划；默认只规划，--execute才允许真机动作。"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm.moveit_wire_adapter import build_motion_spec, load_recognition, write_json_exclusive
from arm.handeye_bridge import sha256
from arm.feedback_safety import MotionAudit, update_joints, require_joints
from arm.assembly_target import (flange_lever_bounds, limit_flange_speed, load_assembly_target,
                                 validate_runtime_assembly, validate_y_approach)
from arm.grasp_angle import closing_angle_deg, CLOSING_MIN_DEG
from arm.moveit_wire_adapter import transform_to_pose

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetPositionFK, GetCartesianPath, GetStateValidity
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rosidl_runtime_py.convert import message_to_ordereddict


DEFAULT_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def parser():
    node = argparse.ArgumentParser(description=__doc__)
    node.add_argument('--recognition', type=Path, required=True)
    node.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json',
                      help='入口独立指定冻结标定，不从recognition选择')
    node.add_argument('--candidate-id')
    node.add_argument('--namespace', default='/piper_x')
    node.add_argument('--group', default='arm')
    node.add_argument('--base-frame', default='base_link')
    node.add_argument('--tip-link', default='tcp_link')
    node.add_argument('--expected-parent-frame', default='link6')
    node.add_argument('--joint-state-topic', default='feedback/joint_states')
    node.add_argument('--tcp-pose-topic', default='feedback/tcp_pose')
    node.add_argument('--control-topic', default='control/joint_states')
    node.add_argument('--gate-service', default='control_enable')
    node.add_argument('--arm-status-topic', default='feedback/arm_status')
    node.add_argument('--motor-status-topic', default='feedback/motor_status')
    node.add_argument('--gripper-status-topic', default='feedback/gripper_status')
    node.add_argument('--joints', nargs='+', default=DEFAULT_JOINTS)
    node.add_argument('--vel', type=float, default=0.05, help='MoveIt速度缩放，最大0.10')
    node.add_argument('--acc', type=float, default=0.05, help='MoveIt加速度缩放，最大0.10')
    node.add_argument('--planning-time', type=float, default=10.0)
    node.add_argument('--planning-attempts', type=int, default=8)
    node.add_argument('--goal-timeout', type=float, default=120.0)
    node.add_argument('--arrival-timeout', type=float, default=20.0)
    node.add_argument('--feedback-timeout', type=float, default=20.0)
    node.add_argument('--position-tolerance-m', type=float, default=0.003)
    node.add_argument('--orientation-tolerance-rad', type=float, default=0.05)
    node.add_argument('--arrival-position-tolerance-m', type=float, default=0.01)
    node.add_argument('--arrival-orientation-tolerance-rad', type=float, default=0.10)
    node.add_argument('--arrival-joint-tolerance-rad', type=float, default=0.02,
                      help='装配六轴到位容差；Y轴末段稳定验收额外收紧至最多0.005rad')
    node.add_argument('--assembly-target', type=Path,
                      help='显式追加固定装配关节目标；抬升后保持夹持并转移，不自动松爪')
    node.add_argument(
        '--allow-nominal-urdf-tool',
        action='store_true',
        help='用户显式授权按URDF名义工具参数执行；不表示该TCP已实测',
    )
    node.add_argument('--execute', action='store_true', help='执行抓取；指定装配目标时抬升后继续转移')
    node.add_argument('--report', type=Path, help='完整报告输出路径；默认写入新的时间戳目录')
    return node


def namespaced(namespace, topic):
    prefix = namespace.strip('/')
    suffix = topic.strip('/')
    return f'/{prefix}/{suffix}' if prefix else f'/{suffix}'


def resolved(namespace, topic):
    return topic if topic.startswith('/') else namespaced(namespace, topic)


def matrix_from_parts(position, orientation):
    q = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=float)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('反馈四元数无效')
    x, y, z, w = q / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [position.x, position.y, position.z]
    return transform


def matrix_from_pose(pose):
    return matrix_from_parts(pose.position, pose.orientation)


def matrix_from_transform(transform):
    return matrix_from_parts(transform.translation, transform.rotation)


def pose_matrix(value):
    position = value['position']
    orientation = value['orientation']
    pose = Pose()
    pose.position.x = position['x']
    pose.position.y = position['y']
    pose.position.z = position['z']
    pose.orientation.x = orientation['x']
    pose.orientation.y = orientation['y']
    pose.orientation.z = orientation['z']
    pose.orientation.w = orientation['w']
    return pose


def rotation_angle(left, right):
    relative = left.T @ right
    return math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def diagnostic_level(value):
    # Humble的本机绑定可能把uint8 level暴露成单字节bytes。
    if isinstance(value, (bytes, bytearray)):
        return int(value[0])
    return int(value)


def reverse_cartesian_trajectory(solution, stage):
    trajectory = copy.deepcopy(solution)
    joints = trajectory.joint_trajectory
    if (set(joints.joint_names) != set(stage['joint_names']) or len(joints.points) < 2
            or trajectory.multi_dof_joint_trajectory.points):
        raise ValueError('装配笛卡尔轨迹需要完整六轴和至少两个轨迹点')
    times = [p.time_from_start.sec*10**9+p.time_from_start.nanosec for p in joints.points]
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('装配笛卡尔轨迹缺少有效递增时间戳')
    previous = None
    for point in joints.points:
        if len(point.positions) != 6 or not np.isfinite(point.positions).all():
            raise ValueError('装配笛卡尔轨迹关节值无效')
        for vector in (point.velocities, point.accelerations, point.effort):
            if vector and (len(vector) != 6 or not np.isfinite(vector).all()):
                raise ValueError('装配笛卡尔轨迹导数无效')
        if previous is not None and np.max(np.abs(np.array(point.positions)-previous)) > 0.1:
            raise ValueError('装配笛卡尔轨迹存在大于0.1rad的关节跳变')
        previous = np.array(point.positions)
    expected = [stage['joint_positions_rad'][stage['joint_names'].index(n)] for n in joints.joint_names]
    if np.max(np.abs(np.array(joints.points[0].positions)-expected)) > 1e-6:
        raise ValueError('装配笛卡尔轨迹起点不是记录的固定六轴目标')
    for point, ns in zip(joints.points, times):
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(times[-1]-ns, 10**9)
        point.velocities = [-v for v in point.velocities]
        point.effort = []
    joints.points = list(reversed(joints.points))
    return trajectory


class WireMoveGroup(Node):
    def __init__(self, args, spec):
        super().__init__('wire_moveit_pose_adapter')
        self.args = args
        self.spec = spec
        self._joint_positions = {}
        self._tcp_pose = None
        self._joint_seen_at = {}
        self._motion_audit = MotionAudit()
        self._tcp_seen_at = 0.0
        self._diagnostics = {}
        self._diagnostic_seen_at = {}
        self._move = ActionClient(self, MoveGroup, namespaced(args.namespace, 'move_action'))
        self._execute_trajectory = ActionClient(self, ExecuteTrajectory,
                                                namespaced(args.namespace, 'execute_trajectory'))
        self._assembly_cartesian = None
        self._gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            namespaced(args.namespace, 'gripper_controller/follow_joint_trajectory'),
        )
        self._gate = self.create_client(SetBool, resolved(args.namespace, args.gate_service))
        # 官方实机驱动持续发布volatile；mock的broadcaster只缓存transient-local首帧。
        joint_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        joint_live_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._joint_qos = joint_qos
        self._joint_sub = self.create_subscription(
            JointState, resolved(args.namespace, args.joint_state_topic), self._joint_callback, joint_live_qos
        )
        self.create_subscription(
            PoseStamped, resolved(args.namespace, args.tcp_pose_topic), self._tcp_callback, 10
        )
        status_topics = {
            'arm_status': args.arm_status_topic,
            'motor_status': args.motor_status_topic,
            'gripper_status': args.gripper_status_topic,
        }
        for name, topic in status_topics.items():
            self.create_subscription(
                DiagnosticArray,
                resolved(args.namespace, topic),
                lambda msg, key=name: self._diagnostic_callback(key, msg),
                10,
            )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _sample_time(self, message):
        # 驱动的 header 使用采样时间；转到单调时钟后，重发旧包不会延长有效期。
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
        if stamp <= 0 or age < 0:
            return -math.inf
        return time.monotonic() - age

    def _joint_callback(self, message):
        before = dict(self._joint_positions)
        update_joints(self._joint_positions, self._joint_seen_at, message.name, message.position, self._sample_time(message))
        fresh = {name: self._joint_positions[name] for name in message.name
                 if name in self._joint_positions and time.monotonic() - self._joint_seen_at[name] <= 2.5}
        self._motion_audit.observe(before, fresh)

    def _require_joint_feedback(self, names=None):
        require_joints(self._joint_positions, self._joint_seen_at,
                       self.args.joints if names is None else names, time.monotonic())

    def _tcp_callback(self, message):
        self._tcp_pose = message.pose
        self._tcp_seen_at = self._sample_time(message)

    def _diagnostic_callback(self, name, message):
        self._diagnostics[name] = [
            {
                'name': status.name,
                'level': diagnostic_level(status.level),
                'message': status.message,
                'values': {item.key: item.value for item in status.values},
            }
            for status in message.status
        ]
        self._diagnostic_seen_at[name] = self._sample_time(message)

    def _spin_for(self, seconds, predicate=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate is not None and predicate():
                return True
        return predicate() if predicate is not None else True

    def _wait_joint_feedback(self):
        def complete():
            now = time.monotonic()
            return all(joint in self._joint_positions and now - self._joint_seen_at[joint] <= 2.5
                       for joint in self.args.joints)

        started = time.monotonic()
        initial_wait = min(2.0, self.args.feedback_timeout)
        if self._spin_for(initial_wait, complete):
            return True
        # 保留实机 volatile 订阅，让慢启动的驱动在剩余等待时间内仍能匹配。
        self._joint_cached_sub = self.create_subscription(
            JointState,
            resolved(self.args.namespace, self.args.joint_state_topic),
            self._joint_callback,
            self._joint_qos,
        )
        remaining = max(0.0, self.args.feedback_timeout - (time.monotonic() - started))
        return self._spin_for(remaining, complete)

    def wait_ready(self, execute):
        if self.args.vel <= 0 or self.args.vel > 0.10 or self.args.acc <= 0 or self.args.acc > 0.10:
            raise ValueError('速度/加速度缩放必须在(0,0.10]内')
        if self.args.position_tolerance_m <= 0 or self.args.position_tolerance_m > 0.01:
            raise ValueError('位置容差必须在(0,10mm]内')
        if self.args.orientation_tolerance_rad <= 0 or self.args.orientation_tolerance_rad > 0.15:
            raise ValueError('姿态容差必须在(0,0.15rad]内')
        if self.args.arrival_position_tolerance_m <= 0 or self.args.arrival_position_tolerance_m > 0.02:
            raise ValueError('执行后位置验收容差必须在(0,20mm]内')
        if (self.args.arrival_orientation_tolerance_rad <= 0
                or self.args.arrival_orientation_tolerance_rad > 0.15):
            raise ValueError('执行后姿态验收容差必须在(0,0.15rad]内')
        if not 0 < self.args.arrival_joint_tolerance_rad <= 0.05:
            raise ValueError('装配关节到位容差必须在(0,0.05rad]内')
        if not self._wait_joint_feedback():
            missing = [joint for joint in self.args.joints if joint not in self._joint_positions]
            raise RuntimeError(f'未收到完整真实关节反馈，缺少{missing}')
        self._require_joint_feedback()
        if not self._move.wait_for_server(timeout_sec=self.args.feedback_timeout):
            raise RuntimeError(f'MoveGroup Action未就绪：{namespaced(self.args.namespace, "move_action")}')
        self._require_parent_tip_identity()
        if not execute:
            return
        if not self.spec.get('execution_allowed'):
            raise ValueError('视觉目标只允许规划，真机执行阻断：' + ', '.join(self.spec['execution_blockers']))
        if not self._spin_for(5.0, lambda: self._tcp_pose is not None):
            raise RuntimeError('未收到真实TCP位姿反馈')
        if time.monotonic() - self._tcp_seen_at > 2.5:
            raise ValueError('真实TCP位姿反馈陈旧')
        required = ('arm_status', 'motor_status', 'gripper_status')
        if not self._spin_for(5.0, lambda: all(name in self._diagnostics for name in required)):
            missing = [name for name in required if name not in self._diagnostics]
            raise RuntimeError(f'未收到完整安全诊断：{missing}')
        self._require_safe_diagnostics()

    def _require_parent_tip_identity(self):
        deadline = time.monotonic() + 5.0
        error = None
        while time.monotonic() < deadline:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.args.expected_parent_frame, self.args.tip_link, Time()
                ).transform
                matrix = matrix_from_transform(transform)
                position_error = float(np.linalg.norm(matrix[:3, 3]))
                angle_error = rotation_angle(np.eye(3), matrix[:3, :3])
                if (position_error > self.spec['parent_to_tip_max_position_error_m']
                        or angle_error > self.spec['parent_to_tip_max_rotation_error_rad']):
                    raise ValueError(
                        f'MoveIt末端{self.args.expected_parent_frame}->{self.args.tip_link}非单位偏置：'
                        f'{position_error * 1000:.2f}mm/{math.degrees(angle_error):.2f}deg；'
                        '视觉输出是法兰目标，请用tcp_offset:=0启动或转换目标'
                    )
                return {
                    'parent_frame': self.args.expected_parent_frame,
                    'tip_link': self.args.tip_link,
                    'translation_m': matrix[:3, 3].tolist(),
                    'rotation_rad': angle_error,
                }
            except Exception as exc:  # TF 可能尚未完成首帧
                error = exc
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f'无法验证MoveIt末端坐标偏置：{error}')

    def _require_safe_diagnostics(self):
        now = time.monotonic()
        for name in ('arm_status', 'motor_status', 'gripper_status'):
            if now - self._diagnostic_seen_at.get(name, 0.0) > 2.5:
                raise ValueError(f'{name}反馈陈旧')
            statuses = self._diagnostics[name]
            if not statuses:
                raise ValueError(f'{name}没有诊断条目')
            ok_level = diagnostic_level(DiagnosticStatus.OK)
            if any(status['level'] != ok_level for status in statuses):
                details = [
                    f"{status['name']}: {status['message']}"
                    for status in statuses if status['level'] != ok_level
                ]
                raise ValueError(f'{name}报告非OK状态：' + '; '.join(details))
        expected = {'arm_status': ['arm'], 'motor_status': DEFAULT_JOINTS, 'gripper_status': ['gripper']}
        for topic, names in expected.items():
            if sorted(s['name'] for s in self._diagnostics[topic]) != sorted(names):
                raise ValueError(f'{topic}诊断条目缺失、重复或名称错误')
        arm = self._diagnostics['arm_status'][0]['values']
        if arm.get('teach_status') not in ('0', '2', '6'):
            raise ValueError('示教状态缺失或仍在记录/回放')
        if arm.get('ctrl_mode') not in ('0', '1'):
            raise ValueError(f"当前控制模式不适合MoveIt执行：ctrl_mode={arm.get('ctrl_mode')}")
        if arm.get('arm_status') != '0':
            raise ValueError(f"机械臂状态非待机：arm_status={arm.get('arm_status')}")
        for status in self._diagnostics['motor_status']:
            if status['values'].get('enabled') != 'true':
                raise ValueError(f"{status['name']}未使能；本程序不会自动使能")
        gripper = self._diagnostics['gripper_status'][0]['values']
        if gripper.get('driver_enable_status') != 'true':
            raise ValueError('夹爪未使能；本程序不会自动使能')

    def _goal(self, stage, plan_only, start_positions=None, start_is_diff=True):
        args = self.args
        constraints = Constraints()
        frame_id = self.spec['frame_id']
        target_pose = stage['pose']

        position = PositionConstraint()
        position.header.frame_id = frame_id
        position.link_name = self.spec['tip_link']
        position.target_point_offset.x = 0.0
        position.target_point_offset.y = 0.0
        position.target_point_offset.z = 0.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [args.position_tolerance_m]
        position.constraint_region = BoundingVolume()
        position.constraint_region.primitives.append(sphere)
        primitive_pose = pose_matrix(target_pose)
        primitive_pose.orientation.x = 0.0
        primitive_pose.orientation.y = 0.0
        primitive_pose.orientation.z = 0.0
        primitive_pose.orientation.w = 1.0
        position.constraint_region.primitive_poses.append(primitive_pose)
        position.weight = 1.0
        constraints.position_constraints.append(position)

        orientation = OrientationConstraint()
        orientation.header.frame_id = frame_id
        orientation.link_name = self.spec['tip_link']
        orientation.orientation = pose_matrix(target_pose).orientation
        orientation.absolute_x_axis_tolerance = args.orientation_tolerance_rad
        orientation.absolute_y_axis_tolerance = args.orientation_tolerance_rad
        orientation.absolute_z_axis_tolerance = args.orientation_tolerance_rad
        orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)

        if stage.get('kind') == 'joint':
            # 固定关节构型才能复现示教姿态，避免位姿逆解切换到另一套关节解。
            constraints = Constraints()
            for name, value in zip(stage['joint_names'], stage['joint_positions_rad']):
                joint = JointConstraint()
                joint.joint_name = name
                joint.position = float(value)
                joint.tolerance_above = joint.tolerance_below = stage['joint_goal_tolerance_rad']
                joint.weight = 1.0
                constraints.joint_constraints.append(joint)

        goal = MoveGroup.Goal()
        goal.request.group_name = args.group
        if stage.get('name') == 'grasp':
            # LIN 从预抓取末态沿笛卡尔直线到夹持点，禁止退回自由关节路径。
            goal.request.pipeline_id = 'pilz_industrial_motion_planner'
            goal.request.planner_id = 'LIN'
        else:
            goal.request.pipeline_id = 'ompl'
        goal.request.goal_constraints = [constraints]
        goal.request.num_planning_attempts = args.planning_attempts
        goal.request.allowed_planning_time = args.planning_time
        goal.request.max_velocity_scaling_factor = args.vel
        goal.request.max_acceleration_scaling_factor = args.acc
        goal.request.start_state = RobotState()
        goal.request.start_state.is_diff = start_is_diff
        if start_positions is not None:
            goal.request.start_state.joint_state.name = list(args.joints)
            goal.request.start_state.joint_state.position = [float(value) for value in start_positions]
        if plan_only:
            if stage.get('name') in ('pregrasp', 'grasp'):
                width = self.spec['gripper']['open_width_m']
            else:
                width = stage.get('planning_gripper_width_m')
            if width is not None:
                goal.request.start_state.joint_state.name.append('gripper')
                goal.request.start_state.joint_state.position.append(width)
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        return goal

    def plan_or_execute(self, name, stage, plan_only, start_positions=None, start_is_diff=True):
        self._require_joint_feedback()
        if stage.get('kind') == 'cartesian_y':
            return self.plan_or_execute_cartesian(stage, plan_only, start_positions)
        started = time.monotonic()
        goal = self._goal(stage, plan_only, start_positions, start_is_diff)
        if not plan_only:
            self._require_safe_diagnostics()
        send_future = self._move.send_goal_async(goal)
        if not self._spin_for(10.0, send_future.done):
            raise RuntimeError(f'{name}发送超时')
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f'{name}目标未被MoveGroup接受')
        wrapped = self._action_result(handle, self.args.goal_timeout, name, monitor=not plan_only)
        if (wrapped.status != GoalStatus.STATUS_SUCCEEDED
                or wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS):
            raise RuntimeError(
                f'{name}失败：status={wrapped.status}, '
                f'error_code={wrapped.result.error_code.val}'
            )
        trajectory = wrapped.result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            raise RuntimeError(f'{name}返回空轨迹')
        final_positions = list(trajectory.points[-1].positions)
        if len(final_positions) != len(trajectory.joint_names):
            raise RuntimeError(f'{name}轨迹末点关节数不一致')
        final_positions = [final_positions[trajectory.joint_names.index(joint)]
                           for joint in self.args.joints]
        return {
            'name': name,
            'ok': True,
            'plan_only': plan_only,
            'elapsed_s': time.monotonic() - started,
            'planning_time_s': float(wrapped.result.planning_time),
            'trajectory_points': len(trajectory.points),
            'final_joint_positions': [float(value) for value in final_positions],
        }

    def _action_result(self, handle, timeout, name, *, monitor):
        future = handle.get_result_async()
        def finished():
            if monitor:
                self._require_safe_diagnostics()
            return future.done()
        try:
            if not self._spin_for(timeout, finished):
                raise RuntimeError(f'{name}在{timeout:g}s内未返回')
            return future.result()
        except (Exception, KeyboardInterrupt):
            cancellation = handle.cancel_goal_async()
            self._spin_for(2.0, cancellation.done)
            raise

    def _call_service(self, kind, service, request):
        client = self.create_client(kind, namespaced(self.args.namespace, service))
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f'装配校验服务未就绪：{service}')
            future = client.call_async(request)
            if not self._spin_for(20.0, future.done) or future.result() is None:
                raise RuntimeError(f'装配校验服务超时：{service}')
            return future.result()
        finally:
            self.destroy_client(client)

    def _assembly_fk(self, names, positions):
        request = GetPositionFK.Request()
        request.header.frame_id = self.args.base_frame
        request.fk_link_names = ['link6']
        request.robot_state.joint_state.name = list(names)
        request.robot_state.joint_state.position = list(positions)
        response = self._call_service(GetPositionFK, 'compute_fk', request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS or len(response.pose_stamped) != 1:
            raise ValueError('装配目标官方正解校验失败')
        return matrix_from_pose(response.pose_stamped[0].pose)

    def validate_assembly_goal(self, stage):
        if self.args.expected_parent_frame != 'link6':
            raise ValueError('装配目标法兰必须使用 link6')
        request = GetParameters.Request()
        request.names = ['robot_description']
        urdf = self._call_service(GetParameters, 'move_group/get_parameters', request).values[0].string_value
        actual = self._assembly_fk(stage['joint_names'], stage['joint_positions_rad'])
        result = validate_runtime_assembly(stage, urdf, actual)
        result['flange_lever_bounds_m'] = dict(zip(stage['joint_names'], flange_lever_bounds(urdf, stage['joint_names'])))
        result['T_base_flange_model'] = actual.tolist()
        return result

    def prepare_assembly_approach(self, stage):
        runtime = self.validate_assembly_goal(stage)
        goal_matrix = np.array(runtime['T_base_flange_model'])
        approach = stage['final_approach']
        entry_matrix = goal_matrix.copy()
        entry_matrix[1, 3] -= approach['direction'] * approach['distance_m']
        request = GetCartesianPath.Request()
        request.header.frame_id = self.args.base_frame
        request.group_name = self.args.group
        request.link_name = self.args.tip_link
        request.start_state.joint_state.name = list(stage['joint_names']) + ['gripper']
        request.start_state.joint_state.position = list(stage['joint_positions_rad']) + [0.0]
        request.waypoints = [pose_matrix(transform_to_pose(entry_matrix))]
        request.max_step = 0.001
        request.jump_threshold = 2.0
        request.revolute_jump_threshold = 0.1
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.args.vel
        request.max_acceleration_scaling_factor = self.args.acc
        # 从固定六轴目标反向求直线，再反转时序，保证终点仍为同一套示教关节解。
        response = self._call_service(GetCartesianPath, 'compute_cartesian_path', request)
        if (response.error_code.val != MoveItErrorCodes.SUCCESS or not math.isfinite(response.fraction)
                or response.fraction < 1.0-1e-9):
            raise ValueError(f'装配Y轴直线未完整规划：fraction={response.fraction:.6f}')
        trajectory = reverse_cartesian_trajectory(response.solution, stage)
        joints = trajectory.joint_trajectory
        transforms = [self._assembly_fk(joints.joint_names, point.positions) for point in joints.points]
        checked = validate_y_approach(transforms, approach['direction'], approach['distance_m'])
        radii = [runtime['flange_lever_bounds_m'][name] for name in joints.joint_names]
        speed_bound = limit_flange_speed(joints, radii, float(approach.get('max_speed_m_s', 0.01)))
        self._assembly_cartesian = trajectory
        stage['kind'] = 'cartesian_y'
        stage['cartesian_check'] = dict(checked, max_speed_m_s=speed_bound, speed_check="spline_upper_bound",
                                         trajectory=message_to_ordereddict(trajectory))
        entry = copy.deepcopy(stage)
        entry.update(name='assembly_entry', kind='joint', final_approach=None,
                     joint_names=list(joints.joint_names),
                     joint_positions_rad=list(joints.points[0].positions),
                     T_base_flange=transforms[0].tolist(), pose=transform_to_pose(transforms[0]))
        entry.pop('cartesian_check', None)
        return entry

    def plan_or_execute_cartesian(self, stage, plan_only, start_positions):
        if self._assembly_cartesian is None:
            raise ValueError('装配直线尚未准备')
        trajectory = self._assembly_cartesian
        joints = trajectory.joint_trajectory
        expected = [joints.points[0].positions[joints.joint_names.index(n)] for n in self.args.joints]
        actual = start_positions if plan_only else [self._joint_positions[n] for n in self.args.joints]
        if actual is None or np.max(np.abs(np.array(actual)-expected)) > 0.005:
            raise ValueError('装配直线入口六轴偏差超过0.005rad，拒绝拼接')
        if not plan_only:
            for point in joints.points:
                request = GetStateValidity.Request()
                request.group_name = self.args.group
                request.robot_state.joint_state.name = list(joints.joint_names) + ['gripper']
                self._require_joint_feedback(['gripper'])
                request.robot_state.joint_state.position = list(point.positions) + [self._joint_positions['gripper']]
                checked = self._call_service(GetStateValidity, 'check_state_validity', request)
                if not checked.valid:
                    raise ValueError('装配直线执行前碰撞复查失败')
            self._require_safe_diagnostics()
            self.require_held_gripper()
            self._require_joint_feedback()
            actual = [self._joint_positions[n] for n in self.args.joints]
            if np.max(np.abs(np.array(actual)-expected)) > 0.005:
                raise ValueError('装配直线执行前入口状态发生变化')
            if not self._execute_trajectory.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('官方ExecuteTrajectory未就绪')
            self._require_safe_diagnostics()
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = trajectory
            future = self._execute_trajectory.send_goal_async(goal)
            if not self._spin_for(10.0, future.done):
                raise RuntimeError('装配直线发送超时')
            handle = future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError('装配直线执行被拒绝')
            wrapped = self._action_result(handle, self.args.goal_timeout, '装配直线', monitor=True)
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError('装配直线执行失败')
        return {'name': 'assembly', 'ok': True, 'plan_only': plan_only,
                'trajectory_points': len(joints.points), 'cartesian_check': stage['cartesian_check'],
                'final_joint_positions': [joints.points[-1].positions[joints.joint_names.index(n)]
                                          for n in self.args.joints]}

    def require_held_gripper(self):
        self._require_joint_feedback(['gripper'])
        width = self._joint_positions['gripper']
        if not 0 <= width <= 0.01:
            raise ValueError(f'装配转移前夹爪未保持闭合：{width:.4f}m')
        return {'width_m': width, 'note': '闭合反馈不等于独立确认夹持成功'}

    def wait_actual_joints(self, stage):
        errors = {}
        def arrived():
            self._require_joint_feedback(stage['joint_names'])
            errors.update({name: abs(self._joint_positions[name] - value)
                           for name, value in zip(stage['joint_names'], stage['joint_positions_rad'])})
            return max(errors.values()) <= self.args.arrival_joint_tolerance_rad
        if not self._spin_for(self.args.arrival_timeout, arrived):
            raise RuntimeError(f'装配六轴到位超时：{errors}')
        return {'joint_errors_rad': dict(errors),
                'tolerance_rad': self.args.arrival_joint_tolerance_rad}

    def wait_assembly_arrival(self, stage):
        target = np.array(stage['T_base_flange'])
        stable_since = None
        measured = {}
        def stable():
            nonlocal stable_since, measured
            now = time.monotonic()
            self._require_joint_feedback(stage['joint_names'])
            if self._tcp_pose is None or now-self._tcp_seen_at > 0.1:
                stable_since = None
                return False
            actual = matrix_from_pose(self._tcp_pose)
            errors = {n: abs(self._joint_positions[n]-q)
                      for n, q in zip(stage['joint_names'], stage['joint_positions_rad'])}
            measured = {'position_error_m': float(np.linalg.norm(actual[:3,3]-target[:3,3])),
                        'rotation_error_rad': rotation_angle(actual[:3,:3], target[:3,:3]),
                        'joint_errors_rad': errors,
                        'actual_joint_positions_rad': [self._joint_positions[n] for n in stage['joint_names']],
                        'actual_T_base_flange': actual.tolist()}
            # 同一轮反馈同时满足关节、位置及姿态，持续0.5秒后才报告装配到位。
            good = (measured['position_error_m'] <= 0.002 and measured['rotation_error_rad'] <= 0.01
                    and max(errors.values()) <= min(self.args.arrival_joint_tolerance_rad, 0.005))
            if not good:
                stable_since = None
                return False
            stable_since = now if stable_since is None else stable_since
            return now-stable_since >= 0.5
        if not self._spin_for(self.args.arrival_timeout, stable):
            raise RuntimeError(f'装配稳定到位超时（2mm/0.01rad/0.005rad，持续0.5s）：{measured}')
        return dict(measured, stable_duration_s=0.5)

    def wait_grasp_angle(self):
        review = self.spec['closing_angle_review']
        tangent = review['wire_tangent_base']
        plan = next(s for s in self.spec['stages'] if s['name'] == 'grasp')
        flange = np.array(plan['T_base_flange'])[:3,:3]
        # 从已验收的目标轴得到工具旋转偏置，真实闭合轴随法兰反馈更新。
        goal_closing = np.array(self.spec['grasp_closing_direction_base'])
        closing_in_flange = flange.T @ goal_closing
        result = {}
        def valid():
            if self._tcp_pose is None or time.monotonic()-self._tcp_seen_at > 0.1:
                return False
            angle = closing_angle_deg(matrix_from_pose(self._tcp_pose)[:3,:3] @ closing_in_flange, tangent)
            result['actual_closing_angle_deg'] = angle
            return angle >= CLOSING_MIN_DEG-1e-8
        if not self._spin_for(self.args.arrival_timeout, valid):
            raise ValueError(f'闭合前真实夹角不满足85–90度：{result}')
        return result

    def wait_actual_pose(self, name, target_pose, timeout=None):
        timeout = self.args.arrival_timeout if timeout is None else timeout
        target = matrix_from_pose(pose_matrix(target_pose))
        expected_at = time.monotonic() + timeout
        best = None
        while time.monotonic() < expected_at:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._tcp_pose is None:
                continue
            if time.monotonic() - self._tcp_seen_at > 1.0:
                continue
            actual = matrix_from_pose(self._tcp_pose)
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle_error = rotation_angle(actual[:3, :3], target[:3, :3])
            best = (position_error, angle_error)
            if (position_error <= self.args.arrival_position_tolerance_m
                    and angle_error <= self.args.arrival_orientation_tolerance_rad):
                return {
                    'name': name,
                    'position_error_m': position_error,
                    'rotation_error_rad': angle_error,
                    'tcp_feedback_age_s': time.monotonic() - self._tcp_seen_at,
                }
        if best is None:
            raise RuntimeError(f'{name}执行后没有收到真实TCP反馈')
        raise RuntimeError(
            f'{name}执行后真实TCP未到目标：'
            f'{best[0] * 1000:.2f}mm/{math.degrees(best[1]):.2f}deg'
        )

    def call_gate(self, enabled):
        if not self._gate.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('官方控制门控服务未就绪')
        request = SetBool.Request()
        request.data = bool(enabled)
        future = self._gate.call_async(request)
        if not self._spin_for(5.0, future.done):
            raise RuntimeError('控制门控服务响应超时')
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f'控制门控调用失败：{getattr(response, "message", "")}')

    def command_gripper(self, width, effort, *, expect_open=False):
        self._require_joint_feedback(list(self.args.joints)+['gripper'])
        preclose = self._joint_positions.get('gripper')
        self._require_safe_diagnostics()
        try:
            self.call_gate(True)
            if not self._gripper_action.wait_for_server(timeout_sec=self.args.feedback_timeout):
                raise RuntimeError(
                    '官方夹爪FollowJointTrajectory Action未就绪：'
                    f'{namespaced(self.args.namespace, "gripper_controller/follow_joint_trajectory")}'
                )
            point = JointTrajectoryPoint()
            point.positions = [float(width)]
            # 官方JTC明确拒绝effort字段；夹持力使用驱动的gripper_default_effort。
            point.time_from_start = RosDuration(sec=2, nanosec=0)
            trajectory = JointTrajectory()
            trajectory.joint_names = ['gripper']
            trajectory.points = [point]
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = trajectory
            self._require_safe_diagnostics()
            send_future = self._gripper_action.send_goal_async(goal)
            if not self._spin_for(10.0, send_future.done):
                raise RuntimeError('夹爪轨迹发送超时')
            handle = send_future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError('夹爪轨迹未被官方控制器接受')
            wrapped = self._action_result(handle, 15.0, '夹爪轨迹', monitor=True)
            if (wrapped.status != GoalStatus.STATUS_SUCCEEDED
                    or wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
                raise RuntimeError(
                    f'夹爪轨迹执行失败：status={wrapped.status}, '
                    f'error_code={wrapped.result.error_code}'
                )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if time.monotonic()-self._joint_seen_at.get('gripper', -math.inf) > 1.:
                    continue
                actual = self._joint_positions.get('gripper')
                if actual is None:
                    continue
                if expect_open and actual >= width - 0.01:
                    break
                if not expect_open and actual <= 0.01:
                    break
            self._require_joint_feedback(['gripper'])
            actual = self._joint_positions.get('gripper')
            if actual is None or (expect_open and actual < width - 0.01):
                raise RuntimeError(f'夹爪未达到开口目标：requested={width}, actual={actual}')
            if not expect_open and actual > 0.01:
                raise RuntimeError(f'夹爪闭合反馈未下降：before={preclose}, actual={actual}')
            return {
                'commanded_width_m': float(width),
                'commanded_force_n': float(effort),
                'width_before_m': preclose,
                'width_after_m': self._joint_positions.get('gripper'),
            }
        finally:
            self.call_gate(False)


def run_motion_sequence(node, args, spec, report, recheck_source):
    """成功完成闭合及抬升后才转移；任一异常直接中断，不自动松爪。"""
    assembly = next((stage for stage in spec['stages'] if stage['name'] == 'assembly'), None)
    if args.execute:
        recheck_source()
        node._motion_audit.attempt('gripper_open')
        report['gripper_results'].append(
            node.command_gripper(spec['gripper']['open_width_m'],
                                 spec['gripper']['force_n'], expect_open=True))
    start_positions, start_is_diff = None, True
    for name in ('pregrasp', 'grasp'):
        if args.execute:
            recheck_source()
            node._motion_audit.attempt(name)
        stage = next(item for item in spec['stages'] if item['name'] == name)
        result = node.plan_or_execute(name, stage, not args.execute, start_positions, start_is_diff)
        report['motion_results'].append(result)
        if args.execute:
            report['feedback_checks'][f'{name}_arrival'] = node.wait_actual_pose(name, stage['pose'])
        else:
            start_positions, start_is_diff = result['final_joint_positions'], False
    if args.execute:
        recheck_source()
        if spec.get('closing_angle_review', {}).get('passed'):
            report['feedback_checks']['grasp_closing_angle'] = node.wait_grasp_angle()
        node._motion_audit.attempt('gripper_close')
        report['gripper_results'].append(
            node.command_gripper(spec['gripper']['close_width_m'], spec['gripper']['force_n']))
    if args.execute or assembly is not None:
        lift = next(item for item in spec['stages'] if item['name'] == 'lift')
        if args.execute:
            recheck_source()
            node._motion_audit.attempt('lift')
        result = node.plan_or_execute('lift', lift, not args.execute, start_positions, start_is_diff)
        report['motion_results'].append(result)
        if args.execute:
            report['feedback_checks']['lift_arrival'] = node.wait_actual_pose('lift', lift['pose'])
        else:
            start_positions, start_is_diff = result['final_joint_positions'], False
    if assembly is not None:
        if args.execute:
            recheck_source()
            node._require_safe_diagnostics()
            report['feedback_checks']['assembly_runtime'] = node.validate_assembly_goal(assembly)
            report['feedback_checks']['assembly_gripper_before'] = node.require_held_gripper()
        entry = next((s for s in spec['stages'] if s['name'] == 'assembly_entry'), None)
        if entry is not None:
            if args.execute:
                recheck_source()
                node._motion_audit.attempt('assembly_entry')
            result = node.plan_or_execute('assembly_entry', entry, not args.execute, start_positions, start_is_diff)
            report['motion_results'].append(result)
            if args.execute:
                report['feedback_checks']['assembly_entry_arrival'] = node.wait_assembly_arrival(entry)
                recheck_source()
                node._require_safe_diagnostics()
                node.require_held_gripper()
            else:
                start_positions, start_is_diff = result['final_joint_positions'], False
        if args.execute:
            node._motion_audit.attempt('assembly')
        report['motion_results'].append(node.plan_or_execute(
            'assembly', assembly, not args.execute, start_positions, start_is_diff))
        if args.execute:
            if assembly.get('final_approach'):
                report['feedback_checks']['assembly_arrival'] = node.wait_assembly_arrival(assembly)
            else:
                report['feedback_checks']['assembly_joint_arrival'] = node.wait_actual_joints(assembly)
                report['feedback_checks']['assembly_arrival'] = node.wait_actual_pose('assembly', assembly['pose'])
            report['feedback_checks']['assembly_gripper_after'] = node.require_held_gripper()
    report['status'] = 'executed' if args.execute else 'planned_only'


def main():
    args = parser().parse_args()
    try:
        if args.report is not None and args.report.exists():
            raise ValueError(f'报告路径已存在，拒绝覆盖：{args.report}')
        source, recognition = load_recognition(args.recognition)
        source_hash = sha256(source)
        spec = build_motion_spec(
            recognition,
            args.candidate_id,
            frame_id=args.base_frame,
            tip_link=args.tip_link,
            expected_parent_frame=args.expected_parent_frame,
            allow_nominal_urdf_tool=args.allow_nominal_urdf_tool,
            binding_file=args.binding,
        )
        assembly = None
        if args.assembly_target is not None:
            assembly = load_assembly_target(args.assembly_target, root=ROOT,
                                            frame_id=args.base_frame, joints=args.joints)
            assembly['planning_gripper_width_m'] = spec['gripper']['close_width_m']
            next(s for s in spec['stages'] if s['name'] == 'lift')['planning_gripper_width_m'] = spec['gripper']['close_width_m']
            spec['stages'].append(assembly)
            spec['warnings'].extend(assembly['warnings'])
        if args.execute and not spec['execution_allowed']:
            raise ValueError('真机执行阻断：'+', '.join(spec['execution_blockers']))
    except (OSError, ValueError) as exc:
        print(f'视觉抓取适配失败：{exc}', file=sys.stderr)
        return 2
    # Ctrl+C 先交给 Python 保存报告和关闭门控，最后再释放 ROS 上下文。
    rclpy.init(signal_handler_options=SignalHandlerOptions.SIGTERM)
    node = WireMoveGroup(args, spec)
    def recheck_source():
        if assembly is not None:
            current = load_assembly_target(assembly['source'], root=ROOT,
                                           frame_id=args.base_frame, joints=args.joints)
            if current['source_sha256'] != assembly['source_sha256']:
                raise ValueError('装配目标文件在会话期间改变')
        if sha256(source) != source_hash:
            raise ValueError('识别文件在会话期间改变')
        checked = build_motion_spec(recognition, args.candidate_id, frame_id=args.base_frame,
                                    tip_link=args.tip_link, expected_parent_frame=args.expected_parent_frame,
                                    allow_nominal_urdf_tool=args.allow_nominal_urdf_tool, binding_file=args.binding)
        if not checked['execution_allowed']:
            raise ValueError('执行前来源或视觉门控复验失败：'+', '.join(checked['execution_blockers']))
    report = {
        'schema_version': 2,
        'status': 'started',
        'execute_requested': bool(args.execute),
        'source_recognition': str(source),
        'motion_spec': spec,
        'feedback_checks': {},
        'motion_results': [],
        'gripper_results': [],
        'physical_motion_performed': False,
    }
    try:
        node.wait_ready(args.execute)
        report['feedback_checks']['parent_tip'] = node._require_parent_tip_identity()
        report['feedback_checks']['initial_joint_positions'] = {
            joint: node._joint_positions[joint] for joint in args.joints
        }
        if assembly is not None:
            report['feedback_checks']['assembly_runtime_initial'] = node.validate_assembly_goal(assembly)
            if assembly.get('final_approach'):
                entry = node.prepare_assembly_approach(assembly)
                spec['stages'].insert(-1, entry)
            print('已启用固定装配转移：抬升后到示教六轴目标，保持夹爪闭合；现场路径由操作者测试。')
        run_motion_sequence(node, args, spec, report, recheck_source)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        report['error'] = '操作者中断'
        exit_code = 130
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        print(f'视觉抓取适配失败：{exc}', file=sys.stderr)
        exit_code = 2
    else:
        exit_code = 0
        print(
            f"{'执行完成' if args.execute else '规划完成'}："
            f"candidate={spec['candidate_id']} stages={len(report['motion_results'])}"
        )
    finally:
        report.update(node._motion_audit.report())
        report['completed_at'] = datetime.now().astimezone().isoformat()
        report_path = args.report or (
            ROOT / 'output/wire_grasp' /
            f"moveit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{spec['candidate_id']}" /
            'report.json'
        )
        try:
            written = write_json_exclusive(report_path, report)
            print(f'报告：{written}')
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
