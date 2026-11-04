#!/usr/bin/env python3
"""将视觉基座坐标交给官方MoveIt规划；默认只规划，--execute才允许真机动作。"""

from __future__ import annotations

import argparse
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

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def parser():
    node = argparse.ArgumentParser(description=__doc__)
    node.add_argument('--recognition', type=Path, required=True)
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
    node.add_argument(
        '--allow-nominal-urdf-tool',
        action='store_true',
        help='用户显式授权按URDF名义工具参数执行；不表示该TCP已实测',
    )
    node.add_argument('--execute', action='store_true', help='执行预抓取、下降、闭合和抬升')
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


class WireMoveGroup(Node):
    def __init__(self, args, spec):
        super().__init__('wire_moveit_pose_adapter')
        self.args = args
        self.spec = spec
        self._joint_positions = {}
        self._tcp_pose = None
        self._joint_seen_at = 0.0
        self._tcp_seen_at = 0.0
        self._diagnostics = {}
        self._diagnostic_seen_at = {}
        self._move = ActionClient(self, MoveGroup, namespaced(args.namespace, 'move_action'))
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

    def _joint_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if positions:
            self._joint_positions.update(positions)
            self._joint_seen_at = time.monotonic()

    def _tcp_callback(self, message):
        self._tcp_pose = message.pose
        self._tcp_seen_at = time.monotonic()

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
        self._diagnostic_seen_at[name] = time.monotonic()

    def _spin_for(self, seconds, predicate=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate is not None and predicate():
                return True
        return predicate() if predicate is not None else True

    def _wait_joint_feedback(self):
        def complete():
            return all(joint in self._joint_positions for joint in self.args.joints)

        started = time.monotonic()
        initial_wait = min(2.0, self.args.feedback_timeout)
        if self._spin_for(initial_wait, complete):
            return True
        self.destroy_subscription(self._joint_sub)
        self._joint_sub = self.create_subscription(
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
        if not self._wait_joint_feedback():
            missing = [joint for joint in self.args.joints if joint not in self._joint_positions]
            raise RuntimeError(f'未收到完整真实关节反馈，缺少{missing}')
        if time.monotonic() - self._joint_seen_at > 2.5:
            raise ValueError('真实关节反馈陈旧')
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
        arm = self._diagnostics['arm_status'][0]['values']
        if arm.get('teach_status') == '1':
            raise ValueError('机械臂仍处于拖动/示教记录状态')
        if arm.get('ctrl_mode') not in (None, '0', '1'):
            raise ValueError(f"当前控制模式不适合MoveIt执行：ctrl_mode={arm.get('ctrl_mode')}")
        if arm.get('arm_status') not in (None, '0'):
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

        goal = MoveGroup.Goal()
        goal.request.group_name = args.group
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
        goal.planning_options.plan_only = plan_only
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        return goal

    def plan_or_execute(self, name, stage, plan_only, start_positions=None, start_is_diff=True):
        started = time.monotonic()
        goal = self._goal(stage, plan_only, start_positions, start_is_diff)
        send_future = self._move.send_goal_async(goal)
        if not self._spin_for(10.0, send_future.done):
            raise RuntimeError(f'{name}发送超时')
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f'{name}目标未被MoveGroup接受')
        result_future = handle.get_result_async()
        if not self._spin_for(self.args.goal_timeout, result_future.done):
            handle.cancel_goal_async()
            raise RuntimeError(f'{name}在{self.args.goal_timeout:.0f}s内未返回')
        wrapped = result_future.result()
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
        if plan_only:
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
        preclose = self._joint_positions.get('gripper')
        self.call_gate(True)
        try:
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
            send_future = self._gripper_action.send_goal_async(goal)
            if not self._spin_for(10.0, send_future.done):
                raise RuntimeError('夹爪轨迹发送超时')
            handle = send_future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError('夹爪轨迹未被官方控制器接受')
            result_future = handle.get_result_async()
            if not self._spin_for(15.0, result_future.done):
                handle.cancel_goal_async()
                raise RuntimeError('夹爪轨迹执行超时')
            wrapped = result_future.result()
            if (wrapped.status != GoalStatus.STATUS_SUCCEEDED
                    or wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
                raise RuntimeError(
                    f'夹爪轨迹执行失败：status={wrapped.status}, '
                    f'error_code={wrapped.result.error_code}'
                )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                actual = self._joint_positions.get('gripper')
                if actual is None:
                    continue
                if expect_open and actual >= width - 0.01:
                    break
                if not expect_open and actual <= 0.01:
                    break
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
            try:
                self.call_gate(False)
            except Exception as exc:
                self.get_logger().error(f'关闭控制门控失败：{exc}')


def main():
    args = parser().parse_args()
    try:
        if args.report is not None and args.report.exists():
            raise ValueError(f'报告路径已存在，拒绝覆盖：{args.report}')
        source, recognition = load_recognition(args.recognition)
        spec = build_motion_spec(
            recognition,
            args.candidate_id,
            frame_id=args.base_frame,
            tip_link=args.tip_link,
            expected_parent_frame=args.expected_parent_frame,
            allow_nominal_urdf_tool=args.allow_nominal_urdf_tool,
        )
    except (OSError, ValueError) as exc:
        print(f'视觉抓取适配失败：{exc}', file=sys.stderr)
        return 2
    rclpy.init()
    node = WireMoveGroup(args, spec)
    report = {
        'schema_version': 1,
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
        if args.execute:
            report['gripper_results'].append(
                node.command_gripper(spec['gripper']['open_width_m'],
                                     spec['gripper']['force_n'], expect_open=True)
            )
        start_positions = None
        start_is_diff = True
        for name in ('pregrasp', 'grasp'):
            stage = next(item for item in spec['stages'] if item['name'] == name)
            result = node.plan_or_execute(
                name, stage, not args.execute, start_positions, start_is_diff
            )
            report['motion_results'].append(result)
            if args.execute:
                report['feedback_checks'][f'{name}_arrival'] = node.wait_actual_pose(name, stage['pose'])
            else:
                start_positions = result['final_joint_positions']
                start_is_diff = False
        if args.execute:
            report['physical_motion_performed'] = True
            report['gripper_results'].append(
                node.command_gripper(spec['gripper']['close_width_m'], spec['gripper']['force_n'])
            )
            lift = next(item for item in spec['stages'] if item['name'] == 'lift')
            report['motion_results'].append(node.plan_or_execute('lift', lift, False))
            report['feedback_checks']['lift_arrival'] = node.wait_actual_pose('lift', lift['pose'])
            report['status'] = 'executed'
        else:
            report['status'] = 'planned_only'
        report['completed_at'] = datetime.now().astimezone().isoformat()
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report['completed_at'] = datetime.now().astimezone().isoformat()
        print(f'视觉抓取适配失败：{exc}', file=sys.stderr)
        exit_code = 2
    else:
        exit_code = 0
        print(
            f"{'执行完成' if args.execute else '规划完成'}："
            f"candidate={spec['candidate_id']} stages={len(report['motion_results'])}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    report_path = args.report or (
        ROOT / 'output/wire_grasp' /
        f"moveit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{spec['candidate_id']}" /
        'report.json'
    )
    written = write_json_exclusive(report_path, report)
    print(f'报告：{written}')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
