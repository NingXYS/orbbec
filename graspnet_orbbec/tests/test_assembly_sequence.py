"""使用假节点检查生产流程的装配触发与失败中断，不调用硬件。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip('rclpy')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('wire_runner_test', ROOT/'scripts/run_moveit_wire_grasp.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeNode:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail
        self._motion_audit = SimpleNamespace(attempt=lambda name: None)

    def command_gripper(self, width, effort, expect_open=False):
        self.events.append('open' if expect_open else 'close')
        return {}

    def plan_or_execute(self, name, stage, plan_only, start=None, is_diff=True):
        self.events.append(name)
        if self.fail == name:
            raise RuntimeError('stage failed')
        if plan_only and name != 'pregrasp':
            assert start == [1.0]*6 and is_diff is False
        return {'final_joint_positions': [1.0]*6}

    def wait_actual_pose(self, name, pose):
        if self.fail == name+'_arrival':
            raise RuntimeError('arrival failed')
        return {}

    def _require_safe_diagnostics(self):
        pass

    def validate_assembly_goal(self, stage):
        return {}

    def require_held_gripper(self):
        if self.fail == 'holding':
            raise ValueError('gripper not closed')
        return {}

    def wait_actual_joints(self, stage):
        if self.fail == 'assembly_joints':
            raise RuntimeError('joint arrival failed')
        return {}

    def wait_assembly_arrival(self, stage):
        if self.fail == stage['name']+'_arrival':
            raise RuntimeError('arrival failed')
        return {'stable_duration_s': .5}


def run(execute=True, assembly=True, fail=None):
    node = FakeNode(fail)
    names = ['pregrasp', 'grasp', 'lift'] + (['assembly'] if assembly else [])
    motion = {'stages': [{'name': n, 'pose': {}} for n in names],
              'gripper': {'open_width_m': 0.1, 'close_width_m': 0., 'force_n': 1.}}
    report = {'gripper_results': [], 'motion_results': [], 'feedback_checks': {}}
    return node, motion, report, SimpleNamespace(execute=execute)


def test_transfer_after_lift_and_never_reopens():
    node, motion, report, args = run()
    runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert node.events == ['open', 'pregrasp', 'grasp', 'close', 'lift', 'assembly']
    assert report['status'] == 'executed'
    assert 'assembly_joint_arrival' in report['feedback_checks']
    assert 'assembly_arrival' in report['feedback_checks']


@pytest.mark.parametrize('fail', ['grasp', 'lift', 'lift_arrival', 'holding'])
def test_failure_before_transfer_never_sends_assembly(fail):
    node, motion, report, args = run(fail=fail)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert 'assembly' not in node.events
    assert node.events.count('open') == 1


def test_plan_only_chains_lift_to_assembly_without_gripper_commands():
    node, motion, report, args = run(execute=False)
    runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert node.events == ['pregrasp', 'grasp', 'lift', 'assembly']
    assert report['status'] == 'planned_only'


def test_changed_source_after_lift_stops_transfer():
    node, motion, report, args = run()
    def recheck():
        if node.events and node.events[-1] == 'lift':
            raise ValueError('target changed')
    with pytest.raises(ValueError, match='target changed'):
        runner.run_motion_sequence(node, args, motion, report, recheck)
    assert 'assembly' not in node.events


def test_joint_arrival_failure_does_not_report_success_or_release_gripper():
    node, motion, report, args = run(fail='assembly_joints')
    with pytest.raises(RuntimeError, match='joint arrival'):
        runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert report.get('status') != 'executed'
    assert node.events[-1] == 'assembly' and node.events.count('open') == 1


def test_no_option_preserves_existing_flow():
    node, motion, report, args = run(execute=False, assembly=False)
    runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert node.events == ['pregrasp', 'grasp']
    node, motion, report, args = run(assembly=False)
    runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert node.events == ['open', 'pregrasp', 'grasp', 'close', 'lift']


def test_assembly_goal_uses_joint_constraints_and_closed_gripper_for_preview():
    import numpy as np
    from arm.moveit_wire_adapter import transform_to_pose
    stage = {'kind': 'joint', 'pose': transform_to_pose(np.eye(4)),
             'joint_names': runner.DEFAULT_JOINTS, 'joint_positions_rad': [0.2]*6,
             'joint_goal_tolerance_rad': .001, 'planning_gripper_width_m': 0.}
    args = runner.parser().parse_args(['--recognition', 'unused.json'])
    node = SimpleNamespace(args=args, spec={'frame_id': 'base_link', 'tip_link': 'tcp_link'})
    goal = runner.WireMoveGroup._goal(node, stage, True, [0.1]*6, False)
    assert goal.planning_options.plan_only
    constraints = goal.request.goal_constraints[0]
    assert len(constraints.joint_constraints) == 6
    assert not constraints.position_constraints and not constraints.orientation_constraints
    assert goal.request.start_state.joint_state.name[-1] == 'gripper'
    assert goal.request.start_state.joint_state.position[-1] == 0.


@pytest.mark.parametrize('execute', [False, True])
def test_y_entry_precedes_cartesian_final_segment(execute):
    node, motion, report, args = run(execute=execute)
    motion['stages'].insert(-1, {'name': 'assembly_entry', 'pose': {}})
    motion['stages'][-1]['final_approach'] = {'direction': -1, 'distance_m': .03}
    runner.run_motion_sequence(node, args, motion, report, lambda: None)
    moves = [e for e in node.events if e not in ('open','close')]
    assert moves == ['pregrasp','grasp','lift','assembly_entry','assembly']


def test_failed_y_entry_never_sends_final_line():
    node, motion, report, args = run(fail='assembly_entry_arrival')
    motion['stages'].insert(-1, {'name': 'assembly_entry', 'pose': {}})
    motion['stages'][-1]['final_approach'] = {'direction': -1, 'distance_m': .03}
    with pytest.raises(RuntimeError):
        runner.run_motion_sequence(node, args, motion, report, lambda: None)
    assert node.events[-1] == 'assembly_entry'


def cartesian_solution():
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    solution = RobotTrajectory()
    solution.joint_trajectory.joint_names = runner.DEFAULT_JOINTS
    for i in range(3):
        p = JointTrajectoryPoint()
        p.positions = [i*.02]*6
        p.velocities = [.01]*6
        p.accelerations = [.002]*6
        p.time_from_start.sec = i
        solution.joint_trajectory.points.append(p)
    return solution, {'joint_names': runner.DEFAULT_JOINTS, 'joint_positions_rad': [0.]*6}


def test_reversed_cartesian_times_and_derivatives_preserve_fixed_endpoint():
    original, stage = cartesian_solution()
    reversed_path = runner.reverse_cartesian_trajectory(original, stage).joint_trajectory
    assert list(reversed_path.points[-1].positions) == [0.]*6
    assert [p.time_from_start.sec for p in reversed_path.points] == [0,1,2]
    assert list(reversed_path.points[1].velocities) == [-.01]*6
    assert list(reversed_path.points[1].accelerations) == [.002]*6
    assert list(original.joint_trajectory.points[0].positions) == [0.]*6


@pytest.mark.parametrize('fault', ['time', 'nan', 'jump', 'endpoint'])
def test_bad_cartesian_solution_is_never_used(fault):
    original, stage = cartesian_solution()
    if fault == 'time':
        original.joint_trajectory.points[1].time_from_start.sec = 0
    elif fault == 'nan':
        original.joint_trajectory.points[1].positions[0] = float('nan')
    elif fault == 'jump':
        original.joint_trajectory.points[1].positions[0] = 1.
    else:
        stage['joint_positions_rad'] = [.01]*6
    with pytest.raises(ValueError):
        runner.reverse_cartesian_trajectory(original, stage)


def test_assembly_arrival_requires_half_second_of_simultaneous_stability(monkeypatch):
    import numpy as np
    from arm.moveit_wire_adapter import transform_to_pose
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    args = SimpleNamespace(arrival_timeout=20., arrival_joint_tolerance_rad=.02)
    node = SimpleNamespace(args=args, _joint_positions=dict(zip(runner.DEFAULT_JOINTS,[0.]*6)),
                           _tcp_pose=runner.pose_matrix(transform_to_pose(np.eye(4))), _tcp_seen_at=0.,
                           _require_joint_feedback=lambda names: None)
    outcomes=[]
    def spin(seconds, predicate):
        for now in [0., .2, .49, .51]:
            clock[0]=now;node._tcp_seen_at=now
            outcomes.append(predicate())
        return outcomes[-1]
    node._spin_for=spin
    stage={'joint_names':runner.DEFAULT_JOINTS,'joint_positions_rad':[0.]*6,'T_base_flange':np.eye(4).tolist()}
    result=runner.WireMoveGroup.wait_assembly_arrival(node,stage)
    assert outcomes == [False,False,False,True]
    assert result['stable_duration_s'] == .5


def test_delayed_volatile_driver_keeps_matching_subscription():
    from unittest.mock import Mock
    from rclpy.qos import DurabilityPolicy, QoSProfile, QoSCompatibility, qos_check_compatible
    live = QoSProfile(depth=10, durability=DurabilityPolicy.VOLATILE)
    cached = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node = SimpleNamespace(args=runner.parser().parse_args(['--recognition','unused']),
                           _joint_qos=cached, _joint_sub=live, _joint_callback=Mock(),
                           create_subscription=Mock(), _spin_for=Mock(side_effect=[False, True]))
    assert runner.WireMoveGroup._wait_joint_feedback(node)
    assert node._joint_sub is live
    assert qos_check_compatible(live, node._joint_sub)[0] != QoSCompatibility.ERROR
    assert qos_check_compatible(live, cached)[0] == QoSCompatibility.ERROR


@pytest.mark.parametrize('age', [60, -1])
def test_old_and_future_message_cannot_refresh_feedback(age):
    from arm.feedback_safety import MotionAudit
    from rclpy.time import Time
    node = SimpleNamespace(args=SimpleNamespace(joints=runner.DEFAULT_JOINTS),
                           get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=100)),
                           _joint_positions={}, _joint_seen_at={}, _motion_audit=MotionAudit())
    node._sample_time = lambda m: runner.WireMoveGroup._sample_time(node, m)
    message = runner.JointState()
    message.header.stamp = Time(seconds=100-age).to_msg()
    message.name, message.position = runner.DEFAULT_JOINTS, [0.]*6
    runner.WireMoveGroup._joint_callback(node, message)
    with pytest.raises(ValueError, match='陈旧'):
        runner.WireMoveGroup._require_joint_feedback(node)
    pose = runner.PoseStamped()
    pose.header = message.header
    runner.WireMoveGroup._tcp_callback(node, pose)
    assert runner.time.monotonic() - node._tcp_seen_at > 2.5


@pytest.mark.parametrize('fault', ['missing_motor', 'duplicate_motor', 'missing_mode', 'missing_teach', 'missing_arm_state'])
def test_diagnostic_contract_covers_each_motor_and_arm_mode(fault):
    def ok(name, **values):
        return dict(name=name, level=0, values=values, message='ok')
    arm = ok('arm', arm_status='0', ctrl_mode='1', teach_status='2')
    motors = [ok(name, enabled='true') for name in runner.DEFAULT_JOINTS]
    status = dict(arm_status=[arm], motor_status=motors, gripper_status=[ok('gripper',driver_enable_status='true')])
    node = SimpleNamespace(_diagnostics=status, _diagnostic_seen_at=dict.fromkeys(status, runner.time.monotonic()))
    runner.WireMoveGroup._require_safe_diagnostics(node)
    if fault == 'missing_motor':
        motors.pop()
    elif fault == 'duplicate_motor':
        motors[-1] = motors[0]
    else:
        del arm['values'][dict(missing_mode='ctrl_mode', missing_teach='teach_status', missing_arm_state='arm_status')[fault]]
    with pytest.raises(ValueError):
        runner.WireMoveGroup._require_safe_diagnostics(node)


def test_gripper_gate_close_failure_propagates(monkeypatch):
    from unittest.mock import Mock
    node = Mock()
    node.args.joints = runner.DEFAULT_JOINTS
    node._joint_positions = {'gripper': .1}
    node._joint_seen_at = {'gripper': runner.time.monotonic()}
    node._action_result.return_value = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))
    node.call_gate.side_effect = [None, RuntimeError('门控未关')]
    monkeypatch.setattr(runner.rclpy, 'spin_once', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='门控未关'):
        runner.WireMoveGroup.command_gripper(node, .1, 1., expect_open=True)
    assert [c.args[0] for c in node.call_gate.call_args_list] == [True, False]


def test_new_fault_blocks_submission_and_cancels_active_motion():
    from unittest.mock import Mock
    node = Mock()
    node._require_safe_diagnostics.side_effect = ValueError('电机故障')
    with pytest.raises(ValueError, match='电机故障'):
        runner.WireMoveGroup.plan_or_execute(node, 'grasp', {'name':'grasp'}, False)
    node._move.send_goal_async.assert_not_called()
    node._spin_for.side_effect = lambda timeout, predicate: predicate()
    handle = Mock()
    with pytest.raises(ValueError, match='电机故障'):
        runner.WireMoveGroup._action_result(node, handle, 1., 'grasp', monitor=True)
    handle.cancel_goal_async.assert_called_once()


@pytest.mark.parametrize('name', ['pregrasp', 'grasp'])
def test_grasp_planner_and_open_gripper_match_execution(name):
    import numpy as np
    from arm.moveit_wire_adapter import transform_to_pose
    node = SimpleNamespace(args=runner.parser().parse_args(['--recognition','unused']),
                           spec={'frame_id':'base_link','tip_link':'tcp_link','gripper':{'open_width_m':.1}})
    stage = {'name':name, 'pose':transform_to_pose(np.eye(4))}
    start = None if name == 'pregrasp' else [0.]*6
    goal = runner.WireMoveGroup._goal(node, stage, True, start, start is None)
    assert goal.request.start_state.joint_state.name[-1] == 'gripper'
    assert goal.request.start_state.joint_state.position[-1] == .1
    if name == 'grasp':
        assert goal.request.pipeline_id == 'pilz_industrial_motion_planner' and goal.request.planner_id == 'LIN'
        assert runner.WireMoveGroup._goal(node, stage, False).request.planner_id == 'LIN'


def test_speed_limit_covers_interior_spline_peak():
    import numpy as np
    from arm.assembly_target import limit_flange_speed
    path, _ = cartesian_solution()
    joints = path.joint_trajectory
    del joints.points[-1]
    for point in joints.points:
        point.velocities = point.accelerations = [0.]*6
    radii = np.array([.5, 0, 0, 0, 0, 0])
    # 五次平滑插值中点速度是区间均速的1.875倍，两个端点速度均为零。
    assert limit_flange_speed(joints, radii, .01) <= .01
    duration = joints.points[-1].time_from_start
    seconds = duration.sec + duration.nanosec * 1e-9
    assert 1.875 * .02 * .5 / seconds <= .01
