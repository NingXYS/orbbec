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
