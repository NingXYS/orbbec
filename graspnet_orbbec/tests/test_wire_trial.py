"""离线运动学、控制前置条件和发送白名单；不连接任何CAN设备。"""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm.wire_geometry import flange_transform
from arm.wire_trial_plan import fk, ik, matrix_pose, sample_segment, prepare_start_segment
from arm.wire_trial_runtime import (pose_error, require_expected_start, require_safe_feedback,
                                    require_known_software_estop, validate_tx)
from arm.wire_trial_runtime import recovery_observation
from arm.wire_trial_runtime import validate_start_preparation, joint_start_packets


MDH = [[.123, 0, 0, np.pi], [0, 0, np.pi/2, .13578661580515886],
       [0, .28503, 0, 2.8380798966679794], [0, .27364, 0, .08063421144213803],
       [0, .07466, -np.pi/2, np.pi/2], [.03526, 0, np.pi/2, 0]]
LIMITS = [[-2.617994, 2.617994], [0, 3.141593], [-2.96706, 0],
          [-1.553344, 1.553344], [-1.553344, 1.553344], [-2.96706, 2.96706]]


def state():
    return {'fresh': True, 'flange_pose_m_rad': [.3, 0, .2, 0, 0, 0],
            'joint_angles_rad': [0]*6,
            'arm_status': {'ctrl_mode': 1, 'arm_status': 0, 'err_status': {'error': False}},
            'driver_states': [{'foc_status': {'driver_enable_status': True, 'collision_status': False}} for _ in range(6)],
            'gripper_status': {'mode': 'width', 'foc_status': {'driver_enable_status': True,
                                                           'homing_status': False, 'driver_error_status': False}}}


def frame(identifier, data):
    return SimpleNamespace(arbitration_id=identifier, data=bytes(data), is_extended_id=False)


@pytest.mark.parametrize('identifier,data', [
    (0x471, [0]*8), (0x150, [2, 0, 0, 0, 0, 0, 0, 0]),
    (0x151, [1, 2, 100, 0, 0, 0, 0, 0]), (0x151, [1, 1, 5, 0, 0, 0, 0, 0]),
    (0x151, [1, 2, 5, 0xAD, 0, 0, 0, 0]),
    (0x159, struct.pack('>iHBB', 0, 1001, 1, 0)),
    (0x159, struct.pack('>iHBB', 25001, 1000, 1, 0)),
    (0x159, struct.pack('>iHBB', 0, 1000, 1, 0xAE))])
def test_motor_enable_reset_fast_motion_and_gripper_zeroing_are_blocked(identifier, data):
    with pytest.raises(ValueError):
        validate_tx(frame(identifier, data))


@pytest.mark.parametrize('identifier,data', [
    (0x150, [1, 0, 0, 0, 0, 0, 0, 0]), (0x151, [1, 2, 5, 0, 0, 0, 0, 0]),
    (0x151, [1, 255, 5, 0, 0, 0, 0, 0]),
    (0x159, struct.pack('>iHBB', 25000, 1000, 1, 0))])
def test_exact_trial_commands_and_damped_stop_are_allowed(identifier, data):
    validate_tx(frame(identifier, data))


def test_teaching_fault_stale_feedback_and_disabled_joints_prevent_motion():
    s = state()
    require_safe_feedback(s)
    s['arm_status']['ctrl_mode'] = 2
    with pytest.raises(ValueError, match='示教'):
        require_safe_feedback(s)
    require_safe_feedback(s, allowed_ctrl_modes=(0, 1, 2))
    s['arm_status']['teach_status'] = 1
    with pytest.raises(ValueError, match='仍活动'):
        require_safe_feedback(s, allowed_ctrl_modes=(0, 1, 2))
    s['arm_status']['teach_status'] = 2
    s['arm_status']['ctrl_mode'] = 4
    with pytest.raises(ValueError):
        require_safe_feedback(s, allowed_ctrl_modes=(0, 1, 2))
    s = state()
    s['driver_states'][2]['foc_status']['driver_enable_status'] = False
    with pytest.raises(ValueError, match='不会自动使能'):
        require_safe_feedback(s)
    s = state()
    s['arm_status']['err_status']['error'] = True
    with pytest.raises(ValueError, match='故障'):
        require_safe_feedback(s)
    with pytest.raises(ValueError, match='陈旧'):
        require_safe_feedback({'fresh': False})


def test_start_pose_or_joint_change_requires_replanning():
    s = state()
    require_expected_start(s, [.3, 0, .2, 0, 0, 0], [0]*6)
    with pytest.raises(ValueError, match='起始位姿'):
        require_expected_start(s, [.31, 0, .2, 0, 0, 0])
    with pytest.raises(ValueError, match='起始关节'):
        require_expected_start(s, [.3, 0, .2, 0, 0, 0], [.1]*6)


def hold_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import enable_wire_current_hold
    return enable_wire_current_hold


def test_restart_hold_rejects_partial_enable_active_teaching_or_fault(monkeypatch):
    m = hold_module(monkeypatch)
    s = state()
    s['arm_status'].update(ctrl_mode=0, teach_status=0)
    for d in s['driver_states']:
        d['foc_status']['driver_enable_status'] = False
    m.boot_feedback(s)
    s['driver_states'][2]['foc_status']['driver_enable_status'] = True
    with pytest.raises(ValueError, match='均未使能'):
        m.boot_feedback(s)
    s['driver_states'][2]['foc_status']['driver_enable_status'] = False
    s['arm_status']['teach_status'] = 1
    with pytest.raises(ValueError, match='仍活动'):
        m.boot_feedback(s)
    s['arm_status'].update(teach_status=0, arm_status=5)
    with pytest.raises(ValueError, match='故障'):
        m.boot_feedback(s)


def test_restart_hold_requires_real_enabled_feedback_and_unchanged_pose(monkeypatch):
    import copy
    m = hold_module(monkeypatch)
    initial = state()
    initial['gripper_status']['value'] = .101
    actual = copy.deepcopy(initial)
    actual['gripper_status']['value'] = .100
    assert m.enabled_hold_observation(initial, actual, .100)['ready']
    actual['driver_states'][2]['foc_status']['driver_enable_status'] = False
    assert not m.enabled_hold_observation(initial, actual, .100)['ready']
    actual['arm_status']['arm_status'] = 6
    assert not m.enabled_hold_observation(initial, actual, .100)['ready']
    actual['arm_status']['arm_status'] = 5
    with pytest.raises(ValueError, match='故障'):
        m.enabled_hold_observation(initial, actual, .100)
    actual['arm_status']['arm_status'] = 0
    actual['flange_pose_m_rad'][0] += .003
    with pytest.raises(ValueError, match='变化超限'):
        m.enabled_hold_observation(initial, actual, .100)


def test_restart_hold_rejects_zero_target_reset_or_extra_enable_packets(monkeypatch):
    m = hold_module(monkeypatch)
    q, width = [.1, .2, -.3, .4, -.5, .6], .100
    values = [round(j*180/np.pi*1000) for j in q]
    joint = [struct.pack('>ii', *values[i:i+2]) for i in (0, 2, 4)]
    ids = [0x151, 0x155, 0x156, 0x157, 0x159, 0x471, 0x155, 0x156, 0x157]
    data = [bytes([1, 1, 5, 0, 0, 0, 0, 0]), *joint,
            struct.pack('>iHBB', 100000, 1000, 1, 0), bytes([7, 2, 0, 0, 0, 0, 0, 0]), *joint]
    packets = list(zip(ids, data))
    m.validate_hold_packets(packets, q, width)
    for bad in (packets+packets[5:6], packets[:1]+[(0x155, bytes(8))]+packets[2:],
                [(0x150, bytes([2, 0, 0, 0, 0, 0, 0, 0]))]+packets[1:]):
        with pytest.raises(ValueError, match='拒绝执行'):
            m.validate_hold_packets(bad, q, width)


def test_known_estop_recovery_requires_ended_teaching_and_healthy_drivers():
    s = state()
    s['arm_status'].update(ctrl_mode=0, arm_status=1, teach_status=2)
    evidence = {'status': 'failed', 'damped_stop_sent': True,
                'application_frames_sent': [{'id': '0x150', 'data_hex': '0100000000000000'}]}
    require_known_software_estop(s, evidence)
    with pytest.raises(ValueError):
        require_known_software_estop(s, {})
    s['arm_status']['teach_status'] = 1
    with pytest.raises(ValueError):
        require_known_software_estop(s, evidence)
    s['arm_status']['teach_status'] = 2
    s['driver_states'][1]['foc_status']['collision_status'] = True
    with pytest.raises(ValueError, match='故障'):
        require_known_software_estop(s, evidence)


def test_recovery_does_not_treat_communication_initialization_as_real_motion():
    initial = state()
    initial['flange_pose_m_rad'] = [.080011, -.002347, .137926, -2.1188646, -.0376817, -1.6454964]
    transient = state()
    transient['arm_status']['arm_status'] = 5
    transient['flange_pose_m_rad'] = [.096678, 0, .214351, -1.4847341, 0, -1.5707963]
    assert not recovery_observation(initial, transient, 100)['ready']
    transient['arm_status']['arm_status'] = 1
    assert not recovery_observation(initial, transient, 100)['ready']


def test_recovery_needs_all_post_reset_groups_and_does_not_require_enable():
    initial = state()
    observed = state()
    assert not recovery_observation(initial, observed, 100)['ready']
    observed['feedback_stamps_s'] = {f'0x{i:03X}': 101 for i in (*range(0x2A1, 0x2A9), *range(0x261, 0x267))}
    for d in observed['driver_states']:
        d['foc_status']['driver_enable_status'] = False
    assert recovery_observation(initial, observed, 100)['ready']
    with pytest.raises(ValueError, match='不会自动使能'):
        require_safe_feedback(observed)
    observed['flange_pose_m_rad'][2] += .01
    with pytest.raises(ValueError, match='有效反馈'):
        recovery_observation(initial, observed, 100)


def test_pose_roundtrip_and_wraparound_do_not_command_a_full_turn():
    for pose in ([.3, .1, .2, -.9, .4, 3.1], [.3, 0, .2, .3, np.pi/2, -.4]):
        matrix = flange_transform(pose)
        assert flange_transform(matrix_pose(matrix)) == pytest.approx(matrix, abs=1e-7)
    _, angle = pose_error([.3, 0, .2, 0, 0, 3.14], [.3, 0, .2, 0, 0, -3.14])
    assert angle < .2
    samples = list(sample_segment(flange_transform([.3, 0, .2, 0, 0, 3.14]),
                                  flange_transform([.32, 0, .2, 0, 0, -3.14])))
    assert len(samples) == 3


def test_local_ik_solves_reachable_pose_and_rejects_unreachable_goal():
    q = np.array([.2, 1.1, -1.4, .3, .6, -.2])
    target = fk(MDH, q)
    solution, error = ik(MDH, LIMITS, target, q+.015)
    assert error['position_error_m'] < 1e-6
    assert fk(MDH, solution) == pytest.approx(target, abs=1e-5)
    bad = target.copy()
    bad[:3, 3] = [5, 5, 5]
    with pytest.raises(ValueError, match='IK未通过'):
        ik(MDH, LIMITS, bad, q)


def preparation_fixture():
    q = [.00932, np.deg2rad(-2.291), np.deg2rad(2.586), .62437, -.01063, .09016]
    initial = {'joint_angles_rad': q, 'joint_limits_rad': LIMITS, 'nominal_mdh': MDH,
               'flange_pose_m_rad': matrix_pose(fk(MDH, q))}
    return {**initial, 'initial_joint_angles_rad': q, 'stages': [prepare_start_segment(initial)]}


def test_preparation_is_monotonic_small_joint_targets_with_verified_nominal_fk():
    plan = preparation_fixture()
    validate_start_preparation(plan)
    stage = plan['stages'][0]
    q = np.array([s['joint_angles_rad'] for s in stage['samples']])
    assert len(q) == 4
    assert np.rad2deg(q[-1, 1:3]) == pytest.approx([.5, -.5])
    assert np.all(np.diff(q[:, 1]) > 0) and np.all(np.diff(q[:, 2]) < 0)
    assert np.all(q[:, [0, 3, 4, 5]] == np.array(plan['initial_joint_angles_rad'])[[0, 3, 4, 5]])
    tip_z = [(fk(MDH, x)@np.array([0, 0, .1425, 1]))[2] for x in q]
    assert np.all(np.diff(tip_z) > 0)
    for sample in stage['samples']:
        assert sample['max_joint_step_deg'] <= 1


@pytest.mark.parametrize('change', ['other_joint', 'wrong_direction', 'wrong_pose', 'extra_step', 'wide_initial'])
def test_preparation_cannot_be_reused_as_unrestricted_joint_motion(change):
    plan = preparation_fixture()
    if change == 'other_joint':
        plan['stages'][0]['samples'][0]['joint_angles_rad'][0] += .01
    elif change == 'wrong_direction':
        plan['stages'][0]['samples'][0]['joint_angles_rad'][1] -= .01
    elif change == 'wrong_pose':
        plan['stages'][0]['samples'][0]['flange_pose_m_rad'][0] += .003
    elif change == 'extra_step':
        plan['stages'][0]['samples'].append(plan['stages'][0]['samples'][-1])
    else:
        plan['initial_joint_angles_rad'][1] = -.2
    with pytest.raises(ValueError):
        validate_start_preparation(plan)


def test_preparation_encoding_is_separate_from_the_linear_grasp_whitelist():
    packets = joint_start_packets([.1, .2, -.3, .4, -.5, .6], include_mode=True)
    assert [i for i, _ in packets] == [0x151, 0x155, 0x156, 0x157]
    assert packets[0][1] == bytes([1, 1, 5, 0, 0, 0, 0, 0])
    for identifier, payload in packets:
        with pytest.raises(ValueError):
            validate_tx(frame(identifier, payload))


def test_old_paths_do_not_bypass_control_incident_marker(monkeypatch, tmp_path):
    hold_module(monkeypatch)
    from execute_wire_trial import require_post_restart_audit
    with pytest.raises(ValueError, match='异常锁'):
        require_post_restart_audit({'stages': []}, tmp_path/'block.json')


def test_timeout_preserves_last_observation_for_logging_after_stop(monkeypatch):
    hold_module(monkeypatch)
    import execute_wire_trial as executor
    clock = iter([0., .1, 6.1])
    monkeypatch.setattr(executor.time, 'monotonic', lambda: next(clock))
    monitor = SimpleNamespace(pump=lambda: None, snapshot=state)
    target = [.31, 0, .2, 0, 0, 0]
    with pytest.raises(ValueError, match='6秒') as failure:
        executor.await_pose(monitor, target, np.zeros(6), lambda value: None)
    diagnostic = failure.value.diagnostic
    assert diagnostic['last_feedback_before_stop']['flange_pose_m_rad'] == [.3, 0, .2, 0, 0, 0]
    assert diagnostic['target_pose_m_rad'] == target
    assert diagnostic['position_error_m'] == pytest.approx(.01)


@pytest.mark.parametrize('case', ['stale_after_move', 'teaching_before_start', 'log_failure'])
def test_executor_faults_stop_before_any_further_motion(monkeypatch, tmp_path, case):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('_test_execute_wire_trial', root/'scripts/execute_wire_trial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    class Bus:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, 'can', SimpleNamespace(Bus=lambda **kwargs: Bus()))
    class Robot:
        def __init__(self):
            self._ctx = SimpleNamespace(comm=SimpleNamespace(send=lambda message: calls.append(('tx', message.arbitration_id))))
        def init_effector(self, kind):
            assert kind == 'agx_gripper'
            return SimpleNamespace(move_gripper_m=lambda *args, **kwargs: calls.append(('gripper', args)))
        def connect(self, start_read_thread):
            assert start_read_thread is False
            calls.append(('connect',))
        def set_speed_percent(self, value):
            assert value == 5
            self._ctx.comm.send(frame(0x151, [1, 255, 5, 0, 0, 0, 0, 0]))
        def set_motion_mode(self, mode):
            assert mode == 'l'
            self._ctx.comm.send(frame(0x151, [1, 2, 5, 0, 0, 0, 0, 0]))
        def set_auto_set_motion_mode_enabled(self, value):
            assert value is False
        def move_l(self, pose):
            calls.append(('move_l', pose))
        def electronic_emergency_stop(self):
            self._ctx.comm.send(frame(0x150, [1, 0, 0, 0, 0, 0, 0, 0]))
        def disconnect(self):
            calls.append(('disconnect',))
    robot = Robot()
    decoder_module = SimpleNamespace(make_decoder=lambda *args: (robot, 'test_only'),
                                     FreshPoseDecoder=lambda *args: None)
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location',
                        lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda obj: None)))
    monkeypatch.setattr(module.importlib.util, 'module_from_spec', lambda spec: decoder_module)
    class Monitor:
        def __init__(self, *args):
            pass
        def pump(self, *args):
            pass
        def snapshot(self):
            s = state()
            if case == 'teaching_before_start':
                s['arm_status']['ctrl_mode'] = 2
            if case == 'stale_after_move' and any(c[0] == 'move_l' for c in calls):
                return {'fresh': False}
            return s
    monkeypatch.setattr(module, 'FeedbackMonitor', Monitor)
    plan = {'initial_flange_pose_m_rad': [.3, 0, .2, 0, 0, 0], 'initial_joint_angles_rad': [0]*6,
            'stages': [{'name': 'raise', 'samples': [{'flange_pose_m_rad': [.3, 0, .208, 0, 0, 0],
                                                      'joint_angles_rad': [0]*6}]}]}
    binding = SimpleNamespace(root=tmp_path, settings={'channel': 'fake_only'},
                              session={'robot': {'firmware_version': 'test_only'}})
    def emit(event):
        if case == 'log_failure' and event['event'] == 'tx':
            raise OSError('test_only_disk_failure')
    with pytest.raises((ValueError, OSError)):
        module.execute(plan, binding, 'hover', emit)
    assert calls[-1] == ('disconnect',)
    if case == 'teaching_before_start':
        assert not any(c[0] in ('connect', 'tx', 'move_l') for c in calls)
    else:
        assert ('tx', 0x150) in calls
        assert sum(c[0] == 'move_l' for c in calls) <= 1
