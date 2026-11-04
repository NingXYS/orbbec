"""视觉基座目标到官方MoveIt位姿适配的纯数据测试；不连接机械臂。"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.moveit_wire_adapter import (
    NOMINAL_TOOL_BLOCKER,
    build_motion_spec,
    matrix_to_quaternion,
    transform_to_pose,
)


def recognition(blockers=None):
    flange_grasp = np.eye(4)
    flange_grasp[:3, 3] = [0.40, -0.03, 0.1715]
    flange_pregrasp = flange_grasp.copy()
    flange_pregrasp[2, 3] -= 0.03
    base_grasp = flange_grasp.copy()
    base_grasp[2, 3] -= 0.1425
    blockers = list(blockers or [])
    plan = {
        'candidate_id': 'A',
        'status': 'base_frame_preview_ready',
        'boundary_confirmed': True,
        'surface_projection_only': False,
        'tcp_compensation_applied': True,
        'offset_mm': 14,
        'T_base_flange_grasp': flange_grasp.tolist(),
        'T_base_flange_pregrasp': flange_pregrasp.tolist(),
        'T_base_grasp': base_grasp.tolist(),
        'grasp_base_m': base_grasp[:3, 3].tolist(),
        'readiness_blockers': blockers,
    }
    return {
        'grasp_plans': [plan],
        'grasp_policy': {
            'capture_snapshot': '/test/snapshot',
            'support_plane': {'normal_camera': [0, 0, -1]},
            'center_height_over_board_m': 0.002,
            'readiness_blockers': blockers,
            'handeye': {
                'camera_serial': 'TEST_ONLY',
                'absolute_accuracy_verified': False,
            },
        },
    }


def test_nominal_tool_can_plan_but_cannot_execute():
    spec = build_motion_spec(recognition([NOMINAL_TOOL_BLOCKER]))
    assert spec['status'] == 'plan_only'
    assert spec['planning_allowed'] is True
    assert spec['execution_allowed'] is False
    assert spec['execution_blockers'] == [NOMINAL_TOOL_BLOCKER]
    assert [stage['name'] for stage in spec['stages']] == ['pregrasp', 'grasp', 'lift']
    assert spec['pregrasp_clearance_m'] == pytest.approx(0.03)


def test_user_can_explicitly_authorize_nominal_urdf_tool_execution():
    spec = build_motion_spec(
        recognition([NOMINAL_TOOL_BLOCKER]),
        allow_nominal_urdf_tool=True,
    )
    assert spec['status'] == 'moveit_ready'
    assert spec['execution_allowed'] is True
    assert spec['execution_blockers'] == []
    assert spec['nominal_urdf_tool_execution_authorized'] is True
    assert 'nominal_urdf_tool_execution_authorized_by_user' in spec['warnings']


def test_measured_tool_with_no_blockers_is_execution_ready():
    spec = build_motion_spec(recognition())
    assert spec['status'] == 'moveit_ready'
    assert spec['execution_allowed'] is True
    assert spec['execution_blockers'] == []
    assert spec['stages'][0]['pose']['position']['z'] == pytest.approx(0.1415)
    assert spec['stages'][1]['pose']['position']['z'] == pytest.approx(0.1715)
    assert spec['stages'][0]['pose']['orientation']['w'] == pytest.approx(1.0)


def test_unrelated_readiness_blocker_rejects_planning():
    data = recognition(['current_board_plane_unavailable'])
    with pytest.raises(ValueError, match='规划阻断'):
        build_motion_spec(data)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('surface_projection_only', True, '支撑面投影'),
        ('tcp_compensation_applied', False, '工具偏置'),
        ('boundary_confirmed', False, '人工确认'),
        ('status', 'boundary_required', '状态为'),
    ],
)
def test_incomplete_visual_target_never_reaches_moveit(field, value, message):
    data = recognition()
    data['grasp_plans'][0][field] = value
    with pytest.raises(ValueError, match=message):
        build_motion_spec(data)


def test_multiple_ready_targets_require_explicit_candidate():
    data = recognition()
    second = dict(data['grasp_plans'][0])
    second['candidate_id'] = 'B'
    data['grasp_plans'].append(second)
    with pytest.raises(ValueError, match='candidate-id'):
        build_motion_spec(data)
    assert build_motion_spec(data, 'B')['candidate_id'] == 'B'


def test_quaternion_conversion_keeps_pose_finite_and_normalized():
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    quaternion = matrix_to_quaternion(rotation)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [0.1, 0.2, 0.3]
    pose = transform_to_pose(transform)
    assert pose['position'] == pytest.approx({'x': 0.1, 'y': 0.2, 'z': 0.3})
