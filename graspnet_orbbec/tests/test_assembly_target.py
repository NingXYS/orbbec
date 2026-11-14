"""装配目标拒绝失配和越界数据；不连接 ROS 或真机。"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm.assembly_target import load_assembly_target, validate_runtime_assembly, validate_y_approach


def load(path):
    return load_assembly_target(path, root=ROOT, frame_id='base_link',
                                joints=[f'joint{i}' for i in range(1, 7)])


def test_saved_assembly_target_matches_runtime_model():
    stage = load(ROOT/'configs/wire_assembly_fixed_20261106.json')
    urdf = ROOT.parent/'agilex_official_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/piper_x/urdf/piper_x_description.urdf'
    assert stage['kind'] == 'joint'
    result = validate_runtime_assembly(stage, urdf.read_text(), stage['T_base_flange'])
    assert result['position_error_m'] == pytest.approx(0)
    stage['joint_positions_rad'][3] = np.deg2rad(89.407)
    with pytest.raises(ValueError, match='超出运行模型限位'):
        validate_runtime_assembly(stage, urdf.read_text(), stage['T_base_flange'])


@pytest.mark.parametrize('fault', ['tool_hash', 'tool_transform', 'tcp', 'nan', 'joint_order', 'frame'])
def test_reject_changed_or_inconsistent_target(tmp_path, fault):
    data = json.loads((ROOT/'configs/wire_assembly_fixed_20261106.json').read_text())
    if fault == 'tool_hash':
        data['tool']['sha256'] = '0'*64
    elif fault == 'tool_transform':
        data['tool']['T_flange_grasp'][2][3] += 0.01
    elif fault == 'tcp':
        data['grasp_tcp_goal']['T_base_frame'][0][3] += 0.01
    elif fault == 'nan':
        data['joint_positions_rad'][0] = float('nan')
    elif fault == 'joint_order':
        data['joint_names'].reverse()
    else:
        data['reference_frame'] = 'world'
    path = tmp_path/'target.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(path)


def test_reject_joint_pose_mismatch():
    stage = load(ROOT/'configs/wire_assembly_fixed_20261106.json')
    urdf = '<robot>' + ''.join(f'<joint name="joint{i}"><limit lower="-4" upper="4"/></joint>' for i in range(1,7)) + '</robot>'
    actual = np.array(stage['T_base_flange']);actual[0,3] += 0.01
    with pytest.raises(ValueError, match='正解与保存法兰不一致'):
        validate_runtime_assembly(stage, urdf, actual)


def y_path():
    matrices = [np.eye(4) for _ in range(4)]
    for t, y in zip(matrices, [.03, .02, .01, 0.]):
        t[1,3] = y
    return matrices


def test_y_approach_keeps_x_z_and_orientation():
    result = validate_y_approach(y_path(), -1, .03)
    assert result['travel_m'] == pytest.approx(.03)
    assert result['max_lateral_error_m'] == 0


@pytest.mark.parametrize('fault', ['lateral', 'rotation', 'reverse', 'length'])
def test_reject_nonstraight_or_wrong_direction_final_path(fault):
    matrices = y_path()
    if fault == 'lateral':
        matrices[1][0,3] += .001
    elif fault == 'rotation':
        a=.01
        matrices[1][:3,:3] = [[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
    elif fault == 'reverse':
        matrices.reverse()
    else:
        matrices[0][1,3] += .01
    with pytest.raises(ValueError, match='单调Y轴直线'):
        validate_y_approach(matrices, -1, .03)
