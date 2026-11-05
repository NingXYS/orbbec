"""六项审计反例回归；物理来源合同使用隔离文件夹，绝不连接硬件。"""

import ast
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm.feedback_safety import MotionAudit, update_joints, require_joints
from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge, sha256
from arm.moveit_wire_adapter import build_motion_spec
from arm import stationary_capture as single
from arm.stationary_sequence import _scene_change
from common.config import Config
from segmentation.multiframe import analyze_sequence
from segmentation.wire import analyze_instance_masks
from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary
from test_stationary_sequence import PairHelper, write
from test_visual_path_gate import valid_recognition, get_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bound_target(tmp_path, monkeypatch):
    """只模拟物理来源文件合同；数值、标定指纹、配对及最终层实际运行。"""
    report = tmp_path/'calibration'
    report.mkdir()
    k = [[500., 0, 120], [0, 500, 60], [0, 0, 1]]
    camera = {'serial': 'TEST_ONLY', 'geometry_verified': True, 'geometry_scope': 'rgb_only',
              'rgb': {'width': 240, 'height': 120, 'K': k, 'D': [0.]*5,
                      'model': 'brown_conrady', 'image_mode': 'raw_distorted'}}
    robot_convention = {'translation_unit': 'm', 'angle_unit': 'rad', 'representation': 'euler',
                        'reference': 'base_flange', 'euler_convention': 'extrinsic_xyz_RzRyRx',
                        'firmware_version': 'TEST', 'audited_source_sha': 'TEST'}
    session = {'calibration_mode': 'eye_in_hand', 'robot': robot_convention,
               'camera': {'serial': 'TEST_ONLY'}}
    calibration = {'methods': {'PARK': {'T_gripper_rgb': np.eye(4).tolist(),
                                      'validation': {'translation_rms_m': 0., 'translation_max_m': 0.}}}}
    for name, content in [('camera_used.json', camera), ('config_used.json', session), ('result.json', calibration)]:
        write(report/name, content)
    hashes = {name: sha256(report/file) for name, file in [('result', 'result.json'), ('camera', 'camera_used.json'), ('session', 'config_used.json')]}
    binding = tmp_path/'binding.json'
    write(binding, {'handeye_root': str(tmp_path), 'calibration_report': 'calibration',
                    'camera_serial': 'TEST_ONLY', 'method': 'PARK', 'channel': 'TEST', 'sha256': hashes})
    bundle = HandeyeBundle(binding)
    snapshot = tmp_path/'snapshot'
    (snapshot/'capture').mkdir(parents=True)
    image = np.full((120, 240, 3), 128, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.line(mask, (20, 60), (220, 60), 1, 3)
    image[mask > 0] = 20
    cv2.imwrite(str(snapshot/'capture/rgb.png'), image)
    cv2.imwrite(str(snapshot/'capture/depth.png'), np.full(mask.shape, 500, np.uint16))
    actual = deepcopy(camera)
    actual.update(pipeline_parameter_image_is_mirrored=0, pipeline_rgb_intrinsics_match_stream=1,
                  capture_pixel_processing={'software_crop_resize_mirror_flip_undistort': 'none'})
    write(snapshot/'capture/camera.json', actual)
    write(snapshot/'capture/frames.json', {name: {'host_receive_timestamp_us': '1000000000'} for name in ('rgb', 'depth')})
    (snapshot/'capture/COMPLETE').write_text('TEST ONLY')
    samples = [{'host_kernel_timestamp_s': 999+i*.005, 'arrival_monotonic_s': 99+i*.005,
                'pose_m_rad': [0.]*6} for i in range(601)]
    write(snapshot/'robot.json', {'status': 'passed', 'firmware_version': 'TEST', 'source_sha': 'TEST',
                                 'application_frames_sent_by_tool': 0, 'sdk_session_connected': False,
                                 'channel': 'TEST', 'samples': samples})
    write(snapshot/'pair.json', {'status': 'paired_stationary', 'data_origin': 'physical_camera',
                                'capture_schema_version': 2, 'capture_kind': 'single_stationary_snapshot',
                                'calibration_sha256': hashes, 'T_base_flange': np.eye(4).tolist(),
                                'image_sha256': sha256(snapshot/'capture/rgb.png')})
    tool = tmp_path/'tool.json'
    write(tool, {'verified': True, 'evidence': 'TEST ONLY FILE CONTRACT', 'camera_serial': 'TEST_ONLY',
                 'frame_definition': 'x_closing_y_cross_z_approach', 'T_flange_grasp': np.eye(4).tolist()})
    plane = tmp_path/'plane.json'
    write(plane, {'schema_version': 1, 'frame_id': 'base_link', 'camera_serial': 'TEST_ONLY',
                  'handeye_sha256': hashes, 'base_z_offset_m': 0., 'normal_base': [0., 0., -1.], 'offset_base_m': -.5})
    monkeypatch.setattr(single, 'helpers', lambda b: PairHelper)
    bridge = WireHandeyeBridge(bundle, snapshot, center_height_m=.002, support_relation_confirmed=True,
                              installation_confirmed=True, tool_file=tool, fixed_plane_file=plane)
    result = analyze_instance_masks(image, [mask], scores=[.9])
    apply_grasp_policy(result, Config())
    chosen = confirm_body_boundary(result, [22, 60], Config())
    bridge.apply(result, Config())
    return result.to_dict(), chosen, binding, snapshot


def test_complete_source_contract_can_pass_without_hardware(bound_target):
    data, chosen, binding, snapshot = bound_target
    spec = build_motion_spec(data, chosen, binding_file=binding)
    assert spec['source_review']['passed'], spec['source_review']
    assert spec['visual_path_review']['passed'] and spec['execution_allowed']


@pytest.mark.parametrize('fault', ['missing_snapshot', 'false_digest', 'simulated', 'legacy', 'changed_pixels',
                                  'changed_tool', 'changed_plane', 'changed_calibration', 'wrong_transform', 'failed_sequence'])
def test_source_contract_tampering_never_grants_execution(bound_target, fault):
    data, chosen, binding, snapshot = bound_target
    policy, plan = data['grasp_policy'], get_plan(data, chosen)
    source = policy['source_provenance']
    if fault == 'missing_snapshot':
        policy['capture_snapshot'] = str(snapshot/'missing')
    elif fault == 'false_digest':
        policy['image_rgb_sha256'] = '0'*64
        plan['visual_path_validation']['evidence']['image_rgb_sha256'] = '0'*64
    elif fault in ('simulated', 'legacy', 'failed_sequence'):
        pair = json.loads((snapshot/'pair.json').read_text())
        if fault == 'simulated':
            pair['data_origin'] = 'simulated'
        elif fault == 'legacy':
            del pair['capture_schema_version']
        else:
            pair.update(capture_kind='stationary_sequence_frame', sequence_manifest='../../manifest.json')
        write(snapshot/'pair.json', pair)
        source['files']['pair.json'] = sha256(snapshot/'pair.json')
    elif fault == 'changed_pixels':
        cv2.imwrite(str(snapshot/'capture/rgb.png'), np.zeros((120, 240, 3), np.uint8))
    elif fault in ('changed_tool', 'changed_plane'):
        key = 'tool' if fault == 'changed_tool' else 'fixed_plane'
        Path(source[key]['path']).write_text('{}')
    elif fault == 'changed_calibration':
        (binding.parent/'calibration/result.json').write_text('{}')
    else:
        plan['coordinate_verification']['T_base_rgb'][0][3] = .1
    spec = build_motion_spec(data, chosen, binding_file=binding)
    assert not spec['source_review']['passed'] and not spec['execution_allowed']


def test_synthetic_visual_fixture_is_not_execution_source():
    data, chosen = valid_recognition()
    spec = build_motion_spec(data, chosen)
    assert spec['visual_path_review']['passed'] and not spec['execution_allowed']
    assert 'execution_source_missing' in spec['execution_blockers']


@pytest.mark.parametrize('authorized', [False, True])
def test_actual_nominal_tool_cannot_hide_behind_missing_upstream_blocker(bound_target, authorized):
    data, chosen, binding, snapshot = bound_target
    contract = data['grasp_policy']['source_provenance']['tool']
    tool_path = Path(contract['path'])
    tool = json.loads(tool_path.read_text())
    tool.update(verified=False, verification='official_model_nominal_preview_only')
    write(tool_path, tool)
    contract['sha256'] = sha256(tool_path)
    spec = build_motion_spec(data, chosen, binding_file=binding, allow_nominal_urdf_tool=authorized)
    assert spec['execution_allowed'] is authorized
    assert spec['nominal_urdf_tool_execution_authorized'] is authorized
    assert 'nominal_finger_tip_reference_not_verified_grasp_center' in spec['warnings']


def test_per_joint_age_cannot_be_refreshed_by_gripper():
    positions, seen = {}, {}
    names = [f'joint{i}' for i in range(1, 7)]
    update_joints(positions, seen, names, [0.]*6, 100.)
    update_joints(positions, seen, ['gripper'], [.05], 110.)
    with pytest.raises(ValueError, match='joint1'):
        require_joints(positions, seen, names, 110.)
    require_joints(positions, seen, ['gripper'], 110.)


@pytest.mark.parametrize('fault', ['nan', 'missing_value', 'duplicate_name'])
def test_invalid_joint_updates_invalidate_previous_value(fault):
    positions, seen = {'joint1': 0.}, {'joint1': 100.}
    names, values = ['joint1'], [float('nan')]
    if fault == 'missing_value': values = []
    if fault == 'duplicate_name': names, values = ['joint1', 'joint1'], [0., 1.]
    update_joints(positions, seen, names, values, 101.)
    with pytest.raises(ValueError): require_joints(positions, seen, ['joint1'], 101.)


def test_motion_attempt_and_observation_are_separate():
    audit = MotionAudit()
    assert audit.report()['physical_motion_performed'] is False
    audit.attempt('gripper_open')
    assert audit.report()['physical_motion_performed'] is None
    audit.observe({'gripper': 0.}, {'gripper': .05})
    assert audit.report()['physical_motion_performed'] is True


@pytest.mark.parametrize('tamper', [False, True])
def test_partial_failure_report_is_unknown_not_no_motion(tmp_path, tamper):
    tree = ast.parse((ROOT/'scripts/run_moveit_wire_grasp.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    source = tmp_path/'recognition.json'
    source.write_text('{}')
    args = SimpleNamespace(report=None, recognition=source, candidate_id='A', base_frame='base_link',
                           tip_link='tcp_link', expected_parent_frame='link6', binding=tmp_path/'binding',
                           allow_nominal_urdf_tool=False, execute=True, joints=['joint1'])
    saved = []
    spec = {'execution_allowed': True, 'candidate_id': 'A', 'gripper': {'open_width_m': .1, 'force_n': 1.},
            'stages': [{'name': 'pregrasp'}]}
    class FakeNode:
        def __init__(self, *args): self._motion_audit, self._joint_positions = MotionAudit(), {'joint1': 0.}
        def wait_ready(self, *args): pass
        def _require_parent_tip_identity(self): return {}
        def command_gripper(self, *args, **kwargs):
            if tamper:
                source.write_text('changed during mock session')
            return {'mock_command_returned': True}
        def plan_or_execute(self, *args): raise RuntimeError('mock failure after gripper request')
        def destroy_node(self): pass
    scope = {'parser': lambda: SimpleNamespace(parse_args=lambda: args), 'sha256': sha256,
             'load_recognition': lambda path: (path, {}), 'build_motion_spec': lambda *a, **k: spec,
             'rclpy': SimpleNamespace(init=lambda: None, shutdown=lambda: None), 'WireMoveGroup': FakeNode,
             'ROOT': ROOT, 'sys': sys, 'datetime': datetime,
             'write_json_exclusive': lambda path, report: saved.append(deepcopy(report)) or path}
    exec(compile(ast.Module(body=[main], type_ignores=[]), '<offline main>', 'exec'), scope)
    assert scope['main']() == 2
    assert saved[0]['physical_motion_performed'] is None
    assert saved[0]['motion_commands_attempted'] == (['gripper_open'] if tamper else ['gripper_open', 'pregrasp'])


def test_invalid_execution_source_rejected_before_ros_init(tmp_path):
    tree = ast.parse((ROOT/'scripts/run_moveit_wire_grasp.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    source = tmp_path/'recognition.json'
    source.write_text('{}')
    args = SimpleNamespace(report=None, recognition=source, candidate_id='A', base_frame='base_link',
                           tip_link='tcp_link', expected_parent_frame='link6', binding=None,
                           allow_nominal_urdf_tool=False, execute=True)
    initialized = []
    scope = {'parser': lambda: SimpleNamespace(parse_args=lambda: args), 'sha256': sha256,
             'load_recognition': lambda path: (path, {}), 'sys': sys,
             'build_motion_spec': lambda *a, **k: {'execution_allowed': False, 'execution_blockers': ['execution_source_missing']},
             'rclpy': SimpleNamespace(init=lambda: initialized.append(True))}
    exec(compile(ast.Module(body=[main], type_ignores=[]), '<offline main>', 'exec'), scope)
    assert scope['main']() == 2 and not initialized


def test_real_callback_only_updates_received_joint_times():
    tree = ast.parse((ROOT/'scripts/run_moveit_wire_grasp.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WireMoveGroup')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_joint_callback')
    scope = {'update_joints': update_joints, 'time': SimpleNamespace(monotonic=lambda: 100.)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<offline callback>', 'exec'), scope)
    node = SimpleNamespace(_joint_positions={'joint1': 0.}, _joint_seen_at={'joint1': 90.}, _motion_audit=MotionAudit())
    scope['_joint_callback'](node, SimpleNamespace(name=['gripper'], position=[.05]))
    assert node._joint_seen_at == {'joint1': 90., 'gripper': 100.}


def test_shifted_parallel_hypotheses_not_deduplicated():
    image = np.full((480, 640, 3), 128, np.uint8)
    masks = []
    for y in (200, 201):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.line(mask, (60, y), (580, y), 1, 3)
        masks.append(mask > 0)
    result = analyze_instance_masks(image, masks)
    assert len(result.instances) == 2 and all(i.ambiguous for i in result.instances)


def test_duplicate_width_hypotheses_keep_single_instance_and_manual_click():
    image = np.full((480, 640, 3), 128, np.uint8)
    masks = []
    for width in (3, 5):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.line(mask, (60, 200), (580, 200), 1, width)
        masks.append(mask > 0)
    result = analyze_instance_masks(image, masks, scores=[.8, .9])
    assert len(result.instances) == 1 and len(result.source_masks) == 2
    assert not result.instances[0].ambiguous
    assert np.array_equal(result.instances[0].mask, masks[1])
    assert result.automation['duplicate_wire_hypotheses']
    apply_grasp_policy(result, Config())
    chosen = confirm_body_boundary(result, [65, 200], Config())
    assert next(p for p in result.grasp_plans if p['candidate_id'] == chosen)['boundary_confirmed']


def test_wire_movement_cannot_hide_behind_500_background_corners():
    image = np.full((480, 640, 3), 128, np.uint8)
    image[:330] = np.random.default_rng(42).integers(0, 256, (330, 640, 3), dtype=np.uint8)
    a, b = image.copy(), image.copy()
    cv2.line(a, (120, 430), (500, 430), (100,)*3, 2)
    cv2.line(b, (120, 438), (500, 438), (100,)*3, 2)
    with pytest.raises(ValueError, match='局部'):
        _scene_change(a, b, [0, 0, 640, 480])


@pytest.mark.parametrize('content', [[], None, 5, {'frames': [None]}, 'broken'])
def test_malformed_manifest_saves_manual_review_report(tmp_path, content):
    sequence = tmp_path/'sequence'
    sequence.mkdir()
    write(sequence/'manifest.json', content)
    report = analyze_sequence(None, sequence, tmp_path/'analysis', Config())
    assert report['status'] == 'manual_confirmation_required'
    assert report['rejection_reasons'] == ['sequence_validation_failed']
    assert (tmp_path/'analysis/analysis.json').exists()
