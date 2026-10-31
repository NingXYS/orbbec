"""合成几何验证与拒绝路径；测试通过不代表实机抓取成功。"""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge
from arm.wire_geometry import (board_plane, fit_pivot, flange_transform,
                               grasp_transforms, pixels_on_plane, rigid_transform)
from common.config import Config
from segmentation.grasp_policy import apply_grasp_policy, plan_candidate
from segmentation.wire import WireResult


K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
D = np.array([.08, -.1, .0002, -.0004, 0.])
PLANE = {'normal_camera': [0, 0, -1], 'origin_camera_m': [0, 0, .5]}


def test_distorted_pixels_roundtrip_to_measured_plane():
    points = np.array([[-.12, -.1, .5], [.08, .04, .5], [.2, .1, .5]])
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    xyz = pixels_on_plane(uv, K, D, PLANE)
    assert xyz == pytest.approx(points, abs=1e-8)
    raised = pixels_on_plane(uv, K, D, PLANE, .003)
    assert raised[:, 2] == pytest.approx(np.full(3, .497))


def test_parallel_and_behind_camera_rays_rejected():
    with pytest.raises(ValueError):
        pixels_on_plane([[320, 240]], K, D,
                        {'normal_camera': [1, 0, 0], 'origin_camera_m': [1, 0, .5]})
    with pytest.raises(ValueError):
        pixels_on_plane([[320, 240]], K, D, PLANE, 1)


@pytest.mark.parametrize('bad', [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]),
                                 np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)])
def test_invalid_transform_never_reaches_tool_compensation(bad):
    with pytest.raises(ValueError):
        rigid_transform(bad)


def test_base_chain_and_rotated_tool_offset_are_not_added_in_base_axes():
    candidate = {'id': 'A', 'inward_path_xy': [[320, 240], [340, 240], [360, 240]]}
    path = pixels_on_plane(candidate['inward_path_xy'], K, np.zeros(5), PLANE)
    plan = plan_candidate(candidate, [320, 240], path_xyz_m=path, plane_normal=[0, 0, -1])
    assert plan['grasp_camera_m'] == pytest.approx([.01, 0, .5])
    base_flange = flange_transform([.3, .1, .2, .2, -.3, 1.1])
    flange_camera = flange_transform([.02, -.04, .03, -.1, .2, .3])
    tool = flange_transform([.01, -.02, .12, 0, .2, .4])
    result = grasp_transforms(plan, base_flange @ flange_camera, tool, .03)
    grasp = np.array(result['T_base_grasp'])
    flange = np.array(result['T_base_flange_grasp'])
    assert flange @ tool == pytest.approx(grasp, abs=1e-10)
    expected = base_flange @ flange_camera @ np.r_[plan['grasp_camera_m'], 1]
    assert result['grasp_base_m'] == pytest.approx(expected[:3])
    pre = np.array(result['T_base_flange_pregrasp']) @ tool
    assert np.linalg.norm(pre[:3, 3]-grasp[:3, 3]) == pytest.approx(.03)
    assert np.dot(pre[:3, 3]-grasp[:3, 3], grasp[:3, 2]) == pytest.approx(-.03)
    assert np.linalg.det(grasp[:3, :3]) == pytest.approx(1)


def test_pivot_recovers_rigid_tip_but_does_not_invent_tool_axes():
    tip = np.array([.01, -.03, .15])
    fixed = np.array([.4, .1, .02])
    poses = []
    for i in range(10):
        t = flange_transform([0, 0, 0, .3*np.sin(i), .4*np.cos(i), .5*np.sin(i*.6)])
        t[:3, 3] = fixed-t[:3, :3]@tip
        poses.append(t)
    result = fit_pivot(poses)
    assert result['tcp_translation_flange_m'] == pytest.approx(tip)
    assert result['fixed_point_base_m'] == pytest.approx(fixed)
    assert result['rms_m'] < 1e-12
    assert result['orientation_calibrated'] is False
    assert result['verified'] is False
    with pytest.raises(ValueError):
        fit_pivot([np.eye(4)]*10)


def test_missing_board_never_falls_back_to_old_training_board_pose():
    with pytest.raises(ValueError, match='当前原图'):
        board_plane(np.zeros((480, 640, 3), np.uint8), K, D,
                    {'inner_corners': [8, 5], 'square_size_m': .02})


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/'handeye'
    report = root/'report'
    report.mkdir(parents=True)
    robot = {'translation_unit': 'm', 'angle_unit': 'rad', 'representation': 'euler',
             'reference': 'base_flange', 'euler_convention': 'extrinsic_xyz_RzRyRx'}
    camera = {'serial': 'TEST_ONLY', 'geometry_verified': True, 'geometry_scope': 'rgb_only',
              'rgb': {'K': K.tolist(), 'D': D.tolist(), 'width': 640, 'height': 480,
                      'image_mode': 'raw_distorted', 'model': 'brown_conrady'}}
    data = {'result.json': {'methods': {'PARK': {'T_gripper_rgb': np.eye(4).tolist(),
                                               'validation': {'translation_rms_m': .005,
                                                              'translation_max_m': .009}}}},
            'camera_used.json': camera,
            'config_used.json': {'robot': robot, 'calibration_mode': 'eye_in_hand',
                                 'camera': {'serial': 'TEST_ONLY'},
                                 'board': {'inner_corners': [8, 5], 'square_size_m': .02}}}
    for name, value in data.items():
        (report/name).write_text(json.dumps(value))
    binding = {'handeye_root': str(root), 'calibration_report': 'report',
               'camera_serial': 'TEST_ONLY', 'method': 'PARK', 'channel': 'can0',
               'sha256': {k: hashlib.sha256((report/n).read_bytes()).hexdigest() for k, n in
                          [('result', 'result.json'), ('camera', 'camera_used.json'),
                           ('session', 'config_used.json')]}}
    path = tmp_path/'binding.json'
    path.write_text(json.dumps(binding))
    return HandeyeBundle(path)


def test_changed_result_requires_reaudit_and_camera_mismatch_rejected(bundle):
    (bundle.report/'result.json').write_text('{}')
    with pytest.raises(ValueError, match='冻结标定'):
        HandeyeBundle(bundle.path)
    with pytest.raises(ValueError, match='序列号'):
        bundle.check_camera({'serial': 'OTHER'})


def test_bridge_refuses_unpaired_or_changed_image(bundle, tmp_path):
    snap = tmp_path/'snap'
    (snap/'capture').mkdir(parents=True)
    cv2.imwrite(str(snap/'capture/rgb.png'), np.zeros((480, 640, 3), np.uint8))
    actual = dict(bundle.camera, pipeline_parameter_image_is_mirrored=0,
                  pipeline_rgb_intrinsics_match_stream=1,
                  capture_pixel_processing={'software_crop_resize_mirror_flip_undistort': 'none'})
    (snap/'capture/camera.json').write_text(json.dumps(actual))
    (snap/'pair.json').write_text(json.dumps({'status': 'unpaired_camera_smoke_test'}))
    with pytest.raises(ValueError, match='快照配对'):
        WireHandeyeBridge(bundle, snap)


def make_preview_bridge(bundle):
    bridge = object.__new__(WireHandeyeBridge)
    bridge.bundle, bridge.snapshot = bundle, Path('/test_only_synthetic')
    bridge.image = np.zeros((480, 640, 3), np.uint8)
    bridge.t_base_flange = np.eye(4)
    bridge.plane = PLANE
    bridge.plane_error = None
    bridge.height, bridge.support_confirmed = None, False
    bridge.installation_confirmed, bridge.tool = False, None
    return bridge


def make_result(image):
    empty = np.zeros(image.shape[:2], bool)
    candidate = {'id': 'A', 'inward_path_xy': [[320, 240], [340, 240], [360, 240]]}
    return WireResult(image.copy(), empty, empty.copy(), np.empty((0, 2)),
                      [candidate], .9, 0, [0, 0, 640, 480])


def test_missing_height_is_explicit_surface_projection_and_never_executable(bundle):
    bridge = make_preview_bridge(bundle)
    result = make_result(bridge.image)
    cfg = Config({'wire': {'grasp_policy': {'offset_mm': 10}}})
    apply_grasp_policy(result, cfg, {'A': [320, 240]})
    bridge.apply(result, cfg)
    plan = result.grasp_plans[0]
    assert plan['status'] == 'base_frame_preview_ready'
    assert plan['surface_projection_only']
    assert plan['grasp_base_m'] is not None
    assert len(plan['readiness_blockers']) == 3
    assert not plan['executable'] and not plan['robot_pose_valid']
    changed = make_result(np.ones_like(bridge.image))
    with pytest.raises(ValueError, match='跨快照'):
        bridge.apply(changed, cfg)


def test_measured_tool_produces_compensated_flange_preview_only(bundle):
    bridge = make_preview_bridge(bundle)
    bridge.height, bridge.support_confirmed, bridge.installation_confirmed = .002, True, True
    bridge.tool = {'verified': True, 'evidence': 'test_only', 'camera_serial': 'TEST_ONLY',
                   'frame_definition': 'x_closing_y_cross_z_approach',
                   'T_flange_grasp': flange_transform([0, 0, .12, 0, 0, 0]).tolist()}
    result = make_result(bridge.image)
    cfg = Config({})
    apply_grasp_policy(result, cfg, {'A': [320, 240]})
    bridge.apply(result, cfg)
    plan = result.grasp_plans[0]
    assert plan['readiness_blockers'] == []
    assert plan['tcp_compensation_applied']
    assert not plan['surface_projection_only']
    assert not plan['executable']
    assert not result.grasp_policy['motion_commands_enabled']


def test_ambiguous_and_unconfirmed_candidates_stay_unlocalized(bundle):
    bridge = make_preview_bridge(bundle)
    result = make_result(bridge.image)
    cfg = Config({})
    apply_grasp_policy(result, cfg)
    bridge.apply(result, cfg)
    assert result.grasp_plans[0]['status'] == 'boundary_required'
    result.candidates[0]['path_ambiguous'] = True
    apply_grasp_policy(result, cfg, {'A': [320, 240]})
    bridge.apply(result, cfg)
    assert result.grasp_plans[0]['status'] == 'wire_instance_ambiguous'


def test_nominal_tip_compensation_is_labelled_and_blocked(bundle):
    bridge = make_preview_bridge(bundle)
    bridge.tool = {'verified': False, 'verification': 'official_model_nominal_preview_only',
                   'evidence': 'test_only', 'camera_serial': 'TEST_ONLY',
                   'frame_definition': 'x_closing_y_cross_z_approach',
                   'T_flange_grasp': flange_transform([0, 0, .1425, 0, 0, 0]).tolist()}
    result = make_result(bridge.image)
    apply_grasp_policy(result, Config({}), {'A': [320, 240]})
    bridge.apply(result, Config({}))
    assert result.grasp_plans[0]['tcp_compensation_applied']
    assert 'nominal_finger_tip_reference_not_verified_grasp_center' in result.grasp_policy['readiness_blockers']
    assert not result.grasp_plans[0]['executable']


def test_missing_plane_preserves_recognition_but_never_adds_coordinates(bundle):
    bridge = make_preview_bridge(bundle)
    bridge.plane, bridge.plane_error = None, 'test_only_missing_board'
    result = make_result(bridge.image)
    apply_grasp_policy(result, Config({}), {'A': [320, 240]})
    bridge.apply(result, Config({}))
    assert result.grasp_plans[0]['status'] == 'metric_path_required'
    assert 'grasp_base_m' not in result.grasp_plans[0]
    assert 'current_board_plane_unavailable' in result.grasp_policy['readiness_blockers']
