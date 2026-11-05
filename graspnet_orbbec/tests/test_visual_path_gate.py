"""局部路径门控的端到端纯数据测试；合成相机/实例，不访问设备或ROS。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.handeye_bridge import WireHandeyeBridge
from arm.moveit_wire_adapter import build_motion_spec, load_recognition
from common.config import Config
from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary
from segmentation.path_validation import execution_path_review
from segmentation.wire import analyze_instance_masks


def setup_target():
    image = np.full((120, 240, 3), 128, np.uint8)
    mask = np.zeros((120, 240), np.uint8)
    cv2.line(mask, (20, 60), (220, 60), 1, 3)
    image[mask > 0] = 20
    result = analyze_instance_masks(image, [mask], scores=[.9])
    cfg = Config()
    apply_grasp_policy(result, cfg)
    chosen = confirm_body_boundary(result, [22, 60], cfg)
    bridge = object.__new__(WireHandeyeBridge)
    bridge.image, bridge.snapshot = image, Path('/synthetic_only')
    bridge.t_base_flange = np.eye(4)
    bridge.plane = {'normal_camera': [0, 0, -1], 'origin_camera_m': [0, 0, .5]}
    bridge.plane_error = None
    bridge.height, bridge.support_confirmed, bridge.installation_confirmed = .002, True, True
    bridge.bundle = SimpleNamespace(
        k=np.array([[500., 0, 120], [0, 500, 60], [0, 0, 1]]), d=np.zeros(5),
        t_flange_rgb=np.eye(4), camera={'serial': 'TEST_ONLY'},
        info=lambda: {'camera_serial': 'TEST_ONLY', 'absolute_accuracy_verified': False})
    bridge.tool = {'verified': True, 'evidence': 'synthetic_only', 'camera_serial': 'TEST_ONLY',
                   'frame_definition': 'x_closing_y_cross_z_approach',
                   'T_flange_grasp': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .135], [0, 0, 0, 1]]}
    return result, chosen, bridge


def valid_recognition():
    result, chosen, bridge = setup_target()
    bridge.apply(result, Config())
    return result.to_dict(), chosen


def get_plan(data, chosen):
    return next(p for p in data['grasp_plans'] if p['candidate_id'] == chosen)


def test_success_keeps_geometry_and_distinguishes_four_validation_stages():
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    assert plan['offset_mm'] == 14
    assert plan['visual_path_validation']['passed']
    assert plan['boundary_verification'] == {'source': 'manual', 'human_confirmed': True, 'point_available': True}
    assert plan['verification_stages']['instance_path_verified']
    assert plan['verification_stages']['coordinate_calculation_succeeded']
    assert plan['readiness_blockers'] == []
    assert plan['selection_v2']['offset_m'] == pytest.approx(.014)
    spec = build_motion_spec(data, chosen)
    assert spec['visual_path_review']['passed']
    assert not spec['execution_allowed'] and spec['source_review']['rejection_reasons'] == ['execution_source_missing']
    assert not plan['executable'] and not data['grasp_policy']['motion_commands_enabled']


@pytest.mark.parametrize('problem,reason', [
    ('crossing', 'visual_path_branch_detected'),
    ('break', 'visual_path_outside_instance_mask'),
    ('width', 'visual_path_width_abnormal'),
])
def test_upstream_geometry_failures_enter_readiness_and_independent_execution_gate(problem, reason):
    result, chosen, bridge = setup_target()
    owner = result.instances[0]
    if problem == 'crossing':
        extra = np.zeros(owner.mask.shape, np.uint8)
        cv2.line(extra, (30, 40), (30, 80), 1, 3)
        owner.mask |= extra > 0
    elif problem == 'break':
        owner.mask[:, 29:31] = False
    else:
        owner.mask[45:76, 30:39] = True
    # 模拟上游未发现异常：path_ambiguous和旧geometry都不变，新检查必须从掩码发现。
    bridge.apply(result, Config())
    data = result.to_dict()
    plan = get_plan(data, chosen)
    assert reason in plan['readiness_blockers']
    assert plan['verification_stages']['coordinate_calculation_succeeded']
    assert not plan['verification_stages']['instance_path_verified']
    # 清除上游阻断/伪造passed，最终适配仍须独立拒绝。
    plan['readiness_blockers'] = []
    plan['visual_path_validation']['passed'] = True
    spec = build_motion_spec(data, chosen)
    assert not spec['execution_allowed'] and reason in spec['execution_blockers']


def test_wrong_instance_id_is_rejected_despite_path_ambiguous_false():
    data, chosen = valid_recognition()
    candidate = next(c for c in data['candidates'] if c['id'] == chosen)
    candidate['path_ambiguous'] = False
    candidate['instance_id'] = 'WRONG_INSTANCE'
    spec = build_motion_spec(data, chosen)
    assert not spec['execution_allowed']
    assert 'visual_path_instance_mismatch' in spec['execution_blockers']


def test_competing_instance_on_local_route_is_rejected_without_upstream_ambiguity_flag():
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    rows = plan['visual_path_validation']['evidence']['other_mask_rows']
    evidence = plan['visual_path_validation']['evidence']
    x, y = plan['boundary_xy']
    px, py = evidence['patch_origin_xy']
    row = list(rows[int(y)-py])
    row[int(x)-px+5] = '1'
    rows[int(y)-py] = ''.join(row)
    spec = build_motion_spec(data, chosen)
    assert not spec['execution_allowed'] and 'visual_path_competing_instance' in spec['execution_blockers']


def test_sudden_turn_is_not_accepted_as_valid_local_path():
    from segmentation.grasp_policy import plan_candidate_v2
    from segmentation.path_validation import build_path_validation

    result, chosen, _ = setup_target()
    candidate = next(c for c in result.candidates if c['id'] == chosen)
    path = np.vstack([np.c_[np.arange(22, 28), np.full(6, 60)],
                      np.c_[np.full(25, 27), np.arange(61, 86)]])
    candidate['inward_path_xy'] = path.tolist()
    owner = result.instances[0]
    owner.mask[:] = False
    cv2.polylines(owner.mask.view(np.uint8), [path.reshape(-1, 1, 2).astype(np.int32)], False, 1, 3)
    xyz = np.c_[path*.001, np.full(len(path), .5)]
    plan = plan_candidate_v2(candidate, path[0], path_xyz_m=xyz, plane_normal=[0, 0, -1])
    plan['instance_id'] = candidate['instance_id']
    validation = build_path_validation(result, candidate, plan, xyz)
    assert not validation['passed']
    assert 'visual_path_direction_discontinuity' in validation['rejection_reasons']


def test_model_prediction_coordinate_success_is_not_manual_confirmation():
    result, chosen, bridge = setup_target()
    candidate = next(c for c in result.candidates if c['id'] == chosen)
    candidate['boundary_confirmation_source'] = 'model_prediction'
    bridge.apply(result, Config())
    data = result.to_dict()
    plan = get_plan(data, chosen)
    assert plan['visual_path_validation']['passed']
    assert plan['verification_stages']['coordinate_calculation_succeeded']
    assert plan['boundary_verification']['source'] == 'model_prediction'
    assert not build_motion_spec(data, chosen)['execution_allowed']
    assert 'visual_path_boundary_not_manually_verified' in plan['readiness_blockers']
    # 人工确认可保持原交界功能，随后必须重新做米制路径验收。
    confirm_body_boundary(result, [22, 60], Config())
    bridge.apply(result, Config())
    spec = build_motion_spec(result.to_dict(), chosen)
    assert spec['visual_path_review']['passed'] and not spec['execution_allowed']


@pytest.mark.parametrize('field', ['visual_path_validation', 'boundary_verification',
                                  'verification_stages', 'coordinate_verification'])
def test_missing_required_new_plan_fields_never_grant_execution(field):
    data, chosen = valid_recognition()
    del get_plan(data, chosen)[field]
    spec = build_motion_spec(data, chosen)
    assert spec['planning_allowed'] and not spec['execution_allowed']
    assert spec['execution_blockers']


@pytest.mark.parametrize('field', ['instances', 'candidates', 'wire_instances_schema_version'])
def test_missing_recognition_instance_evidence_is_blocked(field):
    data, chosen = valid_recognition()
    del data[field]
    assert not build_motion_spec(data, chosen)['execution_allowed']


def test_legacy_json_still_loads_for_analysis_but_no_nominal_tool_override_can_bypass_gate(tmp_path):
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    for key in ['visual_path_validation', 'verification_stages', 'boundary_verification', 'coordinate_verification']:
        del plan[key]
    path = tmp_path/'recognition.json'
    path.write_text(json.dumps(data))
    _, loaded = load_recognition(path)
    for allow in [False, True]:
        spec = build_motion_spec(loaded, chosen, allow_nominal_urdf_tool=allow)
        assert spec['status'] == 'plan_only' and spec['planning_allowed']
        assert spec['execution_blockers'] == ['visual_path_validation_missing', 'execution_source_missing']


def test_upstream_passed_flag_and_image_or_coordinate_tampering_are_not_trusted():
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    plan['visual_path_validation']['evidence']['image_rgb_sha256'] = 'OTHER_IMAGE'
    assert not execution_path_review(data, plan)['passed']
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    plan['coordinate_verification']['T_base_rgb'][0][3] = .01
    assert 'visual_path_coordinate_evidence_invalid_or_missing' in build_motion_spec(data, chosen)['execution_blockers']


@pytest.mark.parametrize('key', ['path_camera_m', 'owner_mask_rows', 'other_mask_rows', 'patch_origin_xy'])
def test_incomplete_embedded_path_evidence_is_blocked(key):
    data, chosen = valid_recognition()
    del get_plan(data, chosen)['visual_path_validation']['evidence'][key]
    spec = build_motion_spec(data, chosen)
    assert not spec['execution_allowed'] and 'visual_path_evidence_invalid' in spec['execution_blockers']


def test_forged_shortcut_or_changed_grasp_center_is_detected():
    data, chosen = valid_recognition()
    get_plan(data, chosen)['grasp_camera_m'][0] += .001
    assert not build_motion_spec(data, chosen)['execution_allowed']
    data, chosen = valid_recognition()
    candidate = next(c for c in data['candidates'] if c['id'] == chosen)
    candidate['inward_path_xy'][5][0] += 8
    assert not build_motion_spec(data, chosen)['execution_allowed']


@pytest.mark.parametrize('mutation', ['missing', 'omit_mask', 'wrong_mask', 'duplicate', 'bad_run'])
def test_independent_full_instance_evidence_catches_upstream_omissions(mutation):
    data, chosen = valid_recognition()
    package = data['grasp_policy']['instance_mask_evidence']
    if mutation == 'missing':
        del data['grasp_policy']['instance_mask_evidence']
    elif mutation == 'omit_mask':
        package['masks'] = []
    elif mutation == 'wrong_mask':
        package['masks'][0]['runs'] = []
    elif mutation == 'duplicate':
        package['masks'].append(dict(package['masks'][0]))
    else:
        package['masks'][0]['runs'] = [[-1, 999999]]
    spec = build_motion_spec(data, chosen)
    assert not spec['execution_allowed']
    assert any('instance_mask_evidence' in r for r in spec['execution_blockers'])


def test_omitted_competing_mask_patch_does_not_bypass_final_gate():
    data, chosen = valid_recognition()
    plan = get_plan(data, chosen)
    # 上游把局部other mask误写成全零，但全局原始实例中仍有交叉像素。
    data['instances'].append({'instance_id': 'W002', 'instance_ambiguous': False, 'ambiguity_reasons': []})
    x, y = plan['boundary_xy']
    width = data['grasp_policy']['instance_mask_evidence']['shape_hw'][1]
    data['grasp_policy']['instance_mask_evidence']['masks'].append(
        {'instance_id': 'W002', 'runs': [[int(y)*width+int(x)+5, 1]]})
    spec = build_motion_spec(data, chosen)
    assert 'visual_path_instance_mask_evidence_mismatch' in spec['execution_blockers']


@pytest.mark.parametrize('field', ['K', 'D', 'T_base_rgb', 'T_flange_grasp'])
def test_coordinate_provenance_missing_is_blocked(field):
    data, chosen = valid_recognition()
    del get_plan(data, chosen)['coordinate_verification'][field]
    assert not build_motion_spec(data, chosen)['execution_allowed']


def test_missing_passed_field_cannot_be_filled_in_by_adapter():
    data, chosen = valid_recognition()
    del get_plan(data, chosen)['visual_path_validation']['passed']
    assert not build_motion_spec(data, chosen)['execution_allowed']
