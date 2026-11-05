"""独立掩码、交叉歧义及人工交界兼容性；全部使用离线合成掩码。"""

import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from segmentation.automatic import AutomaticWireRecognizer, select_wire_masks
from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary, confirm_boundary
from segmentation.wire import analyze_mask, analyze_instance_masks, draw_result, save_result


def scene():
    return np.full((200, 240, 3), 128, np.uint8)


def line(a, b, width=3):
    mask = np.zeros((200, 240), np.uint8)
    cv2.line(mask, a, b, 1, width)
    return mask.astype(bool)


def test_single_line_preserves_legacy_candidates_and_instance_path():
    mask = line((20, 80), (220, 80))
    result = analyze_instance_masks(scene(), [mask], scores=[.9])
    assert len(result.instances) == 1 and not result.instances[0].ambiguous
    assert len(result.candidates) == 2 and len(result.centerline_xy)
    assert np.array_equal(result.instances[0].mask, mask)
    assert result.path_instance_ids == ['W001']
    assert all(c['instance_id'] == 'W001' and not c['path_ambiguous'] for c in result.candidates)
    assert result.instances[0].geometry['branch_pixels'] == 0


def test_parallel_masks_remain_independent_despite_one_display_union():
    masks = [line((20, 60), (220, 60)), line((20, 120), (220, 120))]
    result = analyze_instance_masks(scene(), masks)
    assert len(result.instances) == 2
    assert not any(i.ambiguous for i in result.instances)
    assert len(result.paths_xy) == 2 and result.centerline_xy.size == 0
    assert {i.source_mask_id for i in result.instances} == {'S001', 'S002'}
    assert len({c['instance_id'] for c in result.candidates}) == 2
    assert np.array_equal(result.mask, masks[0] | masks[1])
    for c, region in zip(result.candidates, result.end_regions):
        owner = next(i for i in result.instances if i.instance_id == c['instance_id'])
        assert not region[~owner.mask].any()


def test_crossing_independent_sam_masks_keep_ids_but_do_not_claim_identity():
    masks = [line((20, 100), (220, 100)), line((120, 20), (120, 180))]
    result = analyze_instance_masks(scene(), masks)
    assert len(result.instances) == 2 and len(result.source_masks) == 2
    assert all(i.ambiguous for i in result.instances)
    assert all('touch_or_crossing_unresolved' in i.ambiguity_reasons for i in result.instances)
    assert all(c['path_ambiguous'] for c in result.candidates)
    assert result.instance_relations[0]['overlap_pixels'] > 0
    assert result.centerline_xy.size == 0
    assert all(i.geometry['endpoint_count'] == 2 for i in result.instances)


def test_one_crossed_mask_is_ambiguous_hypothesis_not_two_invented_wires():
    mask = line((20, 100), (220, 100)) | line((120, 20), (120, 180))
    result = analyze_instance_masks(scene(), [mask])
    instance = result.instances[0]
    assert len(result.instances) == 1 and instance.ambiguous
    assert instance.geometry['endpoint_count'] == 4 and instance.geometry['branch_pixels'] > 0
    assert len(result.candidates) == 4 and all(c['path_ambiguous'] for c in result.candidates)


def test_touching_t_shape_is_not_certified_as_single_wire():
    a = line((20, 100), (220, 100))
    b = line((120, 25), (120, 99))
    result = analyze_instance_masks(scene(), [a, b])
    assert all(i.ambiguous for i in result.instances)


@pytest.mark.parametrize('separate_sources', [False, True])
def test_local_break_never_gets_joined_or_declared_same_instance(separate_sources):
    a = line((20, 80), (110, 80))
    b = line((118, 80), (220, 80))
    result = analyze_instance_masks(scene(), [a, b] if separate_sources else [a | b])
    assert len(result.instances) == 2
    assert all(i.ambiguous for i in result.instances)
    assert not result.mask[80, 114]
    assert all(c['path_ambiguous'] for c in result.candidates)
    assert all(not (p[:, 0].min() < 114 < p[:, 0].max()) for p in result.paths_xy)


def test_roi_truncation_keeps_original_coordinates_and_excludes_false_tip():
    roi = [40, 40, 220, 160]
    mask = line((40, 100), (190, 100))
    mask[:, :40] = False
    result = analyze_instance_masks(scene(), [mask], crop_xyxy=roi)
    assert result.instances[0].geometry['roi_truncated']
    assert len(result.candidates) == 1 and result.candidates[0]['tip_xy'][0] > 180
    assert result.instances[0].paths_xy[0][:, 0].min() >= 40


def test_large_width_change_and_sharp_tangent_are_explicit():
    mask = line((20, 80), (220, 80), 1).astype(np.uint8)
    cv2.rectangle(mask, (85, 65), (120, 95), 1, -1)
    result = analyze_mask(scene(), mask)
    assert 'width_inconsistent' in result.instances[0].ambiguity_reasons
    bend = line((20, 80), (120, 80)) | line((120, 80), (120, 180))
    result = analyze_mask(scene(), bend)
    assert result.instances[0].geometry['max_tangent_turn_deg'] > 75
    assert 'tangent_discontinuity' in result.instances[0].ambiguity_reasons


def test_contained_but_distinct_hypotheses_are_not_deduplicated_as_one_wire():
    image = scene()
    thin = line((20, 100), (220, 100))
    crossed = thin | line((120, 20), (120, 180))
    cfg = Config({'wire': {'auto': {'max_width_fraction': .05}}})
    selected = select_wire_masks(image, [{'segmentation': thin, 'predicted_iou': .9},
                                         {'segmentation': crossed, 'predicted_iou': .8}], cfg)
    assert len(selected) == 2
    assert len(select_wire_masks(image, [{'segmentation': thin, 'predicted_iou': .9},
                                       {'segmentation': thin.copy(), 'predicted_iou': .8}], cfg)) == 1


def test_saved_schema_keeps_native_individual_and_source_masks(tmp_path):
    masks = [line((20, 100), (220, 100)), line((120, 20), (120, 180))]
    result = analyze_instance_masks(scene(), masks)
    out = save_result(result, tmp_path/'result')
    data = json.loads((out/'recognition.json').read_text())
    assert data['wire_instances_schema_version'] == 1 and data['instance_ambiguous']
    assert 'centerlines_by_component' in data and 'mask_pixels' in data
    assert data['path_instance_ids'] == ['W001', 'W002']
    assert not data['robot_pose_valid'] and not data['terminal_semantics_verified']
    for instance, mask in zip(data['instances'], masks):
        assert np.array_equal(cv2.imread(str(out/instance['mask_file']), 0) > 0, mask)
    for source, mask in zip(data['source_masks'], masks):
        assert np.array_equal(cv2.imread(str(out/source['mask_file']), 0) > 0, mask)
    assert draw_result(result).shape == scene().shape


def test_manual_click_on_parallel_wire_keeps_instance_ownership():
    masks = [line((20, 60), (220, 60)), line((20, 120), (220, 120))]
    result = analyze_instance_masks(scene(), masks)
    apply_grasp_policy(result, Config())
    first = confirm_body_boundary(result, [22, 60], Config())
    second = confirm_body_boundary(result, [22, 120], Config())
    owners = {c['id']: c['instance_id'] for c in result.candidates}
    assert owners[first] != owners[second]
    assert sum(p['boundary_confirmed'] for p in result.grasp_plans) == 2


def test_manual_interior_click_still_works_on_clean_single_wire():
    result = analyze_instance_masks(scene(), [line((20, 80), (220, 80))])
    result.candidates, result.end_regions = [], []
    chosen = confirm_body_boundary(result, [120, 80], Config())
    candidate = next(c for c in result.candidates if c['id'] == chosen)
    assert candidate['instance_id'] == 'W001' and not candidate['path_ambiguous']


def test_manual_click_cannot_clear_crossing_or_longest_path_ambiguity():
    crossed = line((20, 100), (220, 100)) | line((120, 20), (120, 180))
    result = analyze_instance_masks(scene(), [crossed])
    apply_grasp_policy(result, Config())
    before = copy.deepcopy(result.to_dict())
    with pytest.raises(ValueError, match='歧义'):
        confirm_body_boundary(result, [120, 100], Config())
    assert result.to_dict() == before
    # 即使只走旧端点确认API，已有歧义也必须传递到策略结果。
    chosen = confirm_boundary(result, result.candidates[0]['tip_xy'], Config())
    assert next(p for p in result.grasp_plans if p['candidate_id'] == chosen)['status'] == 'wire_instance_ambiguous'


def test_auto_pipeline_retains_crossed_sources_after_roi_restoration():
    image = np.full((300, 400, 3), 128, np.uint8)
    roi = [80, 50, 320, 250]
    masks = [line((20, 100), (220, 100)), line((120, 20), (120, 180))]
    cfg = Config({'wire': {'auto': {'roi_xyxy': roi, 'terminal_enabled': False,
                                    'max_width_fraction': .05}}})
    recognizer = AutomaticWireRecognizer(cfg)
    recognizer._generate = lambda actual: [
        {'segmentation': mask, 'predicted_iou': .9} for mask in masks]
    result = recognizer.recognize(image)
    assert len(result.instances) == 2 and all(i.ambiguous for i in result.instances)
    assert all(i.mask.shape == image.shape[:2] for i in result.instances)
    assert result.instances[0].mask[150, 150]
    assert not result.instances[0].mask[:50].any()
