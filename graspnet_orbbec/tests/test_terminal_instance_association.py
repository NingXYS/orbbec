"""端子与线材关联只作语义证据；使用离线掩码，不加载YOLO或SAM模型。"""

import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config, load_config
from segmentation.automatic import (
    AutomaticWireRecognizer, attach_terminals, attach_terminal_guided_boundaries,
)
from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary
from segmentation.wire import analyze_instance_masks, save_result


def wire(a=(50, 80), b=(250, 80)):
    mask = np.zeros((200, 300), np.uint8)
    cv2.line(mask, a, b, 1, 3)
    return mask.astype(bool)


def setup(masks=None):
    image = np.full((200, 300, 3), 128, np.uint8)
    masks = [wire()] if masks is None else masks
    for mask in masks:
        image[mask] = 20
    return analyze_instance_masks(image, masks, scores=[.9]*len(masks))


def terminal(rect=(44, 77, 62, 84), name='T1', confidence=.9):
    mask = np.zeros((200, 300), bool)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = True
    return {'id': name, 'confidence': confidence, 'bbox_xyxy': list(rect),
            'mask_file': f'terminal_{name}.png', 'confirmed': False}, mask


def test_correct_match_is_specific_instance_and_never_confirms_grasp():
    result = setup()
    detection, mask = terminal()
    before_paths = [p.copy() for p in result.paths_xy]
    before_candidates = copy.deepcopy(result.candidates)
    attach_terminals(result, [detection], [mask])
    report = detection['wire_association']
    assert report['status'] == 'matched' and report['instance_id'] == 'W001'
    assert report['semantic_evidence_only'] and report['score'] > .55
    assert report['matches'][0]['features']['axis_angle_deg'] == pytest.approx(0)
    assert not detection['confirmed'] and not result.grasp_plans
    assert len(result.candidates) == len(before_candidates)
    assert all(not c['confirmed'] for c in result.candidates)
    assert sum(bool(c['terminal_model_evidence']) for c in result.candidates) == 1
    assert all(np.array_equal(a, b) for a, b in zip(result.paths_xy, before_paths))


def test_two_terminals_on_separate_parallel_wires_match_independently():
    result = setup([wire(), wire((50, 110), (250, 110))])
    a, ma = terminal()
    b, mb = terminal((44, 107, 62, 114), 'T2')
    attach_terminals(result, [a, b], [ma, mb])
    assert a['wire_association']['status'] == b['wire_association']['status'] == 'matched'
    assert a['wire_association']['instance_id'] != b['wire_association']['instance_id']


def test_adjacent_terminal_detections_competing_for_one_end_are_ambiguous():
    result = setup()
    a, ma = terminal(confidence=.99)
    b, mb = terminal((63, 77, 78, 84), 'T2', confidence=.6)
    attach_terminals(result, [a, b], [ma, mb])
    for detection in (a, b):
        assert detection['wire_association']['status'] == 'ambiguous'
        assert 'multiple_terminals_compete_for_same_wire_end' in detection['wire_association']['rejection_reasons']
    assert not any(c['terminal_model_evidence'] for c in result.candidates)


def test_two_similarly_supported_instances_are_not_resolved_by_nearest_distance():
    result = setup([wire(), wire((50, 94), (250, 94))])
    detection, mask = terminal((45, 86, 65, 89))
    attach_terminals(result, [detection], [mask])
    report = detection['wire_association']
    assert report['status'] == 'ambiguous'
    assert not report['instance_id']
    assert 'multiple_instances_with_similar_support' in report['rejection_reasons']


def test_two_terminals_at_opposite_ends_of_same_wire_are_not_conflicting():
    result = setup()
    a, ma = terminal()
    b, mb = terminal((239, 77, 257, 84), 'T2')
    attach_terminals(result, [a, b], [ma, mb])
    assert all(d['wire_association']['status'] == 'matched' for d in (a, b))
    assert a['wire_association']['endpoint_index'] != b['wire_association']['endpoint_index']


def test_shiny_terminal_geometry_matches_but_brightness_does_not_change_decision():
    a, b = setup(), setup()
    d1, m1 = terminal()
    d2, m2 = terminal()
    b.image_bgr[m2] = 255
    attach_terminals(a, [d1], [m1])
    attach_terminals(b, [d2], [m2])
    x, y = d1['wire_association'], d2['wire_association']
    assert x['status'] == y['status'] == 'matched'
    assert x['score'] == y['score']
    assert y['appearance_auxiliary']['bright_achromatic_fraction'] == 1
    assert not y['appearance_auxiliary']['used_for_match_decision']


@pytest.mark.parametrize('rect,reason', [
    ((135, 77, 153, 84), 'terminal_not_near_wire_end'),
    ((44, 58, 50, 82), 'terminal_axis_not_aligned_with_wire'),
    ((20, 20, 40, 26), 'no_local_connection_support'),
    ((42, 73, 60, 91), 'terminal_axis_unreliable'),
])
def test_false_metal_or_reflection_is_not_accepted_from_brightness_or_confidence(rect, reason):
    result = setup()
    detection, mask = terminal(rect, confidence=.999)
    result.image_bgr[mask] = 255
    attach_terminals(result, [detection], [mask])
    report = detection['wire_association']
    assert report['status'] != 'matched'
    assert reason in report['matches'][0]['rejection_reasons']
    assert not any(c['terminal_model_evidence'] for c in result.candidates)


def test_crossing_wires_return_explicit_ambiguity_even_at_nearest_end():
    result = setup([wire(), wire((55, 25), (55, 150))])
    detection, mask = terminal()
    attach_terminals(result, [detection], [mask])
    assert detection['wire_association']['status'] == 'ambiguous'
    assert not attach_terminal_guided_boundaries(result, Config({'wire': {'auto': {'terminal_auto_boundary': True}}}))


def test_missed_yolo_detection_preserves_sam_and_shift_click_flow():
    result = setup()
    original = result.mask.copy()
    attach_terminals(result, [], [])
    apply_grasp_policy(result, Config())
    chosen = confirm_body_boundary(result, [52, 80], Config())
    plan = next(p for p in result.grasp_plans if p['candidate_id'] == chosen)
    assert plan['boundary_verification']['source'] == 'manual'
    assert plan['boundary_confirmed'] and np.array_equal(original, result.mask)
    assert any('端子模型未检出' in w for w in result.warnings)


def test_auto_boundary_default_off_and_opt_in_prediction_stays_unconfirmed():
    result = setup()
    detection, mask = terminal((60, 77, 78, 84))
    attach_terminals(result, [detection], [mask])
    before = len(result.candidates)
    assert not attach_terminal_guided_boundaries(result, Config())
    assert len(result.candidates) == before
    suggestions = attach_terminal_guided_boundaries(result, Config({'wire': {'auto': {'terminal_auto_boundary': True}}}))
    assert len(suggestions) == 1
    chosen = next(iter(suggestions))
    candidate = next(c for c in result.candidates if c['id'] == chosen)
    assert candidate['instance_id'] == 'W001' and not candidate['confirmed']
    apply_grasp_policy(result, Config(), suggestions)
    plan = next(p for p in result.grasp_plans if p['candidate_id'] == chosen)
    assert plan['boundary_verification']['source'] == 'model_prediction'
    assert not plan['boundary_verification']['human_confirmed']
    assert 'visual_path_boundary_not_manually_verified' in plan['readiness_blockers']


def test_saved_json_contains_every_match_feature_score_and_rejection(tmp_path):
    result = setup([wire(), wire((50, 130), (250, 130))])
    detection, mask = terminal()
    attach_terminals(result, [detection], [mask])
    out = save_result(result, tmp_path/'result')
    data = json.loads((out/'recognition.json').read_text())
    association = data['terminal_detections'][0]['wire_association']
    assert len(association['matches']) == 2
    assert all('score' in m and 'features' in m and 'rejection_reasons' in m for m in association['matches'])
    assert data['automation']['terminal_association']['schema_version'] == 1
    assert association['matches'][0]['features']['tangent_xy']


def test_roi_pipeline_associates_in_original_rgb_coordinates():
    image = np.full((300, 500, 3), 128, np.uint8)
    roi = [100, 50, 400, 250]
    cfg = Config({'wire': {'auto': {'roi_xyxy': roi, 'max_width_fraction': .05}}})
    recognizer = AutomaticWireRecognizer(cfg)
    recognizer._generate = lambda image: [{'segmentation': wire(), 'predicted_iou': .9}]
    detection, mask = terminal()
    recognizer._detect_terminals = lambda image: ([detection], [mask])
    result = recognizer.recognize(image)
    association = result.terminal_detections[0]['wire_association']
    assert association['status'] == 'matched'
    x, y = association['boundary_xy']
    assert 145 <= x <= 165 and y == pytest.approx(130, abs=1)
    assert result.automation['terminal_association']['coordinate_space'] == 'original_rgb_pixels'


@pytest.mark.parametrize('case', ['no_instances', 'missing_mask', 'wrong_shape', 'empty', 'duplicate_id'])
def test_missing_or_invalid_evidence_never_fabricates_match(case):
    result = setup()
    detection, mask = terminal()
    detections, masks = [detection], [mask]
    if case == 'no_instances':
        result.instances = []
    elif case == 'missing_mask':
        masks = []
    elif case == 'wrong_shape':
        masks = [np.ones((10, 10), bool)]
    elif case == 'empty':
        masks = [np.zeros_like(mask)]
    else:
        detections.append(dict(detection))
        masks.append(mask.copy())
    attach_terminals(result, detections, masks)
    assert all(d['wire_association']['status'] != 'matched' for d in detections)


def test_configuration_keeps_auto_boundary_and_terminal_seed_usage_off():
    cfg = load_config()
    assert cfg['wire.auto.terminal_auto_boundary'] is False
    assert cfg['wire.auto.terminal_use_seeds'] is False
    assert cfg['wire.grasp_policy.offset_mm'] == 14


def test_fragmented_reflection_mask_is_reported_ambiguous_near_wire():
    result = setup()
    detection, mask = terminal()
    mask[:, 53:55] = False
    attach_terminals(result, [detection], [mask])
    report = detection['wire_association']
    assert report['status'] == 'ambiguous'
    assert 'terminal_mask_fragmented' in report['rejection_reasons']


@pytest.mark.parametrize('confidence', [.1, float('nan'), float('inf')])
def test_invalid_or_low_confidence_cannot_be_rescued_by_geometry(confidence):
    result = setup()
    detection, mask = terminal(confidence=confidence)
    attach_terminals(result, [detection], [mask])
    report = detection['wire_association']
    assert report['status'] != 'matched'
    assert np.isfinite(report['matches'][0]['score'])
    assert any('confidence' in r for r in report['matches'][0]['rejection_reasons'])


def test_ambiguous_model_evidence_does_not_block_manual_sam_confirmation():
    result = setup()
    a, ma = terminal()
    b, mb = terminal((63, 77, 78, 84), 'T2')
    attach_terminals(result, [a, b], [ma, mb])
    apply_grasp_policy(result, Config())
    chosen = confirm_body_boundary(result, [52, 80], Config())
    plan = next(p for p in result.grasp_plans if p['candidate_id'] == chosen)
    assert plan['boundary_verification']['human_confirmed']
    assert plan['boundary_verification']['source'] == 'manual'
    assert not plan['executable']
