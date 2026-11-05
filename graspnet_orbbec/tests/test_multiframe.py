"""合成序列/模拟推理验证，不连接设备，也不把稳定率当准确率。"""

from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import Config
from arm.handeye_bridge import read_json
from segmentation import multiframe as mf
from segmentation.automatic import AutomaticWireRecognizer, attach_terminals
from segmentation.wire import analyze_instance_masks
from scripts.analyze_wire_sequence import parser
from test_stationary_sequence import rig, collect, write


def detection(x=20, y=55, width=10, height=10, cls=0):
    return {'id': 'T1', 'class_id': cls, 'confidence': .9,
            'bbox_xyxy': [x, y, x+width, y+height], 'confirmed': False, 'mask_file': 'terminal_T1.png'}


def records(detections):
    return [{'frame_id': f'f{i}', 'host_receive_timestamp_us': str(1000000+i*1000),
             'detections': ds, 'error': None, 'gray_mean_delta': 0} for i, ds in enumerate(detections)]


def test_stable_boxes_associate_despite_reversed_order():
    frames = records([[detection(20), detection(90)], [detection(91), detection(21)],
                      [detection(19), detection(89)], [detection(20), detection(90)]])
    before = deepcopy(frames)
    report = mf.analyze_detections(frames, 'f0')
    assert not report['rejection_reasons']
    assert len(report['tracks']) == 2
    for track in report['tracks']:
        assert track['detection_ratio'] == 1
        assert track['max_center_deviation_px'] == 1
        assert track['center_std_xy_px'][0] > 0
        assert track['status'] == 'experimental_stable'
    assert frames == before
    assert not report['statistics_used_for_geometry']


@pytest.mark.parametrize('missing,expected', [(1, 'experimental_stable'), (2, 'manual_confirmation_required')])
def test_missed_detections_use_all_frames_as_denominator(missing, expected):
    frames = records([[detection()] for _ in range(5)])
    for i in range(1, missing+1):
        frames[i]['detections'] = []
    track = mf.analyze_detections(frames, 'f0')['tracks'][0]
    assert track['detection_ratio'] == (5-missing)/5
    assert track['status'] == expected


def test_false_detection_is_separate_track_and_not_averaged():
    report = mf.analyze_detections(records([[detection()], [detection(), detection(120)], [detection()]]), 'f0')
    a, b = report['tracks']
    assert a['detection_ratio'] == 1 and a['max_center_deviation_px'] == 0
    assert b['detection_ratio'] == pytest.approx(1/3)
    assert 'absent_in_reference' in b['rejection_reasons']
    assert report['unmatched_boxes_averaged'] is False
    assert report['rejection_reasons']


def test_competing_neighbors_never_force_nearest_assignment():
    report = mf.analyze_detections(records([[detection(20), detection(28)], [detection(21), detection(29)]]), 'f0')
    assert all(t['ambiguous_frames'] == ['f1'] for t in report['tracks'])
    assert report['matching_events'][0]['matched'] == []
    assert len(report['matching_events'][0]['ambiguous_edges']) == 4


@pytest.mark.parametrize('position', [26, 80])
def test_moving_target_rejected_inside_or_outside_match_gate(position):
    report = mf.analyze_detections(records([[detection()], [detection(position)], [detection()]]), 'f0')
    assert report['rejection_reasons']
    if position == 26:
        assert 'center_unstable' in report['tracks'][0]['rejection_reasons']
    else:
        assert len(report['tracks']) == 2


def test_boundary_changes_without_center_movement():
    report = mf.analyze_detections(records([[detection(width=20)], [detection(x=14, width=32)]]), 'f0')
    track = report['tracks'][0]
    assert track['max_center_deviation_px'] == 0
    assert 'boundary_unstable' in track['rejection_reasons']


def test_missing_reference_target_and_wrong_class_require_manual():
    report = mf.analyze_detections(records([[], [detection()], [detection()]]), 'f0')
    assert 'absent_in_reference' in report['tracks'][0]['rejection_reasons']
    report = mf.analyze_detections(records([[detection(cls=0)], [detection(cls=1)]]), 'f0')
    assert len(report['tracks']) == 2


def test_illumination_change_and_model_error_are_reported():
    frames = records([[detection()] for _ in range(3)])
    frames[1]['gray_mean_delta'] = 40
    frames[2].update(error='model failed', detections=[])
    report = mf.analyze_detections(frames, 'f0')
    assert 'illumination_inconsistent' in report['rejection_reasons']
    assert 'per_frame_detection_failed' in report['rejection_reasons']
    assert report['tracks'][0]['detection_ratio'] == pytest.approx(2/3)


@pytest.mark.parametrize('bad', ['duplicate', 'backward', 'negative'])
def test_timestamp_anomalies(bad):
    frames = records([[detection()] for _ in range(3)])
    frames[1]['host_receive_timestamp_us'] = {'duplicate': '1000000', 'backward': '999999', 'negative': '-1'}[bad]
    with pytest.raises(ValueError, match='时间戳'):
        mf.analyze_detections(frames, 'f0')


@pytest.mark.parametrize('settings', [{'min_detection_ratio': 1.1}, {'wire_check_frames': 3},
                                      {'max_center_deviation_px': float('nan')},
                                      {'max_center_deviation_px': 0}, {'wire_check_frames': True}])
def test_bad_experimental_config(settings):
    with pytest.raises(ValueError):
        mf.StabilityConfig(**settings)


def wire_result(image=None, y=60, two=False):
    image = np.zeros((120, 160, 3), np.uint8) if image is None else image
    masks = []
    for yy in ([y, y+30] if two else [y]):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.line(mask, (30, yy), (130, yy), 1, 3)
        masks.append(mask.astype(bool))
    return analyze_instance_masks(image, masks, scores=[.9]*len(masks))


def test_wire_instances_match_geometry_not_local_ids():
    a, b = wire_result(two=True), wire_result(y=61, two=True)
    b.instances.reverse()
    for instance in b.instances:
        instance.instance_id += '_other_frame'
    report = mf.compare_wire_instances(a, b, mf.StabilityConfig())
    assert report['status'] == 'consistent' and len(report['matches']) == 2


@pytest.mark.parametrize('fault', ['moved', 'crossing', 'broken', 'missing'])
def test_wire_position_and_instance_continuity(fault):
    a, b = wire_result(), wire_result(y=68 if fault == 'moved' else 60)
    if fault == 'crossing':
        b.instances[0].ambiguity_reasons.append('crossing_direction_unresolved')
    elif fault == 'broken':
        path = b.instances[0].paths_xy[0]
        b.instances[0].paths_xy[0] = np.concatenate([path[:20], path[40:]])
    elif fault == 'missing':
        b.instances = []
    assert mf.compare_wire_instances(a, b, mf.StabilityConfig())['status'] == 'manual_confirmation_required'


@pytest.fixture
def pipeline(rig, tmp_path):
    root = collect(rig, tmp_path/'sequence')
    seen = {'detector': [], 'sam': []}

    def detector(image):
        seen['detector'].append(image.copy())
        mask = np.zeros(image.shape[:2], bool)
        mask[58:62, 20:30] = True
        return [detection(y=58, height=4)], [mask]

    def recognizer(frame_id, image, detections, masks):
        seen['sam'].append(frame_id)
        result = wire_result(image)
        attach_terminals(result, deepcopy(detections), [m.copy() for m in masks])
        return result

    def run(**kwargs):
        output = kwargs.pop('output', tmp_path/'analysis')
        return mf.analyze_sequence(rig.bundle, root, output, Config(), allow_simulated=True,
                                   _detector=kwargs.pop('_detector', detector),
                                   _recognizer=kwargs.pop('_recognizer', recognizer), **kwargs)

    return root, seen, run


def test_pipeline_independent_yolo_reference_only_and_ab(pipeline):
    root, seen, run = pipeline
    entries = read_json(root/'manifest.json')['frames']
    reference = entries[2]
    report = run(reference_id=reference['frame_id'])
    assert len(seen['detector']) == 5
    assert len(seen['sam']) == 2 and seen['sam'][0] == reference['frame_id']
    assert report['status'] == 'experimental_consistent_manual_boundary_required'
    assert report['reference']['snapshot'] == str(root/reference['snapshot'])
    assert report['reference']['pixel_source_frame_id'] == report['reference']['pose_source_frame_id']
    assert not report['reference']['cross_frame_pixels_used_for_geometry']
    ab = report['ab_comparison']
    assert ab['single_frame']['terminal_count'] == ab['multi_frame']['experimental_stable_tracks'] == 1
    assert ab['accuracy_measured'] is False and ab['multi_frame']['sam_frames'] == 2
    assert not report['executable'] and not report['planning_allowed']
    assert len(report['timing_s']['yolo_per_frame']) == 5
    assert len(report['timing_s']['sam_and_instances_per_frame']) == 2
    recognition = read_json(root.parent/'analysis/reference/recognition.json')
    assert not recognition['grasp_policy']['planning_allowed']
    assert all(not p['executable'] for p in recognition['grasp_plans'])
    assert len(report['tracks'][0]['mask_boundary_changes']) == 5


def test_missing_sequence_frame_blocks_before_inference(pipeline):
    root, seen, run = pipeline
    (root/'frames/frame_001/capture/rgb.png').unlink()
    report = run()
    assert report['rejection_reasons'] == ['sequence_validation_failed']
    assert not seen['detector'] and not seen['sam']


def test_manifest_timestamp_tampering_blocks_before_inference(pipeline):
    root, seen, run = pipeline
    manifest = read_json(root/'manifest.json')
    manifest['frames'][1]['host_receive_timestamp_us'] = manifest['frames'][0]['host_receive_timestamp_us']
    write(root/'manifest.json', manifest)
    assert run()['rejection_reasons'] == ['sequence_validation_failed']
    assert not seen['detector']


def test_simulated_data_forbidden_without_opt_in(pipeline, rig, tmp_path):
    root, seen, run = pipeline
    report = mf.analyze_sequence(rig.bundle, root, tmp_path/'refuse', Config())
    assert report['rejection_reasons'] == ['sequence_validation_failed']
    assert not seen['detector']


def test_zero_wire_checks_remains_manual(pipeline):
    root, seen, run = pipeline
    report = run(stability=mf.StabilityConfig(wire_check_frames=0))
    assert len(seen['sam']) == 1
    assert 'cross_frame_wire_validation_required' in report['rejection_reasons']


def test_bad_reference_and_simulated_geometry_refused(pipeline, tmp_path):
    root, seen, run = pipeline
    report = run(reference_id='missing')
    assert report['rejection_reasons'] == ['reference_frame_not_in_sequence']
    assert not seen['detector']
    called = []
    report = run(output=tmp_path/'geometry', _geometry=lambda result, snapshot: called.append(snapshot))
    assert not called
    assert '模拟序列' in report['sam_or_geometry_error']


def test_inference_error_falls_back_without_rerunning_yolo(pipeline):
    root, seen, run = pipeline

    def fail(image):
        seen['detector'].append(image.copy())
        raise ValueError('mock detector failed')

    report = run(_detector=fail)
    assert len(seen['detector']) == 5
    assert 'per_frame_detection_failed' in report['rejection_reasons']
    assert report['status'] == 'manual_confirmation_required'


def test_other_frame_pixels_cannot_use_reference_pose(pipeline):
    root, seen, run = pipeline

    def wrong(frame_id, image, detections, masks):
        return wire_result(image+1)

    report = run(_recognizer=wrong)
    assert 'SAM2输出原图' in report['sam_or_geometry_error']
    assert report['status'] == 'manual_confirmation_required'


def test_inference_time_file_tampering_blocks_geometry(pipeline):
    root, seen, run = pipeline
    called = []

    def tamper(frame_id, image, detections, masks):
        (root/'frames/frame_000/capture/frames.json').write_text('{}')
        return wire_result(image)

    report = run(_recognizer=tamper, _geometry=lambda *args: called.append(args))
    assert not called and report['status'] == 'manual_confirmation_required'


def test_cached_yolo_roi_compatible_and_never_reruns_detector(monkeypatch):
    image = np.zeros((120, 160, 3), np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[55:65, 20:30] = True
    cfg = Config({'wire': {'auto': {'roi_xyxy': [10, 20, 150, 110], 'terminal_auto_boundary': False}}})
    recognizer = mf.ReferenceRecognizer(cfg, image, [detection()], [mask])
    monkeypatch.setattr(recognizer, '_generate', lambda image: [])
    result = recognizer.recognize(image)
    assert result.terminal_detections[0]['bbox_xyxy'] == detection()['bbox_xyxy']
    assert np.array_equal(result.terminal_masks[0], mask)
    with pytest.raises(ValueError, match='不对应'):
        recognizer.recognize(image+1)


def test_cli_has_no_camera_or_execute_options():
    args = parser().parse_args(['--sequence', 'somewhere'])
    assert not args.geometry_preview
    with pytest.raises(SystemExit):
        parser().parse_args(['--sequence', 'somewhere', '--execute'])


def test_experimental_threshold_is_configurable():
    frames = records([[detection(20)], [detection(23)]])
    assert not mf.analyze_detections(frames, 'f0')['rejection_reasons']
    report = mf.analyze_detections(frames, 'f0', mf.StabilityConfig(max_center_deviation_px=2))
    assert 'center_unstable' in report['tracks'][0]['rejection_reasons']


def test_geometry_receives_only_selected_reference_snapshot(pipeline, monkeypatch):
    root, seen, run = pipeline
    manifest = read_json(root/'manifest.json')
    reference = manifest['frames'][3]
    # 仅此单元测试模拟验证器的物理数据合同，真实入口仍强制完整验证。
    manifest['data_origin'] = 'physical_camera'
    write(root/'manifest.json', manifest)
    monkeypatch.setattr(mf, 'verify_stationary_sequence', lambda *args, **kwargs: {})
    called = []
    report = run(reference_id=reference['frame_id'],
                 _geometry=lambda result, snapshot: called.append(snapshot))
    assert called == [root/reference['snapshot']]
    assert report['reference']['pixel_source_frame_id'] == reference['frame_id']
    assert report['reference']['pose_source_frame_id'] == reference['frame_id']
    assert not report['executable']


def test_failed_sam_time_is_reported(pipeline):
    root, seen, run = pipeline

    def fail(*args):
        raise ValueError('simulated SAM failure')

    report = run(_recognizer=fail)
    assert 'reference_or_wire_analysis_failed' in report['rejection_reasons']
    assert len(report['timing_s']['sam_and_instances_per_frame']) == 1


def test_full_image_terminal_mask_remains_reviewable(pipeline):
    root, seen, run = pipeline
    report = run(_detector=lambda image: ([detection()], [np.ones(image.shape[:2], bool)]))
    assert len(report['tracks'][0]['mask_boundary_changes']) == 5
    assert report['status'] == 'manual_confirmation_required'
