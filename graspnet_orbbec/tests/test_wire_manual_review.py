"""手动线身纠正、独立复核和最终拒绝检查；只使用文件/模拟窗口及predictor。"""

from copy import deepcopy
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from segmentation.grasp_policy import apply_grasp_policy, confirm_body_boundary, confirm_manual_mask
from segmentation.path_validation import execution_path_review, manual_mask_review
from segmentation.wire import Sam2WireSegmenter, analyze_mask, save_result
from test_visual_path_gate import setup_target, get_plan


def manual_target():
    result, _, bridge = setup_target()
    result.automation = {'method': 'manual_prompts', 'manual_prompts': True,
                         'coordinate_space': 'original_rgb_pixels',
                         'prompts': {'box': None, 'points': [[120., 60.]], 'labels': [1]}}
    apply_grasp_policy(result, Config())
    return result, bridge


def test_shift_does_not_replace_manual_mask_review():
    result, bridge = manual_target()
    chosen = confirm_body_boundary(result, [218, 60], Config())
    bridge.apply(result, Config())
    plan = get_plan(result.to_dict(), chosen)
    assert plan['boundary_confirmed'] and plan['visual_path_validation']['passed']
    assert plan['offset_mm'] == 14
    assert 'visual_path_manual_mask_review_required' in plan['readiness_blockers']
    # 即使上游遗漏阻断，最终层仍须从复核记录及实例掩码独立拒绝。
    plan['readiness_blockers'] = []
    review = execution_path_review(result.to_dict(), plan)
    assert not review['passed']
    assert 'visual_path_manual_mask_review_required' in review['rejection_reasons']


def reviewed_target():
    result, bridge = manual_target()
    assert confirm_manual_mask(result, Config()) == result.instances[0].instance_id
    chosen = confirm_body_boundary(result, [218, 60], Config())
    bridge.apply(result, Config())
    return result, chosen


def test_review_preserves_geometry_and_saves_bound_evidence(tmp_path):
    result, chosen = reviewed_target()
    plan = get_plan(result.to_dict(), chosen)
    assert manual_mask_review(result)['passed']
    assert plan['readiness_blockers'] == []
    assert execution_path_review(result.to_dict(), plan)['passed']
    assert plan['selection_v2']['offset_m'] == pytest.approx(.014)
    assert not result.grasp_policy['motion_commands_enabled'] and not plan['executable']
    import json
    saved = json.loads((save_result(result, tmp_path/'review')/'recognition.json').read_text())
    assert saved['grasp_policy']['mask_origin'] == 'manual_prompts'
    assert saved['automation']['prompts']['points'] == [[120., 60.]]
    assert saved['automation']['manual_mask_review']['image_rgb_sha256'] == hashlib.sha256(result.image_bgr.tobytes()).hexdigest()
    assert execution_path_review(saved, get_plan(saved, chosen))['passed']


@pytest.mark.parametrize('fault', ['cross', 'branch', 'break', 'parallel', 'forged_geometry'])
def test_review_cannot_clear_instance_or_topology_ambiguity(fault):
    result, _ = manual_target()
    mask = result.mask.astype(np.uint8)
    if fault in ('cross', 'branch', 'forged_geometry'):
        cv2.line(mask, (120, 30), (120, 90 if fault != 'branch' else 60), 1, 3)
    elif fault == 'break':
        mask[:, 118:122] = 0
    else:
        cv2.line(mask, (20, 85), (220, 85), 1, 3)
    if fault == 'forged_geometry':
        result.instances[0].mask = mask > 0
    else:
        result = analyze_mask(result.image_bgr, mask)
        result.automation = {'method': 'manual_prompts'}
    before = [list(i.ambiguity_reasons) for i in result.instances]
    with pytest.raises(ValueError):
        confirm_manual_mask(result, Config())
    assert [i.ambiguity_reasons for i in result.instances] == before
    assert not manual_mask_review(result)['passed']


@pytest.mark.parametrize('fault', ['image', 'mask', 'new_result', 'bad_record'])
def test_review_invalidates_on_new_mask_or_image(fault):
    result, _ = reviewed_target()
    if fault == 'image':
        result.image_bgr[0, 0, 0] ^= 1
    elif fault == 'mask':
        result.instances[0].mask[10, 10] = True
    elif fault == 'new_result':
        result.automation.pop('manual_mask_review')
    else:
        result.automation['manual_mask_review'] = ['invalid']
    assert not manual_mask_review(result)['passed']
    apply_grasp_policy(result, Config())
    assert all('visual_path_manual_mask_review_required' in p['readiness_blockers'] for p in result.grasp_plans)


@pytest.mark.parametrize('fault', ['missing', 'human_false', 'instance', 'image_hash', 'mask_hash',
                                 'mask_origin', 'summary', 'outside_local_patch', 'large_shape',
                                 'float_shape', 'unordered_runs', 'record_type'])
def test_final_layer_rechecks_manual_evidence(fault):
    result, chosen = reviewed_target()
    data = deepcopy(result.to_dict())
    record = data['automation']['manual_mask_review']
    package = data['grasp_policy']['instance_mask_evidence']
    if fault == 'missing':
        del data['automation']['manual_mask_review']
    elif fault == 'human_false':
        record['human_reviewed'] = False
    elif fault == 'instance':
        record['instance_id'] = 'WRONG'
    elif fault in ('image_hash', 'mask_hash'):
        record['image_rgb_sha256' if fault == 'image_hash' else 'mask_sha256'] = '0'*64
    elif fault == 'mask_origin':
        data['grasp_policy']['mask_origin'] = 'line_prompts'
    elif fault == 'summary':
        data['grasp_policy']['manual_mask_review']['passed'] = False
    elif fault == 'outside_local_patch':
        # 局部证据未变，整张实例掩码与人工审核的绑定仍应发现变化。
        package['masks'][0]['runs'].insert(0, [0, 1])
    elif fault == 'large_shape':
        package['shape_hw'] = [10**10, 10**10]
    elif fault == 'float_shape':
        package['shape_hw'] = [120., 240.]
    elif fault == 'unordered_runs':
        package['masks'][0]['runs'].reverse()
    else:
        data['automation']['manual_mask_review'] = ['bad']
    review = execution_path_review(data, get_plan(data, chosen))
    assert not review['passed']
    assert 'visual_path_manual_mask_review_required' in review['rejection_reasons']


def test_automatic_mode_does_not_require_manual_mask_review():
    result, chosen, bridge = setup_target()
    bridge.apply(result, Config())
    assert manual_mask_review(result) == {'required': False, 'passed': True, 'rejection_reasons': []}
    assert execution_path_review(result.to_dict(), get_plan(result.to_dict(), chosen))['passed']
    with pytest.raises(ValueError, match='V用于'):
        confirm_manual_mask(result, Config())


@pytest.mark.parametrize('review_first', [True, False])
def test_handeye_window_accepts_prompts_v_and_shift_on_same_snapshot(monkeypatch, review_first):
    import segmentation.wire_gui as gui

    original, _, bridge = setup_target()
    image, mask = original.image_bgr.copy(), original.mask
    calls, callbacks, states = [], [], []

    class Predictor:
        def set_image(self, crop):
            assert crop.shape == image.shape
        def predict(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return np.array([mask]), np.array([.9]), None

    model = Sam2WireSegmenter(Config())
    model.predictor, model.device = Predictor(), 'cpu'

    class Camera:
        closed = False
        def open(self): pass
        def close(self): self.closed = True
        def read(self): return NS(color_bgr=image.copy())

    class Recognizer:
        def prepare_terminal_runtime(self): pass

    camera = Camera()
    worker = gui.SegmentationWorker(Config(), segmenter=model, auto_recognizer=Recognizer())
    monkeypatch.setattr(gui, 'FileCamera', lambda *args, **kwargs: camera)
    monkeypatch.setattr(gui, 'SegmentationWorker', lambda cfg: worker)
    for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
        monkeypatch.setattr(gui.cv2, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.cv2, 'setMouseCallback', lambda name, cb: callbacks.append(cb))
    monkeypatch.setattr(gui.cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(gui, 'render_canvas', lambda live, state: states.append(state) or live)

    def wait_result():
        deadline = time.monotonic()+5
        while worker.results.empty() and time.monotonic() < deadline:
            time.sleep(.005)
        assert not worker.results.empty()

    step = [0]
    def key(delay):
        step[0] += 1
        mouse = callbacks[0]
        if step[0] == 1:
            mouse(cv2.EVENT_LBUTTONDOWN, 120, 124, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 120, 124, 0, None)
            wait_result()
        elif step[0] == 2:
            assert states[-1].result.automation['method'] == 'manual_prompts'
            mouse(cv2.EVENT_RBUTTONDOWN, 120, 99, 0, None)
            wait_result()
        elif step[0] == 3:
            return ord('v') if review_first else -1
        elif step[0] == 4:
            mouse(cv2.EVENT_LBUTTONDOWN, 218, 124, cv2.EVENT_FLAG_SHIFTKEY, None)
            mouse(cv2.EVENT_LBUTTONUP, 218, 124, cv2.EVENT_FLAG_SHIFTKEY, None)
        else:
            return ord('q')
        return -1

    monkeypatch.setattr(gui.cv2, 'waitKey', key)
    args = NS(from_dir='mock-file-camera', no_target=True, wire_auto=False, wire_box=None,
              wire_point=[], wire_seconds=0, wire_preview=None)
    assert gui.run(Config(), args, NS(stop_flag=False), handeye_bridge=bridge) == 0
    assert len(calls) == 2
    assert calls[1]['point_labels'].tolist() == [1, 0]
    assert calls[1]['point_coords'].tolist() == [[120, 60], [120, 35]]
    result = states[-1].result
    assert np.array_equal(result.image_bgr, image)
    plan = next(p for p in result.grasp_plans if p['boundary_confirmed'])
    assert plan['status'] == 'base_frame_preview_ready'
    assert plan['offset_mm'] == 14 and not plan['executable']
    assert manual_mask_review(result)['passed'] is review_first
    assert ('visual_path_manual_mask_review_required' in plan['readiness_blockers']) is not review_first
    assert camera.closed and not worker.is_alive()
