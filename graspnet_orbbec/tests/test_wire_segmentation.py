"""验证真实掩码几何、提示坐标和异步快照隔离；不依赖网络或机械臂。"""

import json
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config
from segmentation.wire import SegmentationError, Sam2WireSegmenter, analyze_mask, save_result
from segmentation.wire_gui import PromptState, SegmentationWorker, panel_point, render_canvas


def scene():
    image = np.full((100, 200, 3), 128, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.line(mask, (20, 50), (180, 50), 1, 3)
    return image, mask


def test_line_candidates_are_inside_mask_with_pixel_units():
    image, mask = scene()
    result = analyze_mask(image, mask, inset_px=12)
    assert len(result.candidates) == 2
    for item in result.candidates:
        x, y = item['near_tip_xy']
        assert mask[y, x]
        assert item['inset_px'] >= 12
        assert np.linalg.norm(item['tangent_xy']) == pytest.approx(1)
        assert item['confirmed'] is False
    assert not result.to_dict()['robot_pose_valid']
    assert not result.to_dict()['terminal_semantics_verified']
    for c, region in zip(result.candidates, result.end_regions):
        assert not region[mask == 0].any()
        x, y = c['near_tip_xy']
        assert region[y, x]
        assert region.sum() < mask.sum() / 2


def test_boundary_cut_is_not_a_terminal():
    image, mask = scene()
    cv2.line(mask, (0, 50), (30, 50), 1, 3)
    result = analyze_mask(image, mask)
    assert len(result.candidates) == 1
    assert result.candidates[0]['tip_xy'][0] > 150
    assert any('截断' in w for w in result.warnings)


def test_crop_cut_is_also_not_a_terminal():
    image, mask = scene()
    result = analyze_mask(image, mask, crop_xyxy=[18, 0, 200, 100])
    assert len(result.candidates) == 1


def test_disconnected_visible_legs_keep_both_visible_tips():
    image, mask = scene()
    mask[:] = 0
    cv2.line(mask, (0, 30), (150, 30), 1, 3)
    cv2.line(mask, (0, 70), (180, 70), 1, 3)
    result = analyze_mask(image, mask)
    assert len(result.candidates) == 2
    assert len(result.paths_xy) == 2
    assert result.centerline_xy.size == 0
    assert {c['component_id'] for c in result.candidates} == {1, 2}
    assert all(c['tip_xy'][0] > 140 for c in result.candidates)
    assert any('不连通' in w for w in result.warnings)


def test_closed_loop_and_tiny_masks_do_not_fabricate_endpoints():
    image, mask = scene()
    mask[:] = 0
    cv2.circle(mask, (100, 50), 25, 1, 3)
    result = analyze_mask(image, mask)
    assert not result.candidates
    assert result.warnings
    with pytest.raises(SegmentationError):
        analyze_mask(image, np.zeros_like(mask))
    with pytest.raises(SegmentationError):
        analyze_mask(image, mask, inset_px=-1)


def test_branching_is_explicitly_ambiguous():
    image, mask = scene()
    cv2.line(mask, (100, 50), (100, 10), 1, 3)
    result = analyze_mask(image, mask)
    assert any('端点' in w for w in result.warnings)
    assert len(result.candidates) == 3
    assert all(not c['confirmed'] for c in result.candidates)


def test_crossed_wires_preserve_four_visible_end_candidates():
    image = np.zeros((200,200,3),np.uint8)
    mask = np.zeros((200,200),np.uint8)
    cv2.line(mask,(20,100),(180,100),1,3)
    cv2.line(mask,(100,20),(100,180),1,3)
    result=analyze_mask(image,mask)
    assert len(result.candidates)==4
    for c in result.candidates:
        x,y=c['near_tip_xy']
        assert mask[y,x] and not c['confirmed']
    assert any('实例对应' in w for w in result.warnings)


def test_saved_mask_is_native_binary_and_metadata_stays_unverified(tmp_path):
    image, mask = scene()
    result = analyze_mask(image, mask)
    out = save_result(result, tmp_path / 'result')
    saved = cv2.imread(str(out / 'mask.png'), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(saved)) == {0, 255}
    assert np.array_equal(saved > 0, mask > 0)
    assert np.array_equal(cv2.imread(str(out / 'color.png')), image)
    assert not json.loads((out / 'recognition.json').read_text())['robot_pose_valid']
    assert np.array_equal(cv2.imread(str(out / 'end_region_A.png'), 0) > 0, result.end_regions[0])
    with pytest.raises(FileExistsError):
        save_result(result, out)


class Predictor:
    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.prompts = kwargs
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.line(mask, (5, h // 2), (w - 1, h // 2), 1, 3)
        return np.stack([np.ones_like(mask), mask]), np.array([0.99, 0.8]), None


def fake_model():
    model = Sam2WireSegmenter(Config({'wire': {'crop_margin_px': 0}}))
    model.predictor = Predictor()
    model.device = 'cpu'
    return model


def test_crop_prompts_and_mask_return_to_original_coordinates():
    image, _ = scene()
    model = fake_model()
    result = model.segment(image, box=[40, 20, 160, 80], points=[[100, 50]], labels=[1])
    assert model.predictor.image.shape[:2] == (61, 121)
    assert np.array_equal(model.predictor.prompts['point_coords'], [[60, 30]])
    assert np.array_equal(model.predictor.prompts['box'], [0, 0, 120, 60])
    assert result.mask.shape == image.shape[:2]
    assert result.mask[50, 100]
    assert result.mask.sum() < 1000  # 高分整桌面掩码必须被排除。
    assert not result.mask[:, :40].any()


def test_fractional_border_prompt_cannot_index_outside_image():
    image, _ = scene()
    result = fake_model().segment(image, points=[[199.9, 50]], labels=[1])
    assert result.mask[50, 199]


@pytest.mark.parametrize('points,labels', [([[-1, 4]], [1]), ([[200, 4]], [1]),
                                         ([[50, 50]], [2]), ([[float('nan'), 1]], [1])])
def test_invalid_prompts_fail_before_model(points, labels):
    image, _ = scene()
    with pytest.raises(SegmentationError):
        fake_model().segment(image, points=points, labels=labels)


def test_negative_seed_rejects_contradictory_mask():
    image, _ = scene()
    with pytest.raises(SegmentationError):
        fake_model().segment(image, points=[[100, 50], [110, 50]], labels=[1, 0])


def test_low_score_background_is_rejected():
    image, _ = scene()
    model = fake_model()
    original = model.predictor.predict

    def low_score(**kwargs):
        masks, _, logits = original(**kwargs)
        return masks, np.array([0.1, 0.2]), logits

    model.predictor.predict = low_score
    with pytest.raises(SegmentationError):
        model.segment(image, box=[0, 0, 199, 99])


def test_old_result_is_rejected_after_clear_or_new_prompt():
    image, mask = scene()
    result = analyze_mask(image, mask)
    state = PromptState(source=image.copy())
    old = state.revision
    state.clear()
    assert not state.accept(old, result, None)
    assert state.result is None and state.source is None
    assert state.accept(state.revision, result, None)
    state.invalidate()
    assert not state.accept(old + 1, result, None)


def test_canvas_uses_result_snapshot_not_live_image():
    image, mask = scene()
    result = analyze_mask(image, mask)
    state = PromptState(source=image.copy(), result=result)
    canvas = render_canvas(np.zeros_like(image), state)
    assert np.array_equal(canvas[64 + 5, 200 + 5], image[5, 5])
    assert panel_point(199, 64, 200, 100) == (199, 0)
    assert panel_point(200, 64, 200, 100) is None
    assert panel_point(10, 63, 200, 100) is None


def test_worker_copies_snapshot_and_waits_for_inflight_job():
    image, mask = scene()
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    class Segmenter:
        def segment(self, frame, **kwargs):
            entered.set()
            assert release.wait(3)
            assert np.array_equal(frame, np.full_like(frame, 128))
            return analyze_mask(frame, mask)

    worker = SegmentationWorker(Config(), segmenter=Segmenter())
    worker.submit(7, image, None, [[20, 50]], [1])
    image[:] = 0
    worker.start()
    assert entered.wait(3)
    stopper = threading.Thread(target=lambda: (worker.stop(), done.set()))
    stopper.start()
    try:
        assert not done.wait(0.03)
    finally:
        release.set()
        stopper.join(3)
    assert done.is_set()
    revision, result, error = worker.results.get_nowait()
    assert revision == 7 and result is not None and error is None
