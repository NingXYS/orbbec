"""自动SAM2逐候选正负点合同；仅用合成图和模拟predictor，不加载模型或硬件。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config, load_config
from segmentation.automatic import (
    AutomaticWireRecognizer, automatic_background_points, line_contrast,
    predict_automatic_prompts,
)


def scene():
    image = np.full((120, 240, 3), 128, np.uint8)
    cv2.line(image, (10, 60), (230, 60), (10,)*3, 3)
    return image


def config(**options):
    return Config({'wire': {'auto': {'negative_prompts': {
        'enabled': True, 'coordinate_space': 'roi',
        'trusted_regions': [[0, 0, 240, 120]],
        'max_distance_px': 40, **options}}}})


def points():
    return np.array([[40, 60], [200, 60]], np.float32)


class Predictor:
    def __init__(self, *, bad_negative=False, reject_negative=False, remove_positive=False):
        self.calls = []
        self.bad_negative = bad_negative
        self.reject_negative = reject_negative
        self.remove_positive = remove_positive

    def set_image(self, image):
        self.image = image.copy()

    def predict(self, point_coords, point_labels, multimask_output):
        assert point_coords.ndim == 3 and point_coords.shape[2] == 2
        assert point_labels.shape == point_coords.shape[:2]
        assert point_coords.dtype == np.float32 and point_labels.dtype == np.int32
        assert np.all(point_labels[:, 0] == 1)
        assert np.all(point_labels[:, 1:] == 0)
        self.calls.append((point_coords.copy(), point_labels.copy()))
        mixed = point_coords.shape[1] > 1
        if mixed and self.reject_negative:
            raise ValueError('模拟错误负点导致请求被拒绝')
        masks = np.zeros((len(point_coords), 1, 120, 240), bool)
        for i, coords in enumerate(point_coords):
            x, y = map(int, coords[0])
            masks[i, 0, 59:62, 10:231] = True
            masks[i, 0, y, x] = not (mixed and self.remove_positive)
            if mixed and self.bad_negative:
                nx, ny = map(int, coords[1])
                masks[i, 0, ny, nx] = True
        scores = np.full((len(point_coords), 1), .9)
        # 与仓库SAM2相同，单批次时predict会squeeze第0维。
        return (masks[0], scores[0], None) if len(masks) == 1 else (masks, scores, None)


def test_single_wire_negatives_are_local_and_not_broadcast():
    image, positive = scene(), points()
    groups, info = automatic_background_points(image, positive, config())
    contrast, _ = line_contrast(image, Config())
    assert len(groups) == 2 and all(len(g) == 2 for g in groups)
    assert groups[0] != groups[1]
    for pos, group in zip(positive, groups):
        for x, y in group:
            assert contrast[y, x] < 6
            assert abs(y-60) > 6
            assert 12 <= np.linalg.norm(pos-[x, y]) <= 40
    assert all(c['point_labels'] == [1, 0, 0] for c in info['candidates'])


def test_background_texture_and_second_wire_are_not_used_as_negatives():
    image = scene()
    cv2.line(image, (10, 40), (230, 40), (245,)*3, 3)
    for y in range(0, 25, 4):
        image[y:y+2, :] = 0
    groups, _ = automatic_background_points(image, points(), config())
    for group in groups:
        for x, y in group:
            assert y > 25 and abs(y-40) > 6 and abs(y-60) > 6


def test_roi_local_and_rgb_trust_regions_give_identical_points():
    roi = [100, 200, 340, 320]
    rgb_cfg = config(coordinate_space='original_rgb', trusted_regions=[[100, 200, 340, 320]])
    a, info = automatic_background_points(scene(), points(), rgb_cfg, roi)
    b, _ = automatic_background_points(scene(), points(), config(), roi)
    assert a == b
    for candidate, group in zip(info['candidates'], a):
        assert candidate['negative_rgb_xy'] == (np.asarray(group)+[100, 200]).tolist()
    assert info['candidates'][0]['positive_rgb_xy'] == [140, 260]


@pytest.mark.parametrize('regions', [[], [[0, 58, 240, 63]], [[300, 0, 400, 120]]])
def test_no_reliable_region_or_wrong_region_on_wire_falls_back(regions):
    groups, info = automatic_background_points(scene(), points(), config(trusted_regions=regions))
    assert groups == [[], []]
    assert all(c['fallback_reason'] == 'no_reliable_background' for c in info['candidates'])


def test_disabled_mode_ignores_experimental_region_settings():
    groups, info = automatic_background_points(scene(), points(), config(
        enabled=False, trusted_regions='ignored', max_points=-1))
    assert groups == [[], []] and not info['enabled']


def test_variable_prompt_counts_batch_without_padding_or_cross_candidate_labels():
    positive = np.array([[40, 60], [120, 60], [200, 60]], np.float32)
    cfg = config(trusted_regions=[[0, 0, 80, 120]])
    groups, info = automatic_background_points(scene(), positive, cfg)
    assert len(groups[0]) == 2 and not groups[1] and not groups[2]
    predictor = Predictor()
    records = predict_automatic_prompts(predictor, scene().shape, positive, groups, 8, .5, info)
    assert len(records) == 3
    assert [x[0].shape for x in predictor.calls] == [(1, 3, 2), (2, 1, 2)]
    assert info['mixed_candidates'] == 1 and info['positive_only_candidates'] == 2


@pytest.mark.parametrize('failure', ['bad_negative', 'reject_negative', 'remove_positive'])
def test_wrong_mixed_prediction_retries_only_positive(failure):
    positive = points()[:1]
    groups, info = automatic_background_points(scene(), positive, config())
    predictor = Predictor(**{failure: True})
    records = predict_automatic_prompts(predictor, scene().shape, positive, groups, 8, .5, info)
    assert len(records) == 1
    assert [x[0].shape for x in predictor.calls] == [(1, 3, 2), (1, 1, 2)]
    assert info['candidates'][0]['final_mode'] == 'positive_only'
    assert info['candidates'][0]['fallback_reason']
    assert info['candidates'][0]['final_point_labels'] == [1]


def test_old_positive_only_mode_preserves_batch_and_records():
    groups, info = automatic_background_points(scene(), points(), Config())
    predictor = Predictor()
    records = predict_automatic_prompts(predictor, scene().shape, points(), groups, 8, .5, info)
    assert len(records) == 2 and len(predictor.calls) == 1
    coords, labels = predictor.calls[0]
    assert np.array_equal(coords, points()[:, None, :])
    assert np.array_equal(labels, np.ones((2, 1), np.int32))


def test_missing_output_for_one_candidate_does_not_retry_successful_neighbor():
    positive = points()
    groups, info = automatic_background_points(scene(), positive, config())

    class OneBad(Predictor):
        def predict(self, **kwargs):
            masks, scores, logits = super().predict(**kwargs)
            if len(kwargs['point_coords']) == 2:
                masks[1] = False
            return masks, scores, logits

    predictor = OneBad()
    records = predict_automatic_prompts(predictor, scene().shape, positive, groups, 8, .5, info)
    assert len(records) == 2
    assert len(predictor.calls[1][0]) == 1
    assert predictor.calls[1][0][0, 0].tolist() == positive[1].tolist()
    assert info['candidates'][0]['final_mode'] == 'positive_negative'
    assert info['candidates'][1]['final_mode'] == 'positive_only'


def test_unrelated_predictor_runtime_error_is_not_hidden():
    groups, info = automatic_background_points(scene(), points(), config())

    class Broken:
        def predict(self, **kwargs):
            raise RuntimeError('模拟GPU或模型故障')

    with pytest.raises(RuntimeError):
        predict_automatic_prompts(Broken(), scene().shape, points(), groups, 8, .5, info)


@pytest.mark.parametrize('options', [
    {'trusted_regions': None}, {'max_points': 0}, {'max_points': 9},
    {'min_distance_px': 50, 'max_distance_px': 40}, {'max_contrast': float('nan')},
    {'coordinate_space': 'unknown'}, {'enabled': 'false'},
])
def test_invalid_negative_configuration_fails_explicitly(options):
    with pytest.raises(ValueError):
        automatic_background_points(scene(), points(), config(**options))


def test_generate_integration_saves_per_candidate_labels_and_debug_images(tmp_path, monkeypatch):
    import segmentation.automatic as module

    cfg = config()
    cfg['wire']['auto']['seed_debug'] = {'enabled': True, 'directory': str(tmp_path)}
    predictor = Predictor()
    segmenter = SimpleNamespace(device='cpu', predictor=predictor, _load=lambda: None)

    def seeds(image, cfg, *args, diagnostics=None, **kwargs):
        diagnostics.update(filter_enabled=False, seed_points_roi_xy=points().tolist())
        return points(), False

    monkeypatch.setattr(module, 'automatic_line_seeds', seeds)
    recognizer = AutomaticWireRecognizer(cfg, segmenter=segmenter)
    recognizer.seed_roi_xyxy = [100, 200, 340, 320]
    records = recognizer._generate(scene())
    assert len(records) == 2
    directory = Path(recognizer.generation_stats['seed_debug_directory'])
    report = json.loads((directory/'prompts.json').read_text())
    assert report['mixed_candidates'] == 2
    assert report['candidates'][0]['positive_rgb_xy'] == [140, 260]
    for i in (0, 1):
        assert cv2.imread(str(directory/f'prompt_{i:03d}.png')).shape == scene().shape


def test_formal_off_experimental_on_and_no_trusted_regions_by_default():
    default = load_config()
    experiment = load_config(Path(__file__).resolve().parents[1]/'configs/wire_negative_prompts_experimental.yaml')
    assert default['wire.auto.negative_prompts.enabled'] is False
    assert experiment['wire.auto.negative_prompts.enabled'] is True
    assert experiment['wire.auto.negative_prompts.trusted_regions'] == []
    groups, _ = automatic_background_points(scene(), points(), experiment)
    assert groups == [[], []]


def test_manual_box_positive_negative_api_is_unchanged():
    from segmentation.wire import Sam2WireSegmenter

    class ManualPredictor:
        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, point_coords, point_labels, box, **kwargs):
            assert point_coords.shape == (2, 2)
            assert point_labels.tolist() == [1, 0]
            assert box.tolist() == [0, 0, 120, 60]
            mask = np.zeros(self.shape, bool)
            mask[30, 10:111] = True
            return mask[None], np.array([.9]), None

    model = Sam2WireSegmenter(Config({'wire': {'crop_margin_px': 0}}))
    model.predictor = ManualPredictor()
    model.device = 'cpu'
    model._load = lambda: None
    image = np.full((100, 200, 3), 128, np.uint8)
    result = model.segment(image, box=[40, 20, 160, 80],
                           points=[[100, 50], [100, 30]], labels=[1, 0])
    assert result.mask[50, 100] and not result.mask[30, 100]
