"""自动正点区域与坐标合同测试；SAM2集成使用假预测器，无硬件或权重依赖。"""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from segmentation.automatic import (
    AutomaticWireRecognizer, automatic_line_seeds, line_contrast, seed_search_mask,
)
from scripts.compare_wire_seeds import compare


def config(**options):
    return Config({'wire': {'auto': {'seed_filter': {'enabled': True, **options}}}})


def scene():
    image = np.full((100, 200, 3), 128, np.uint8)
    for y, color in [(30, 10), (70, 245)]:
        cv2.line(image, (10, y), (190, y), (color,)*3, 3)
    return image


def test_no_regions_matches_legacy_and_points_land_on_wires():
    image = scene()
    old, capped = automatic_line_seeds(image, Config())
    new, filtered_capped = automatic_line_seeds(image, config())
    assert np.array_equal(old, new) and capped == filtered_capped
    assert len(new) > 0
    assert np.all(np.minimum(abs(new[:, 1]-30), abs(new[:, 1]-70)) <= 2)


def test_disabled_filter_ignores_invalid_regions_and_preserves_points():
    image = scene()
    cfg = config(search_regions='ignored', exclude_regions='ignored', coordinate_space='bad')
    cfg['wire']['auto']['seed_filter']['enabled'] = False
    expected, cap = automatic_line_seeds(image, Config(), [[10, 30]])
    actual, actual_cap = automatic_line_seeds(image, cfg, [[10, 30]])
    assert np.array_equal(actual, expected) and actual_cap == cap


def test_search_union_exclusion_and_stats():
    report = {}
    points, _ = automatic_line_seeds(scene(), config(
        search_regions=[[0, 20, 200, 40], [0, 60, 200, 80]],
        exclude_regions=[[80, 0, 130, 100]]), diagnostics=report)
    assert len(points) > 0
    assert all(not (80 <= x < 130) for x, y in points)
    assert any(abs(y-30) <= 2 for x, y in points)
    assert any(abs(y-70) <= 2 for x, y in points)
    assert report['allowed_pixels'] == 6000
    assert report['contrast_pixels_filtered'] > 0
    assert report['contrast_pixels_before_filter'] == (
        report['contrast_pixels_after_filter'] + report['contrast_pixels_filtered'])


def test_roi_rgb_and_local_regions_are_equivalent():
    roi = [100, 200, 300, 300]
    rgb = config(search_regions=[[100, 220, 300, 240]], exclude_regions=[[180, 220, 230, 240]])
    local = config(coordinate_space='roi', search_regions=[[0, 20, 200, 40]],
                   exclude_regions=[[80, 20, 130, 40]])
    report = {}
    a, _ = automatic_line_seeds(scene(), rgb, roi_xyxy=roi, diagnostics=report)
    b, _ = automatic_line_seeds(scene(), local, roi_xyxy=roi)
    assert np.array_equal(a, b) and len(a)
    assert np.array_equal(np.asarray(report['seed_points_rgb_xy']), a+[100, 200])


@pytest.mark.parametrize('regions', [[], [[200, 0, 300, 100]], [[10, 10, 10, 20]]])
def test_empty_search_returns_no_seeds(regions):
    report = {}
    points, capped = automatic_line_seeds(scene(), config(search_regions=regions), diagnostics=report)
    assert points.shape == (0, 2) and not capped
    assert report['allowed_pixels'] == 0 and report['seed_points_rgb_xy'] == []


def test_all_excluded_removes_extra_seeds_and_no_fallback():
    report = {}
    points, _ = automatic_line_seeds(scene(), config(exclude_regions=[[-10, -10, 220, 120]]),
                                     [[10, 30], [80, 70]], diagnostics=report)
    assert not len(points) and report['extra_points_filtered'] == 2


def test_regions_clip_to_boundaries_and_use_half_open_coordinates():
    cfg = config(coordinate_space='roi', search_regions=[[-10, -10, 201, 101]],
                 exclude_regions=[[0, 0, 1, 100], [199, 0, 200, 100]])
    mask, _ = seed_search_mask((100, 200), cfg)
    assert mask[:, 1:199].all() and not mask[:, [0, 199]].any()
    points, _ = automatic_line_seeds(scene(), cfg, [[0, 30], [1, 30], [199, 70], [200, 70]])
    assert all(1 <= x < 199 for x, y in points)
    assert [1, 30] in points.tolist()


@pytest.mark.parametrize('regions', ['bad', [[2, 0, 1, 10]], [[0., 0, 20, 20]],
                                    [[False, 0, 20, 20]], [[0, 0, 1]]])
def test_invalid_region_config_fails_explicitly(regions):
    with pytest.raises(ValueError, match='种子区域'):
        automatic_line_seeds(scene(), config(search_regions=regions))


def test_invalid_coordinate_space_and_roi_fail():
    with pytest.raises(ValueError, match='coordinate_space'):
        automatic_line_seeds(scene(), config(coordinate_space='unknown'))
    with pytest.raises(ValueError, match='ROI'):
        automatic_line_seeds(scene(), config(), roi_xyxy=[1, 2, 50, 100])


def test_threshold_uses_only_allowed_search_pixels():
    image = scene()
    search = np.zeros(image.shape[:2], bool)
    search[:10] = True
    contrast, threshold = line_contrast(image, Config(), search)
    assert threshold == 12 and contrast.shape == search.shape
    assert not len(automatic_line_seeds(np.full_like(image, 128), config())[0])
    assert not len(automatic_line_seeds(np.zeros((1, 1, 3), np.uint8), config())[0])


def test_empty_search_does_not_load_sam_or_fall_back_to_full_image():
    recognizer = AutomaticWireRecognizer(config(search_regions=[]), segmenter=SimpleNamespace(
        _load=lambda: pytest.fail('空区域不应加载SAM2')))
    assert recognizer._generate(scene()) == []
    assert recognizer.generation_stats['seed_count'] == 0


def test_dense_mode_rejects_seed_filter_instead_of_silently_ignoring_it():
    cfg = config()
    cfg['wire']['auto']['method'] = 'dense'
    recognizer = AutomaticWireRecognizer(cfg)
    with pytest.raises(ValueError, match='仅支持line_prompts'):
        recognizer._generate(scene())


def test_budget_and_extra_points_are_filtered_before_spending_budget():
    cfg = config(search_regions=[[0, 20, 200, 40]])
    cfg['wire']['auto']['max_seeds'] = 1
    report = {}
    points, capped = automatic_line_seeds(scene(), cfg, [[10, 70], [20, 30]], diagnostics=report)
    assert points.tolist() == [[20, 30]] and capped
    assert report['extra_points_filtered'] == 1


def test_regions_do_not_clip_sam_mask_and_debug_preserves_rgb_coordinates(tmp_path):
    roi = [100, 200, 300, 300]
    full = np.full((400, 500, 3), 128, np.uint8)
    crop = scene()
    full[200:300, 100:300] = crop
    mask = np.zeros(crop.shape[:2], bool)
    mask[29:32, 10:191] = True
    cfg = config(search_regions=[[100, 220, 300, 240]],
                 exclude_regions=[[180, 200, 230, 300]])
    cfg['wire']['auto'].update(roi_xyxy=roi, terminal_enabled=False, max_width_fraction=.05,
                               seed_debug={'enabled': True, 'directory': str(tmp_path)})
    received = []

    class Predictor:
        def set_image(self, image):
            assert np.array_equal(image, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        def predict(self, point_coords, **kwargs):
            received.extend(point_coords[:, 0].tolist())
            n = len(point_coords)
            return np.tile(mask, (n, 1, 1, 1)), np.full((n, 1), .9), None

    segmenter = SimpleNamespace(device='cpu', predictor=Predictor(), _load=lambda: None)
    result = AutomaticWireRecognizer(cfg, segmenter=segmenter).recognize(full)
    assert received and all(not (80 <= x < 130) for x, y in received)
    assert result.mask[230, 200]  # 排除区域中的真实线身仍可由SAM2返回。
    sampling = result.automation['sampling']
    assert sampling['seed_points_xy'] == sampling['seed_filter']['seed_points_rgb_xy']
    directory = Path(sampling['seed_debug_directory'])
    saved = json.loads((directory/'seeds.json').read_text())
    assert saved['roi_xyxy_original_rgb'] == roi
    assert cv2.imread(str(directory/'seeds_roi.png')).shape == crop.shape


def test_ab_artifacts_compare_positions_counts_and_refuse_overwrite(tmp_path):
    image = scene()
    original = image.copy()
    cfg = config(exclude_regions=[[0, 60, 200, 90]])
    before_cfg = copy.deepcopy(cfg)
    out = tmp_path/'comparison'
    summary = compare(image, cfg, None, out)
    before = json.loads((out/'before.json').read_text())
    after = json.loads((out/'after.json').read_text())
    assert before['seed_count'] > after['seed_count'] > 0
    assert any(y >= 60 for x, y in before['seed_points_rgb_xy'])
    assert all(y < 60 for x, y in after['seed_points_rgb_xy'])
    assert not summary['sam_inference_performed'] and summary['robot_commands_sent'] == 0
    assert cv2.imread(str(out/'after.png')).shape == image.shape
    assert np.array_equal(image, original) and cfg == before_cfg
    with pytest.raises(FileExistsError):
        compare(image, cfg, None, out)
