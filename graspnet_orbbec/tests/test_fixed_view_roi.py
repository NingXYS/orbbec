"""固定工位ROI只约束推理区域，绝不改变原图与坐标合同。"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import load_config
from segmentation.automatic import AutomaticWireRecognizer


def fake_recognizer(cfg, calls):
    recognizer = AutomaticWireRecognizer(cfg)
    recognizer._detect_terminals = lambda image: ([], [])
    def generate(image):
        calls.append(image.shape)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.line(mask, (10, 20), (image.shape[1]-20, 30), 1, 3)
        return [{'segmentation': mask > 0, 'predicted_iou': .9}]
    recognizer._generate = generate
    return recognizer


def test_default_fixed_view_keeps_original_coordinates_and_image():
    cfg = load_config()
    assert cfg['wire.auto.roi_xyxy'] == [40, 160, 555, 218]
    assert cfg['wire.auto.roi_reference_size_wh'] == [640, 480]
    image = np.full((480, 640, 3), 128, np.uint8)
    calls = []
    result = fake_recognizer(cfg, calls).recognize(image)
    assert calls == [(58, 515, 3)]
    assert result.mask.shape == (480, 640) and np.array_equal(result.image_bgr, image)
    y, x = np.where(result.mask)
    assert x.min() >= 40 and x.max() < 555 and y.min() >= 160 and y.max() < 218
    assert result.crop_xyxy == [40, 160, 555, 218]
    assert all(np.asarray(p)[:, 1].min() >= 160 for p in result.paths_xy)


@pytest.mark.parametrize('shape', [(960, 1280, 3), (240, 320, 3)])
def test_different_resolution_refused_before_inference(shape):
    calls = []
    with pytest.raises(ValueError, match='尺寸不匹配'):
        fake_recognizer(load_config(), calls).recognize(np.zeros(shape, np.uint8))
    assert not calls


def test_explicit_full_frame_opt_out_remains_supported():
    cfg = load_config()
    cfg['wire']['auto']['roi_xyxy'] = None
    calls = []
    result = fake_recognizer(cfg, calls).recognize(np.zeros((480, 640, 3), np.uint8))
    assert calls == [(480, 640, 3)] and result.crop_xyxy == [0, 0, 640, 480]
