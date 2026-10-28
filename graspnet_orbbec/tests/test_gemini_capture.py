"""Gemini 336L 采集模块：接口、USB 识别、边缘叠加。不要求接相机。"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.align_vis import edge_overlay, measure_edge_alignment_px
from capture.camera import AstraCamera, FileCamera, open_camera
from capture.gemini336l import Gemini336LCamera
from capture.usb_detect import GEMINI_336L_PID, detect_capture_backend, is_gemini330_pid
from common.config import Config


def test_public_methods_match_astra():
    needed = ("open", "close", "read", "frames", "get_intrinsics", "__enter__", "__exit__")
    for name in needed:
        assert hasattr(Gemini336LCamera, name)
        assert callable(getattr(Gemini336LCamera, name))
    astra_read = inspect.signature(AstraCamera.read)
    gemini_read = inspect.signature(Gemini336LCamera.read)
    assert "timeout_ms" in astra_read.parameters
    assert "timeout_ms" in gemini_read.parameters


def test_open_camera_dispatches_gemini_without_sdk():
    cfg = Config({"camera": {"backend": "gemini336l"}})
    cam = open_camera(cfg)
    assert isinstance(cam, Gemini336LCamera)


def test_open_camera_keeps_astra():
    cfg = Config({"camera": {"backend": "astra"}})
    cam = open_camera(cfg)
    assert isinstance(cam, AstraCamera)
    cfg2 = Config({"camera": {"backend": "pyorbbec"}})
    assert isinstance(open_camera(cfg2), AstraCamera)


def test_open_camera_file():
    data = ROOT / "third_party" / "graspnet-baseline" / "doc" / "example_data"
    cam = open_camera(Config({"camera": {"backend": "file"}}), data_dir=data)
    assert isinstance(cam, FileCamera)


def test_gemini330_pid():
    assert is_gemini330_pid(GEMINI_336L_PID)
    assert not is_gemini330_pid(0x060F)
    backend = detect_capture_backend()
    assert backend in ("gemini336l", "astra", "dabai_dc1")


def test_edge_overlay_perfect_align():
    h, w = 160, 200
    color = np.full((h, w, 3), 30, np.uint8)
    color[:, 90:110] = 255
    depth = np.full((h, w), 0.8, np.float32)
    # 中间一条深度台阶，几何边与 RGB 竖条重合
    depth[:, 90:110] = 0.4
    overlay, err = edge_overlay(color, depth)
    assert overlay.shape == color.shape
    assert err is not None
    assert err <= 3.0, err


def test_measure_returns_none_on_empty_depth():
    color = np.zeros((40, 50, 3), np.uint8)
    depth = np.zeros((40, 50), np.float32)
    assert measure_edge_alignment_px(color, depth) is None
