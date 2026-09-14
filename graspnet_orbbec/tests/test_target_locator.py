"""bbox → 3D 质心：合成深度 + 空洞报错 + 耗时。不接相机。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.types import Intrinsics, RGBDFrame
from vlm.target_locator import (
    InsufficientDepthError,
    centroid_in_bbox,
    locate_from_depth,
    locate_target,
)


def _k(w=64, h=48, fx=50.0) -> Intrinsics:
    return Intrinsics(fx=fx, fy=fx, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, width=w, height=h)


def test_median_centroid_ignores_far_outliers():
    h, w = 48, 64
    depth = np.full((h, w), 2.0, np.float32)
    # 目标块在 z=0.50 m，占 bbox 大部分；右下角掺远点
    depth[10:30, 20:40] = 0.50
    depth[26:30, 36:40] = 2.00
    K = _k(w, h)
    r = locate_from_depth(depth, K, [20, 10, 39, 29], min_valid_points=10, bbox_shrink=0.0, depth_band_m=0.06)
    assert abs(r.xyz_m[2] - 0.50) < 0.01
    assert centroid_in_bbox(r)
    # 单像素中心若落在远点会得到 ~2 m；中位数必须仍是 0.5
    assert r.n_inlier >= 10


def test_not_single_pixel():
    h, w = 32, 32
    depth = np.zeros((h, w), np.float32)
    depth[8:24, 8:24] = 0.80
    depth[16, 16] = 0.10  # 中心坏点
    K = _k(w, h, fx=40.0)
    r = locate_from_depth(depth, K, [8, 8, 23, 23], min_valid_points=20, bbox_shrink=0.0, depth_band_m=0.05)
    assert abs(r.xyz_m[2] - 0.80) < 0.02


def test_insufficient_depth_message():
    depth = np.zeros((40, 40), np.float32)
    K = _k(40, 40)
    with pytest.raises(InsufficientDepthError, match="有效深度点") as ei:
        locate_from_depth(depth, K, [5, 5, 20, 20], min_valid_points=50, target="空洞")
    msg = str(ei.value)
    assert "阈值 50" in msg
    assert "空洞" in msg
    assert "无法估计 3D 质心" in msg


def test_clip_and_frame_mismatch():
    h, w = 20, 30
    depth = np.full((h, w), 0.4, np.float32)
    K = _k(w, h)
    r = locate_from_depth(depth, K, [-10, -4, 100, 80], min_valid_points=10, bbox_shrink=0.0)
    assert r.bbox_xyxy == [0, 0, 29, 19]
    color = np.zeros((h, w, 3), np.uint8)
    frame = RGBDFrame(
        color_bgr=color,
        depth_u16=(depth * 1000).astype(np.uint16),
        depth_m=depth,
        color_intrinsics=K,
        depth_intrinsics=K,
        aligned=True,
    )
    r2 = locate_target(frame, [2, 2, 10, 10], target="t")
    assert r2.xyz_m[2] == pytest.approx(0.4, abs=0.01)

    bad = RGBDFrame(
        color_bgr=np.zeros((10, 10, 3), np.uint8),
        depth_u16=np.zeros((20, 30), np.uint16),
        depth_m=depth,
        color_intrinsics=K,
        depth_intrinsics=K,
        aligned=False,
    )
    with pytest.raises(ValueError, match="对齐"):
        locate_target(bad, [1, 1, 5, 5])


def test_locate_faster_than_50ms():
    h, w = 800, 1280
    depth = np.full((h, w), 0.8, np.float32)
    depth[200:500, 400:900] = 0.45
    K = Intrinsics(fx=600.0, fy=600.0, cx=640.0, cy=400.0, width=w, height=h)
    locate_from_depth(depth, K, [400, 200, 899, 499], min_valid_points=50, bbox_shrink=0.08)
    t0 = time.perf_counter()
    for _ in range(20):
        locate_from_depth(depth, K, [400, 200, 899, 499], min_valid_points=50, bbox_shrink=0.08)
    avg_ms = (time.perf_counter() - t0) / 20 * 1000.0
    assert avg_ms < 50.0, f"定位平均 {avg_ms:.1f} ms"


def test_saved_aligned_frame_if_present():
    data = ROOT / "vlm" / "locator_data" / "desk"
    if not (data / "color.png").exists():
        pytest.skip("无冻结对齐帧")
    from capture.camera import load_rgbd_dir

    frame = load_rgbd_dir(data)
    r = locate_target(frame, [479, 398, 544, 538], target="黑色杯子")
    assert r.xyz_m[2] > 0.3
    assert centroid_in_bbox(r)
    assert r.elapsed_ms < 50.0
