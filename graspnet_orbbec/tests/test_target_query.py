"""点名链路：found=false / 超时 / 无点云，且主循环路径不打 API。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.types import Intrinsics, RGBDFrame
from common.config import Config
from infer.target_query import QueryError, QueryWorker, run_target_query, snapshot_frame
from tests.test_match_grasp import _gg
from vlm.vlm_client import VLMDetectResult


def _frame(h=48, w=64, z=0.5, hole=False) -> RGBDFrame:
    color = np.zeros((h, w, 3), np.uint8)
    color[10:30, 20:40] = (0, 200, 255)
    depth = np.full((h, w), 0.0 if hole else z, np.float32)
    if not hole:
        depth[10:30, 20:40] = z
    K = Intrinsics(fx=50.0, fy=50.0, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h)
    return RGBDFrame(
        color_bgr=color,
        depth_u16=(np.clip(depth * 1000, 0, 65535)).astype(np.uint16),
        depth_m=depth,
        color_intrinsics=K,
        depth_intrinsics=K,
        aligned=True,
    )


def _cfg() -> Config:
    return Config(
        {
            "live": {
                "vlm_timeout_s": 10.0,
                "vlm_max_retries": 0,
                "min_match_score": 0.08,
                "max_match_dist_m": 0.30,
            },
            "locator": {
                "min_depth_m": 0.12,
                "max_depth_m": 3.5,
                "min_valid_points": 20,
                "bbox_shrink": 0.0,
                "depth_band_m": 0.08,
            },
            "camera": {"min_depth_m": 0.12, "max_depth_m": 3.5},
        }
    )


class FakeClient:
    def __init__(self, found=True, bbox=None, delay=0.0, exc=None):
        self.found = found
        self.bbox = bbox or [20, 10, 39, 29]
        self.delay = delay
        self.exc = exc
        self.calls = 0

    def detect(self, image, target, timeout_s=None, max_retries=None):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return VLMDetectResult(
            found=self.found,
            bbox_2d=self.bbox,
            target=str(target),
            latency_s=0.01,
            retries=0,
            model="fake",
            image_size=(64, 48),
        )


def test_happy_path_nearest_and_fast():
    frame = _frame()
    xyz_box = np.array([((29.5) - 31.5) * 0.5 / 50.0, ((19.5) - 23.5) * 0.5 / 50.0, 0.5])
    gg = _gg(
        [xyz_box + np.array([0.08, 0, 0]), xyz_box + np.array([0.01, 0, 0]), xyz_box + np.array([0.20, 0, 0])],
        [0.99, 0.55, 0.80],
    )
    t0 = time.perf_counter()
    r = run_target_query(frame, "黄块", gg, FakeClient(), cfg=_cfg())
    assert time.perf_counter() - t0 < 5.0
    assert r.match.index == 1
    assert r.match.score == pytest.approx(0.55)
    assert r.match.score >= 0.08
    assert r.elapsed_s < 5.0


def test_found_false():
    with pytest.raises(QueryError) as ei:
        run_target_query(_frame(), "香蕉", _gg([[0, 0, 0.5]], [0.9]), FakeClient(found=False), cfg=_cfg())
    assert ei.value.kind == "not_found"
    assert "未找到目标" in ei.value.message
    assert "重新输入" in ei.value.message


def test_timeout_and_network():
    with pytest.raises(QueryError) as ei:
        run_target_query(_frame(), "x", _gg([[0, 0, 0.5]], [0.9]), FakeClient(exc=TimeoutError("timed out")), cfg=_cfg())
    assert ei.value.kind == "timeout"
    assert "超时" in ei.value.message
    with pytest.raises(QueryError) as ei:
        run_target_query(
            _frame(), "x", _gg([[0, 0, 0.5]], [0.9]), FakeClient(exc=ConnectionError("Network unreachable")), cfg=_cfg()
        )
    assert ei.value.kind == "network"
    assert "网络" in ei.value.message


def test_no_depth():
    with pytest.raises(QueryError) as ei:
        run_target_query(_frame(hole=True), "x", _gg([[0, 0, 0.5]], [0.9]), FakeClient(), cfg=_cfg())
    assert ei.value.kind == "no_depth"
    assert "无有效点云" in ei.value.message


def test_twenty_queries_no_crash():
    frame = _frame()
    gg = _gg([[0.0, 0.0, 0.5]], [0.7])
    client = FakeClient()
    n_ok = 0
    for i in range(20):
        r = run_target_query(frame, f"目标{i}", gg, client, cfg=_cfg())
        assert r.match.score >= 0.08
        n_ok += 1
    assert n_ok == 20
    assert client.calls == 20


def test_snapshot_is_copy():
    f = _frame()
    s = snapshot_frame(f)
    f.color_bgr[:] = 1
    assert int(s.color_bgr[15, 25, 1]) == 200


def test_query_worker_timeout_does_not_block():
    cfg = _cfg()
    cfg["live"]["vlm_timeout_s"] = 0.4
    w = QueryWorker(cfg, FakeClient(delay=1.2))
    w.start()
    t0 = time.perf_counter()
    w.submit(_frame(), "x", _gg([[0.0, 0.0, 0.5]], [0.9]))
    while w.busy and time.perf_counter() - t0 < 2.5:
        time.sleep(0.02)
    elapsed = time.perf_counter() - t0
    w.stop_flag = True
    assert not w.busy
    assert w.error is not None and w.error.kind == "timeout"
    assert elapsed < 1.5, f"超时释放 busy 用了 {elapsed:.2f}s，主循环会被拖死"


def test_idle_overlay_fps():
    from capture.camera import load_rgbd_dir
    from viz.visualize import overlay_grasps_on_rgb

    data = ROOT / "vlm" / "locator_data" / "desk"
    if not (data / "color.png").exists():
        pytest.skip("无对齐帧")
    frame = load_rgbd_dir(data)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.2:
        overlay_grasps_on_rgb(frame.color_bgr, None, frame.color_intrinsics, top_k=8)
        n += 1
    fps = n / (time.perf_counter() - t0)
    assert fps >= 5.0, f"等待态叠加仅 {fps:.1f} FPS"
