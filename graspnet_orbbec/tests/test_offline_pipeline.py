"""离线数据通路：官方 example_data 能读出对齐 RGB-D 与点云。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.camera import load_rgbd_dir
from pointcloud.backproject import frame_to_cloud
from pointcloud.workspace import workspace_mask


def test_official_example_loads():
    data = ROOT / "third_party" / "graspnet-baseline" / "doc" / "example_data"
    frame = load_rgbd_dir(data)
    assert frame.aligned
    assert frame.color_bgr.ndim == 3
    assert frame.depth_m.shape == frame.color_bgr.shape[:2]
    assert frame.color_intrinsics.fx > 0
    cloud = frame_to_cloud(frame, min_depth=0.1, max_depth=2.0)
    pts, cols = cloud.flatten()
    assert pts.shape[0] > 1000
    assert pts.shape[1] == 3
    mask, plane = workspace_mask(cloud)
    assert mask.any()
    assert np.isfinite(pts).all()


def test_unaligned_frame_keeps_sensor_geometry(tmp_path):
    from capture.camera import save_rgbd_dir
    from capture.types import RGBDFrame, Intrinsics
    import pytest
    rgb = np.zeros((4, 4, 3), np.uint8)
    depth = np.ones((2, 2), np.uint16)
    raw = RGBDFrame(rgb, depth, depth * .001, Intrinsics(250, 250, 2, 2, 4, 4),
                    Intrinsics(100, 100, 1, 1, 2, 2), aligned=False)
    save_rgbd_dir(raw, tmp_path)
    frame = load_rgbd_dir(tmp_path)
    assert not frame.aligned and frame.depth_m.shape == (2, 2)
    assert frame.depth_intrinsics == raw.depth_intrinsics
    with pytest.raises(ValueError, match='对齐'):
        frame_to_cloud(frame)


def test_file_intrinsics_peek_and_non_mm_roundtrip(tmp_path):
    import cv2
    import json
    from capture.camera import FileCamera, save_rgbd_dir
    cv2.imwrite(str(tmp_path/'color.png'), np.zeros((2, 2, 3), np.uint8))
    cv2.imwrite(str(tmp_path/'depth.png'), np.full((2, 2), 500, np.uint16))
    (tmp_path/'intrinsics.json').write_text(json.dumps(dict(fx=10, fy=10, cx=1, cy=1, factor_depth=4000)))
    cam = FileCamera(tmp_path)
    assert cam.get_intrinsics().fx == 10
    original = cam.read()
    save_rgbd_dir(original, tmp_path/'again')
    np.testing.assert_allclose(load_rgbd_dir(tmp_path/'again').depth_m, .125)


def test_failed_depth_image_write_leaves_previous_frame_intact(tmp_path, monkeypatch):
    import cv2
    import pytest
    from capture.camera import save_rgbd_dir
    frame = load_rgbd_dir(ROOT/'vlm/locator_data/desk')
    save_rgbd_dir(frame, tmp_path)
    previous = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    write = cv2.imwrite
    monkeypatch.setattr(cv2, 'imwrite', lambda path, data: False if path.endswith('depth.png') else write(path, data))
    frame.color_bgr.fill(0)
    with pytest.raises(OSError, match='depth.png'):
        save_rgbd_dir(frame, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == previous
