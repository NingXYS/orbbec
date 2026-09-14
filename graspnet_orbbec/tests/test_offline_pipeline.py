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
