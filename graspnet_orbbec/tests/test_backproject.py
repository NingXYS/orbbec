"""反投影几何正确性：不依赖相机与 GPU。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.types import Intrinsics
from pointcloud.backproject import backproject


def test_principal_point_at_unit_depth():
    K = Intrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    depth = np.zeros((480, 640), np.float32)
    depth[240, 320] = 1.0
    xyz = backproject(depth, K, organized=True)
    np.testing.assert_allclose(xyz[240, 320], [0.0, 0.0, 1.0], atol=1e-5)


def test_known_offset():
    K = Intrinsics(fx=500, fy=500, cx=100, cy=80, width=200, height=160)
    depth = np.zeros((160, 200), np.float32)
    u, v, z = 150, 100, 0.8
    depth[v, u] = z
    xyz = backproject(depth, K, organized=True)
    expected = np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])
    np.testing.assert_allclose(xyz[v, u], expected, atol=1e-5)


def test_50mm_span_at_0_7m():
    """0.7 m 处 50 mm 水平跨度，用内参反投影后 x 方向距离应 ≈ 50 mm。"""
    K = Intrinsics(fx=580, fy=580, cx=320, cy=240, width=640, height=480)
    z = 0.70
    # 50mm 对应像素跨度 = 0.05 * fx / z
    du = 0.05 * K.fx / z
    u0, v0 = 320, 240
    u1 = int(round(u0 + du))
    depth = np.zeros((480, 640), np.float32)
    depth[v0, u0] = z
    depth[v0, u1] = z
    xyz = backproject(depth, K, organized=True)
    dist = np.linalg.norm(xyz[v0, u1] - xyz[v0, u0])
    assert abs(dist - 0.05) < 0.001, dist
