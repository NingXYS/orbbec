"""抓取规则过滤单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config
from infer.filters import filter_grasps
from pointcloud.workspace import TablePlane


class DummyGG:
    def __init__(self, T, R, widths, scores):
        self.translations = np.asarray(T)
        self.rotation_matrices = np.asarray(R)
        self.widths = np.asarray(widths)
        self.scores = np.asarray(scores)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            return DummyGG(self.translations[idx], self.rotation_matrices[idx], self.widths[idx], self.scores[idx])
        return DummyGG(self.translations[idx], self.rotation_matrices[idx], self.widths[idx], self.scores[idx])


def _gg(n=3):
    T = np.zeros((n, 3))
    T[:, 2] = 0.7
    R = np.zeros((n, 3, 3))
    # 接近方向 = 第三列 = +Z
    R[:, 0, 0] = 1
    R[:, 1, 1] = 1
    R[:, 2, 2] = 1
    widths = np.array([0.04, 0.20, 0.05][:n], dtype=np.float64)
    scores = np.array([0.8, 0.9, 0.01][:n], dtype=np.float64)
    if n == 1:
        widths = np.array([0.04], dtype=np.float64)
        scores = np.array([0.8], dtype=np.float64)
    return DummyGG(T, R, widths, scores)


def test_width_and_score_filter():
    cfg = Config(
        {
            "filter": {
                "min_score": 0.05,
                "min_width_m": 0.01,
                "max_width_m": 0.10,
                "max_approach_angle_deg": 90,
                "min_height_above_table_m": -1,
                "max_height_above_table_m": 10,
            }
        }
    )
    out = filter_grasps(_gg(), plane=None, cfg=cfg)
    # 丢掉 width=0.20 与 score=0.01
    assert len(out) == 1
    assert abs(out.widths[0] - 0.04) < 1e-9


def test_table_height_filter():
    cfg = Config(
        {
            "filter": {
                "min_score": 0.0,
                "min_width_m": 0.0,
                "max_width_m": 1.0,
                "max_approach_angle_deg": 90,
                "min_height_above_table_m": 0.02,
                "max_height_above_table_m": 0.10,
            }
        }
    )
    # 平面 z=0.75（相机系），法向 -Z
    plane = TablePlane(normal=np.array([0.0, 0.0, -1.0]), offset=0.75, inlier_count=10)
    gg = _gg(1)
    gg.translations[0] = np.array([0.0, 0.0, 0.70])  # 离桌 5 cm
    out = filter_grasps(gg, plane=plane, cfg=cfg)
    assert len(out) == 1
    gg.translations[0] = np.array([0.0, 0.0, 0.749])  # 几乎贴桌
    out = filter_grasps(gg, plane=plane, cfg=cfg)
    assert len(out) == 0
