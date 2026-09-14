"""最近抓取匹配：必须是距质心最近且分数合格。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer.match_grasp import NoGraspMatchError, match_nearest_grasp


class DummyGG:
    def __init__(self, T, R, widths, scores):
        self.translations = np.asarray(T)
        self.rotation_matrices = np.asarray(R)
        self.widths = np.asarray(widths)
        self.scores = np.asarray(scores)

    def __len__(self):
        return len(self.scores)


def _gg(translations, scores, widths=None):
    T = np.asarray(translations, dtype=np.float64)
    n = len(T)
    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1] = 1
    R[:, 2, 2] = 1
    W = np.full(n, 0.04) if widths is None else np.asarray(widths)
    return DummyGG(T, R, W, np.asarray(scores, dtype=np.float64))


def test_picks_nearest_not_highest_score():
    gg = _gg(
        [[0.20, 0.0, 0.50], [0.01, 0.0, 0.50], [0.40, 0.0, 0.50]],
        [0.95, 0.40, 0.90],
    )
    m = match_nearest_grasp(gg, [0.0, 0.0, 0.50], min_score=0.08, max_dist_m=0.30)
    assert m.index == 1
    assert m.score == pytest.approx(0.40)
    assert m.distance_m == pytest.approx(0.01, abs=1e-6)


def test_skips_abnormally_low_score():
    gg = _gg(
        [[0.0, 0.0, 0.50], [0.05, 0.0, 0.50]],
        [0.01, 0.50],
    )
    m = match_nearest_grasp(gg, [0.0, 0.0, 0.50], min_score=0.08, max_dist_m=0.30)
    assert m.index == 1
    assert m.score >= 0.08


def test_rejects_too_far_and_empty():
    gg = _gg([[1.0, 0.0, 0.5]], [0.9])
    with pytest.raises(NoGraspMatchError, match="距质心"):
        match_nearest_grasp(gg, [0.0, 0.0, 0.5], min_score=0.08, max_dist_m=0.20)
    with pytest.raises(NoGraspMatchError, match="无 GraspNet"):
        match_nearest_grasp(_gg([], []), [0.0, 0.0, 0.5])
