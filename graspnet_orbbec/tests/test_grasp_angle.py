"""收紧夹持夹角，不改变接近轴或历史识别文件。"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm.grasp_angle import closing_angle_deg, closing_angle_review


@pytest.mark.parametrize('angle,passed', [(70, False), (84.9, False), (85, True), (89, True), (90, True)])
def test_closing_angle_boundary(angle, passed):
    t = np.eye(4)
    tangent = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]
    review = closing_angle_review({'coordinate_verification': {'T_base_rgb': t.tolist()},
                                  'T_base_grasp': t.tolist(), 'wire_tangent_camera': tangent})
    assert review['passed'] is passed
    assert review['angle_deg'] == pytest.approx(angle)
    assert closing_angle_deg([-1,0,0], tangent) == pytest.approx(angle)


def test_missing_or_degenerate_wire_direction_fails():
    assert not closing_angle_review({})['passed']
    with pytest.raises(ValueError):
        closing_angle_deg([1,0,0], [0,0,0])
