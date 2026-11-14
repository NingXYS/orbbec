"""夹爪闭合轴与局部线材切线的锐角约束；不影响接近轴倾角。"""

import numpy as np

CLOSING_MIN_DEG = 85.0
CLOSING_MAX_DEG = 90.0


def closing_angle_deg(closing, tangent):
    a, b = np.asarray(closing, float), np.asarray(tangent, float)
    if (a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9):
        raise ValueError('闭合角需要有效闭合轴和线材切线')
    # 轴方向正负不改变夹持夹角，统一使用0–90度锐角。
    return float(np.degrees(np.arccos(np.clip(abs(a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))))


def closing_angle_review(plan):
    try:
        transform = np.asarray(plan['coordinate_verification']['T_base_rgb'], float)
        tangent = transform[:3, :3] @ np.asarray(plan['wire_tangent_camera'], float)
        closing = np.asarray(plan['T_base_grasp'], float)[:3, 0]
        angle = closing_angle_deg(closing, tangent)
        return {'passed': CLOSING_MIN_DEG-1e-8 <= angle <= CLOSING_MAX_DEG+1e-8,
                'angle_deg': angle, 'range_deg': [CLOSING_MIN_DEG, CLOSING_MAX_DEG],
                'wire_tangent_base': (tangent/np.linalg.norm(tangent)).tolist()}
    except (KeyError, TypeError, ValueError, IndexError):
        return {'passed': False, 'range_deg': [CLOSING_MIN_DEG, CLOSING_MAX_DEG],
                'reason': 'closing_axis_or_wire_tangent_unavailable'}
