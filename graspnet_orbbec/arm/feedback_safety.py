"""与ROS无关的逐轴反馈和动作证据合同，供控制入口及离线测试复用。"""

import math


def update_joints(positions, received_at, names, values, now):
    if len(names) != len(values) or len(set(names)) != len(names):
        for name in names:
            positions.pop(name, None)
            received_at.pop(name, None)
        return
    for name, value in zip(names, values):
        if not math.isfinite(value):
            positions.pop(name, None)
            received_at.pop(name, None)
        else:
            positions[name], received_at[name] = float(value), now


def require_joints(positions, received_at, names, now, max_age=2.5):
    bad = [name for name in names if name not in positions or not math.isfinite(positions[name])
           or name not in received_at or not 0 <= now-received_at[name] <= max_age]
    if bad:
        raise ValueError('关节反馈缺失、无效或陈旧：'+', '.join(bad))


class MotionAudit:
    """动作尝试不是实际运动证据；异常后保留unknown而非断言未动。"""
    def __init__(self):
        self.attempts = []
        self.observed = False

    def attempt(self, stage):
        self.attempts.append(stage)

    def observe(self, before, after):
        if self.attempts and any(name in before and abs(value-before[name]) > 1e-6
                                 for name, value in after.items()):
            self.observed = True

    def report(self):
        return {'physical_motion_performed': True if self.observed else None if self.attempts else False,
                'motion_commands_attempted': list(self.attempts),
                'motion_evidence': 'feedback_change_observed' if self.observed else
                                   'unknown_after_attempt' if self.attempts else 'no_motion_requested'}
