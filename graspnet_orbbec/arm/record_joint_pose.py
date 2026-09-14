#!/usr/bin/env python3
"""从真机 /joint_states 记录一个关节姿态到 YAML。

在控制器 Jazzy、且 Hardware 已起来后运行。RViz 里把橙色 Goal 拖到安全位、
Plan + Execute 到达后，再运行本脚本保存当前真实关节角。

用法（Thor 上，DOMAIN 与真机入口打印的一致）：
  python3 record_joint_pose.py --name pose_a --out /tmp/arm_safe/poses.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from zhixing_joints import ALL_JOINTS, GROUP_DUAL
except ImportError:
    from arm.zhixing_joints import ALL_JOINTS, GROUP_DUAL


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("record_joint_pose")
        self.pos: dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self._cb, qos_profile_sensor_data)

    def _cb(self, msg: JointState) -> None:
        self.pos.update(zip(msg.name, msg.position))

    def wait_complete(self, timeout: float = 8.0) -> dict[str, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(n in self.pos for n in ALL_JOINTS):
                return {n: float(self.pos[n]) for n in ALL_JOINTS}
        missing = [n for n in ALL_JOINTS if n not in self.pos]
        raise RuntimeError(f"/joint_states 缺轴: {missing}。真机栈是否已启动？")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="姿态名，如 pose_a")
    p.add_argument("--out", default="/tmp/arm_safe/poses.yaml")
    args = p.parse_args()
    rclpy.init()
    node = Recorder()
    try:
        joints = node.wait_complete()
    except Exception as e:
        print(f"RECORD_FAIL: {e}", file=sys.stderr)
        rclpy.shutdown()
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    if out.is_file():
        doc = yaml.safe_load(out.read_text()) or {}
    doc.setdefault("group", GROUP_DUAL)
    doc.setdefault("joint_names", ALL_JOINTS)
    doc.setdefault("poses", {})
    doc["poses"][args.name] = {
        "rad": [joints[n] for n in ALL_JOINTS],
        "deg": [math.degrees(joints[n]) for n in ALL_JOINTS],
    }
    out.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
    print(f"RECORD_OK {args.name} -> {out}")
    for n in ALL_JOINTS:
        print(f"  {n}: {math.degrees(joints[n]):.4f} deg")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
