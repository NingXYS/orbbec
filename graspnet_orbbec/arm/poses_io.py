"""关节姿态 YAML 读写（不依赖 rclpy）。"""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    from zhixing_joints import ALL_JOINTS, GROUP_DUAL
except ImportError:  # 仓库内 pytest：sys.path 是工程根
    from arm.zhixing_joints import ALL_JOINTS, GROUP_DUAL


def load_poses(path: Path) -> tuple[list[str], str, dict[str, list[float]]]:
    doc = yaml.safe_load(path.read_text()) or {}
    names = list(doc.get("joint_names") or ALL_JOINTS)
    group = str(doc.get("group") or GROUP_DUAL)
    poses: dict[str, list[float]] = {}
    for key, val in (doc.get("poses") or {}).items():
        poses[str(key)] = [float(x) for x in val["rad"]]
    if len(poses) < 2:
        raise RuntimeError("poses.yaml 里至少要有两个姿态（先 record pose_a / pose_b）")
    if any(len(q) != len(names) for q in poses.values()):
        raise RuntimeError("姿态关节数与 joint_names 不一致")
    return names, group, poses
