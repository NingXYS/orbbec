"""项目内固定路径。所有模块通过这里定位仓库根目录。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
GRASPNET_ROOT = THIRD_PARTY / "graspnet-baseline"
GRASPNET_API_ROOT = THIRD_PARTY / "graspnetAPI"
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "output"
CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
CONDA_PREFIX = ROOT / ".conda"


def ensure_sys_path() -> None:
    """把项目根与 GraspNet 源码加入 sys.path，保证独立脚本可导入。"""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    for extra in (
        GRASPNET_ROOT,
        GRASPNET_ROOT / "models",
        GRASPNET_ROOT / "utils",
        GRASPNET_ROOT / "dataset",
        GRASPNET_ROOT / "pointnet2",
        GRASPNET_ROOT / "knn",
    ):
        s = str(extra)
        if extra.exists() and s not in sys.path:
            sys.path.append(s)


def timestamp_dir(base: Path | None = None) -> Path:
    """创建带时间戳的输出子目录。"""
    from datetime import datetime

    base = base or OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out
