"""隔离加载 PyOrbbecSDK v2.x，避免与 Astra 用的 v1 `pyorbbecsdk` 抢同一导入名。

v1 是 site-packages / third_party/pyorbbecsdk-v1 里的顶层扩展模块；
v2 安装在 third_party/pyorbbecsdk2/（pip --target --no-deps）。
同一进程只能加载其中一个：open_camera 按 backend 二选一。
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from common.paths import ROOT

LOGGER = logging.getLogger("graspnet_orbbec.capture.sdk_v2")

V2_ROOT = ROOT / "third_party" / "pyorbbecsdk2"
V2_PKG = V2_ROOT / "pyorbbecsdk"
V2_LIB = V2_PKG / "libOrbbecSDK.so.2"

_CACHED: Optional[ModuleType] = None


def _preload_native_libs() -> None:
    """先把 v2 的 SONAME 装进进程，避免 LD_LIBRARY_PATH 里的 v1 libOrbbecSDK 抢先。"""
    lib_dir = str(V2_PKG)
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(os.pathsep) if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib_dir] + parts)
    if V2_LIB.exists():
        ctypes.CDLL(str(V2_LIB), mode=ctypes.RTLD_GLOBAL)


def import_sdk_v2() -> ModuleType:
    """导入 PyOrbbecSDK v2，返回 `pyorbbecsdk` 模块对象。"""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    existing = sys.modules.get("pyorbbecsdk")
    if existing is not None:
        loc = str(getattr(existing, "__file__", "") or "")
        if "pyorbbecsdk2" in loc:
            _CACHED = existing
            return existing
        raise RuntimeError(
            "当前进程已加载 PyOrbbecSDK v1，无法再加载 v2。"
            "请只打开一种相机（camera.backend=gemini336l 或 astra），并重启进程。"
        )

    if not (V2_PKG / "__init__.py").exists():
        raise RuntimeError(
            f"未找到 PyOrbbecSDK v2：{V2_PKG}。请执行 bash scripts/install_pyorbbecsdk2.sh"
        )

    _preload_native_libs()
    sys.path.insert(0, str(V2_ROOT))
    import pyorbbecsdk as ob  # noqa: WPS433  隔离路径后的首次导入

    loc = str(getattr(ob, "__file__", "") or "")
    if "pyorbbecsdk2" not in loc:
        raise RuntimeError(f"导入到了错误的 pyorbbecsdk：{loc}（期望 third_party/pyorbbecsdk2）")
    ver = getattr(ob, "__version__", "?")
    LOGGER.info("已加载 PyOrbbecSDK v2 %s  (%s)", ver, loc)
    _CACHED = ob
    return ob
