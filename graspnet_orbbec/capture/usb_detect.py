"""不打开 SDK，仅用 sysfs 识别已插入的 Orbbec USB 设备。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

ORBBEC_VID = 0x2BC5
ASTRA_PRO_PLUS_DEPTH_PID = 0x060F
ASTRA_PRO_PLUS_COLOR_PID = 0x050F
GEMINI_336L_PID = 0x0807
DABAI_DC1_DEPTH_PID = 0x0657
DABAI_DC1_COLOR_PID = 0x0557
# Gemini 330 系列常见 PID 段（335/336/330 及 L 变体）
GEMINI330_PID_MIN = 0x0800
GEMINI330_PID_MAX = 0x0810


def list_orbbec_usb() -> list[dict]:
    """扫描 /sys/bus/usb/devices，返回 Orbbec 设备 {vid,pid,product,serial}。"""
    root = Path("/sys/bus/usb/devices")
    if not root.exists():
        return []
    found: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for node in root.iterdir():
        vendor = node / "idVendor"
        product = node / "idProduct"
        if not vendor.exists() or not product.exists():
            continue
        try:
            vid = int(vendor.read_text().strip(), 16)
            pid = int(product.read_text().strip(), 16)
        except ValueError:
            continue
        if vid != ORBBEC_VID:
            continue
        name = ""
        serial = ""
        prod_file = node / "product"
        sn_file = node / "serial"
        if prod_file.exists():
            name = prod_file.read_text(errors="ignore").strip()
        if sn_file.exists():
            serial = sn_file.read_text(errors="ignore").strip()
        key = (vid, pid, serial)
        if key in seen:
            continue
        seen.add(key)
        found.append({"vid": vid, "pid": pid, "product": name, "serial": serial})
    return found


def is_gemini330_pid(pid: int) -> bool:
    return GEMINI330_PID_MIN <= int(pid) <= GEMINI330_PID_MAX


def detect_capture_backend() -> str:
    """按插入设备选择 capture backend：gemini336l / dabai_dc1 / astra。

    两台都在时优先 Gemini 336L（当前主采集设备）。
    """
    devices = list_orbbec_usb()
    pids = {int(d["pid"]) for d in devices}
    if GEMINI_336L_PID in pids or any(is_gemini330_pid(p) for p in pids):
        return "gemini336l"
    if DABAI_DC1_DEPTH_PID in pids:
        return "dabai_dc1"
    if ASTRA_PRO_PLUS_DEPTH_PID in pids or ASTRA_PRO_PLUS_COLOR_PID in pids:
        return "astra"
    return "gemini336l"
