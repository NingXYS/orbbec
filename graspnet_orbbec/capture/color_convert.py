"""Orbbec 彩色帧转 OpenCV BGR。Astra Pro Plus 常见 MJPG / YUYV / RGB。"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _format_name(fmt) -> str:
    return str(fmt).split(".")[-1].upper()


def frame_to_bgr(frame) -> Optional[np.ndarray]:
    """把 pyorbbecsdk VideoFrame 转成 HxWx3 BGR uint8。"""
    width = frame.get_width()
    height = frame.get_height()
    fmt = _format_name(frame.get_format())
    data = np.asanyarray(frame.get_data())
    if data.size == 0:
        return None

    if fmt in ("RGB", "RGB888"):
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt in ("BGR", "BGR888"):
        return np.resize(data, (height, width, 3)).copy()
    if fmt in ("YUYV", "YUY2"):
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == "UYVY":
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    if fmt in ("MJPG", "MJPEG"):
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return decoded
    if fmt == "I420":
        y = data[0:height, :].reshape(height, width) if data.ndim == 2 else None
        if y is None:
            buf = np.frombuffer(data, dtype=np.uint8)
            yuv = buf.reshape(height * 3 // 2, width)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_I420)
    if fmt == "NV12":
        buf = np.frombuffer(np.ascontiguousarray(data), dtype=np.uint8)
        yuv = buf.reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if fmt == "NV21":
        buf = np.frombuffer(np.ascontiguousarray(data), dtype=np.uint8)
        yuv = buf.reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)

    # 未知格式：尝试按 RGB888 解读
    if data.size == height * width * 3:
        img = data.reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return None
