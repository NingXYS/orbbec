"""二维识别专用彩色流；后台持续取最新帧，不依赖深度或修改激光。"""

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
import cv2

from capture.color_convert import frame_to_bgr
from common.config import load_config


@dataclass
class ColorFrame:
    color_bgr: np.ndarray
    timestamp_ns: int


class FileColorCamera:
    """识别回放只读原图；没有深度时不补造RGB-D数据。"""
    def __init__(self, data_dir, loop=True):
        root = Path(data_dir)
        self.paths = ([root] if root.is_file() else
                      [p for p in (root/'color.png', root/'rgb.png') if p.is_file()])
        if not self.paths and root.is_dir():
            self.paths = sorted(root.glob('*/color.png'))
        if not self.paths:
            raise FileNotFoundError(f'未找到彩色原图：{root}')
        self.loop, self.index = loop, 0

    def open(self): pass
    def close(self): pass

    def read(self):
        if self.index >= len(self.paths):
            if not self.loop:
                raise StopIteration
            self.index = 0
        path = self.paths[self.index]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'无法读取彩色原图：{path}')
        return ColorFrame(image, 0)


class OrbbecColorCamera:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.context = self.device = self.pipeline = self.thread = None
        self.started = False
        self.stopping = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = self.last_read_sequence = 0
        self.error = None

    def open(self):
        if self.started:
            return
        backend = str(self.cfg.get_path('camera.backend', 'auto')).lower()
        if backend == 'auto':
            from capture.usb_detect import detect_capture_backend
            backend = detect_capture_backend()
        if backend in ('gemini336l', 'gemini', 'gemini330', 'gemini336'):
            from capture.sdk_v2 import import_sdk_v2
            ob = import_sdk_v2()
        elif backend in ('dabai_dc1', 'dc1', 'astra', 'pyorbbec', 'astra_pro_plus'):
            from capture.dabai_dc1 import _import_sdk_v1
            ob = _import_sdk_v1()
        else:
            raise ValueError(f'彩色识别模式不支持相机后端：{backend}')
        try:
            self.context = ob.Context()
            devices = self.context.query_devices()
            serial = str(self.cfg.get_path('camera.serial', '') or '')
            if devices.get_count() > 1 and not serial:
                raise RuntimeError('检测到多台相机，请配置 camera.serial')
            for i in range(devices.get_count()):
                candidate = devices.get_device_by_index(i)
                if not serial or candidate.get_device_info().get_serial_number() == serial:
                    self.device = candidate
                    break
            if self.device is None:
                raise RuntimeError(f'未找到相机 serial={serial or "未指定"}')
            self.pipeline = ob.Pipeline(self.device)
            profile = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                int(self.cfg.get_path('camera.color_width', 640)),
                int(self.cfg.get_path('camera.color_height', 480)),
                getattr(ob.OBFormat, str(self.cfg.get_path('camera.color_format', 'MJPG')).upper()),
                int(self.cfg.get_path('camera.color_fps', 30)))
            config = ob.Config()
            config.enable_stream(profile)
            self.pipeline.start(config)
            self.started = True
            self.stopping.clear()
            self.thread = threading.Thread(target=self._capture, name='orbbec-color', daemon=True)
            self.thread.start()
            print('二维识别：仅彩色流，后台持续取最新帧，不要求有效深度。', flush=True)
        except BaseException:
            self.close()
            raise

    def _capture(self):
        while not self.stopping.is_set():
            try:
                frames = self.pipeline.wait_for_frames(200)
                frame = frames.get_color_frame() if frames is not None else None
                image = frame_to_bgr(frame) if frame is not None else None
                if image is None:
                    continue
                with self.condition:
                    self.latest = ColorFrame(image, time.time_ns())
                    self.sequence += 1
                    self.error = None
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                    self.condition.notify_all()
                self.stopping.wait(0.05)

    def read(self, timeout_ms=400):
        if not self.started:
            raise RuntimeError('彩色相机未打开')
        with self.condition:
            fresh = self.condition.wait_for(
                lambda: self.sequence > self.last_read_sequence or self.stopping.is_set(),
                timeout=timeout_ms / 1000)
            if not fresh or self.sequence <= self.last_read_sequence:
                raise TimeoutError(f'彩色取流暂时超时：{self.error or "尚未收到新的彩色帧"}')
            self.last_read_sequence = self.sequence
            return self.latest

    def close(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        try:
            if self.started:
                self.pipeline.stop()
        finally:
            self.started = False
            self.pipeline = self.device = self.context = None
            self.latest = None
            self.sequence = self.last_read_sequence = 0


def open_color_camera(cfg=None):
    return OrbbecColorCamera(cfg)
