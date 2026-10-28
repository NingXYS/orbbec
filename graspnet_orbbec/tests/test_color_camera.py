"""彩色识别不要求深度，持续采集只保留最新帧，超时不重用陈旧帧。"""

from pathlib import Path
import queue
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import Config
from capture.color_camera import OrbbecColorCamera, FileColorCamera


def test_color_file_replay_does_not_require_a_depth_file(tmp_path):
    import cv2
    image=np.full((8,10,3),100,np.uint8)
    cv2.imwrite(str(tmp_path/'color.png'),image)
    camera=FileColorCamera(tmp_path)
    assert np.array_equal(camera.read().color_bgr,image)
    assert np.array_equal(camera.read().color_bgr,image)
    assert not hasattr(camera.read(),'depth_m')


@pytest.fixture
def camera(monkeypatch):
    import capture.dabai_dc1 as dc1
    import capture.color_camera as module
    packets = queue.Queue()
    calls = []

    class Pipeline:
        def get_stream_profile_list(self, sensor):
            assert sensor == 'COLOR'
            return NS(get_video_stream_profile=lambda *a: 'color_profile')

        def start(self, config):
            assert config.profiles == ['color_profile']
            calls.append('start')

        def wait_for_frames(self, timeout):
            try:
                packet = packets.get(timeout=timeout/1000)
            except queue.Empty:
                return None
            if isinstance(packet, Exception):
                raise packet
            return NS(get_color_frame=lambda: packet,
                      get_depth_frame=lambda: pytest.fail('二维取流不能要求深度'))

        def stop(self): calls.append('stop')

    class StreamConfig:
        def __init__(self): self.profiles = []
        def enable_stream(self, profile): self.profiles.append(profile)

    device = NS(get_device_info=lambda: NS(get_serial_number=lambda: 'test'))
    devices = NS(get_count=lambda: 1, get_device_by_index=lambda i: device)
    sdk = NS(Context=lambda: NS(query_devices=lambda: devices), Pipeline=lambda d: Pipeline(),
             Config=StreamConfig, OBSensorType=NS(COLOR_SENSOR='COLOR'), OBFormat=NS(MJPG='MJPG'))
    monkeypatch.setattr(dc1, '_import_sdk_v1', lambda: sdk)
    monkeypatch.setattr(module, 'frame_to_bgr', lambda frame: frame)
    cam = OrbbecColorCamera(Config({'camera': {'backend': 'dabai_dc1', 'serial': 'test'}}))
    cam.open()
    try:
        yield cam, packets, calls
    finally:
        cam.close()
    assert calls == ['start', 'stop']


def test_color_only_stream_works_without_depth(camera):
    cam, packets, calls = camera
    packets.put(np.ones((8, 10, 3), np.uint8))
    frame = cam.read(timeout_ms=1000)
    assert frame.color_bgr.shape == (8, 10, 3)
    assert frame.timestamp_ns > 0
    assert not hasattr(frame, 'depth_m')


def test_reader_keeps_latest_not_full_native_queue(camera):
    cam, packets, _ = camera
    for value in (1, 2, 3, 4):
        packets.put(np.full((8,10,3), value, np.uint8))
    with cam.condition:
        assert cam.condition.wait_for(lambda: cam.sequence >= 4, timeout=2)
    frame = cam.read(timeout_ms=100)
    assert np.all(frame.color_bgr == 4)
    with pytest.raises(TimeoutError, match='尚未收到新的彩色帧'):
        cam.read(timeout_ms=10)


def test_capture_error_does_not_return_old_frame(camera):
    cam, packets, _ = camera
    packets.put(np.ones((8,10,3), np.uint8))
    cam.read(timeout_ms=1000)
    packets.put(RuntimeError('USB interrupted'))
    with cam.condition:
        assert cam.condition.wait_for(lambda: cam.error is not None, timeout=2)
    with pytest.raises(TimeoutError, match='USB interrupted'):
        cam.read(timeout_ms=10)
    packets.put(np.full((8,10,3), 2, np.uint8))
    assert np.all(cam.read(timeout_ms=1000).color_bgr == 2)
    assert cam.error is None


def test_close_joins_capture_before_destroying_sdk(camera):
    cam, packets, calls = camera
    thread = cam.thread
    cam.close()
    assert not thread.is_alive()
    assert cam.pipeline is None and not cam.started
    assert calls == ['start', 'stop']
    cam.close()
    assert calls == ['start', 'stop']
