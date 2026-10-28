"""DC1 的几何、单位、陈旧帧和失败清理约束；不依赖相机硬件。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import dabai_dc1
from capture.camera import load_rgbd_dir, open_camera, save_rgbd_dir
from capture.dabai_dc1 import DaBaiDC1Camera
from capture.types import Extrinsics, Intrinsics
from capture import usb_detect
from common.config import Config


class Frame:
    def __init__(self, data, fmt, received=200_000, scale=0.5):
        self.data, self.fmt, self.received, self.scale = data, fmt, received, scale

    def get_data(self): return self.data
    def get_format(self): return self.fmt
    def get_width(self): return self.data.shape[1]
    def get_height(self): return self.data.shape[0]
    def get_depth_scale(self): return self.scale
    def get_system_timestamp_us(self): return self.received


def pair(depth_shape=(3, 4), color_received=199_000, value=1000, scale=0.5, fmt="Y16"):
    depth = Frame(np.full(depth_shape, value, np.uint16), fmt, scale=scale)
    color = Frame(np.full((3, 4, 3), 120, np.uint8), "BGR", received=color_received)
    return NS(get_depth_frame=lambda: depth, get_color_frame=lambda: color)


@pytest.fixture
def camera():
    cfg = Config({"camera": {"backend": "dabai_dc1", "min_depth_m": 0.3, "max_depth_m": 2}})
    cam = DaBaiDC1Camera(cfg)
    cam._color_k = Intrinsics(200, 200, 2, 1, 4, 3)
    cam._depth_k = Intrinsics(200, 200, 2, 1, 4, 3)
    cam._extra = Extrinsics(np.eye(3), np.zeros(3))
    return cam


def test_dispatch_and_usb_detection(monkeypatch):
    assert isinstance(open_camera(Config({"camera": {"backend": "dc1"}})), DaBaiDC1Camera)
    monkeypatch.setattr(usb_detect, "list_orbbec_usb", lambda: [{"pid": 0x0657}, {"pid": 0x0557}])
    assert usb_detect.detect_capture_backend() == "dabai_dc1"


def test_depth_scale_and_saved_roundtrip(camera, tmp_path):
    fs = pair()
    frame = camera._decode_pair(fs.get_depth_frame(), fs.get_color_frame())
    np.testing.assert_allclose(frame.depth_m, 0.5)
    np.testing.assert_array_equal(frame.depth_u16, 500)
    assert frame.aligned and frame.extra["frame_sync"] is False
    save_rgbd_dir(frame, tmp_path)
    np.testing.assert_allclose(load_rgbd_dir(tmp_path).depth_m, frame.depth_m)
    meta = json.loads((tmp_path / "intrinsics.json").read_text())
    assert meta["source_depth_scale_mm"] == 0.5
    assert meta["capture_metadata"]["geometry_verified"] is False


@pytest.mark.parametrize("kwargs", [
    {"depth_shape": (2, 4)}, {"scale": 0}, {"scale": float("nan")},
    {"scale": float("inf")}, {"fmt": "Y11"},
])
def test_bad_geometry_or_scale_is_rejected(camera, kwargs):
    fs = pair(**kwargs)
    with pytest.raises(RuntimeError):
        camera._decode_pair(fs.get_depth_frame(), fs.get_color_frame())


def test_stale_pair_is_skipped_and_receive_clock_is_saved(camera):
    sequence = iter([pair(color_received=1), pair()])
    camera._pipeline = NS(wait_for_frames=lambda timeout: next(sequence))
    frame = camera.read()
    assert frame.timestamp_ns == 200_000_000
    assert frame.extra["receive_delta_ms"] == 1
    assert frame.extra["frame_sync"] is False


def test_cached_frames_are_not_counted_again(camera):
    camera._last_pair = (200_000, 199_000)
    fresh = pair(color_received=230_000)
    fresh.get_depth_frame().received = 235_000
    sequence = iter([pair(), fresh])
    camera._pipeline = NS(wait_for_frames=lambda timeout: next(sequence))
    frame = camera.read()
    assert frame.timestamp_ns == 235_000_000


def test_zero_depth_is_not_returned_as_usable(camera):
    camera._pipeline = NS(wait_for_frames=lambda timeout: pair(value=0))
    with pytest.raises(TimeoutError, match="深度全部为零"):
        camera.read(timeout_ms=1)


def test_wrong_serial_does_not_start_camera(monkeypatch):
    device = NS(get_device_info=lambda: NS(get_serial_number=lambda: "other-camera"))
    devices = NS(get_count=lambda: 1, get_device_pid_by_index=lambda i: 0x0657,
                 get_device_by_index=lambda i: device)
    monkeypatch.setattr(dabai_dc1, "_import_sdk_v1", lambda: NS(Context=lambda: NS(query_devices=lambda: devices)))
    cam = DaBaiDC1Camera(Config({"camera": {"serial": "expected-camera"}}))
    with pytest.raises(RuntimeError, match="未找到 DC1"):
        cam.open()
    assert cam._pipeline is None and cam._device is None


@pytest.mark.parametrize("failure", [RuntimeError("start failed"), KeyboardInterrupt()])
def test_failed_start_stops_pipeline_and_restores_laser(monkeypatch, failure):
    state = {"laser": False, "stopped": False}
    info = NS(get_serial_number=lambda: "test", get_pid=lambda: 0x0657, get_vid=lambda: 0x2BC5,
              get_name=lambda: "DaBai DC1", get_firmware_version=lambda: "RD1001")
    device = NS(get_device_info=lambda: info, get_bool_property=lambda prop: state["laser"],
                set_bool_property=lambda prop, value: state.update(laser=value))
    devices = NS(get_count=lambda: 1, get_device_pid_by_index=lambda i: 0x0657,
                 get_device_by_index=lambda i: device)
    def fail_start(config):
        raise failure
    pipe = NS(get_stream_profile_list=lambda sensor: NS(get_video_stream_profile=lambda *args: object()),
              start=fail_start, stop=lambda: state.update(stopped=True))
    sdk = NS(Context=lambda: NS(query_devices=lambda: devices), Pipeline=lambda dev: pipe,
             OBPropertyID=NS(OB_PROP_LASER_BOOL=3), OBSensorType=NS(COLOR_SENSOR=1, DEPTH_SENSOR=2),
             OBFormat=NS(MJPG=1, Y11=2), OBAlignMode=NS(HW_MODE=1),
             Config=lambda: NS(enable_stream=lambda profile: None, set_align_mode=lambda mode: None,
                               set_depth_scale_require=lambda enabled: None))
    monkeypatch.setattr(dabai_dc1, "_import_sdk_v1", lambda: sdk)
    cam = DaBaiDC1Camera(Config({"camera": {"backend": "dc1"}}))
    with pytest.raises(type(failure)):
        cam.open()
    assert state == {"laser": False, "stopped": True}
    assert cam._pipeline is None and cam._context is None
