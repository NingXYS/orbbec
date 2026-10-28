"""DaBai DC1：SDK v1 硬件 D2C，输出彩色坐标系下的米制 RGB-D。"""

from __future__ import annotations

import logging
import sys
import time
from typing import Iterator, Optional

import numpy as np

from capture.color_convert import frame_to_bgr
from capture.types import Extrinsics, Intrinsics, RGBDFrame
from capture.usb_detect import DABAI_DC1_DEPTH_PID
from common.config import Config, load_config
from common.paths import ROOT

LOGGER = logging.getLogger("graspnet_orbbec.capture.dc1")


def _import_sdk_v1():
    existing = sys.modules.get("pyorbbecsdk")
    if existing is not None and "pyorbbecsdk2" in str(getattr(existing, "__file__", "")):
        raise RuntimeError("当前进程已加载 SDK v2；DC1 需要 SDK v1，请重启程序")
    sdk_dir = ROOT / "third_party" / "pyorbbecsdk-v1" / "install" / "lib"
    if existing is None:
        sys.path.insert(0, str(sdk_dir))
    try:
        import pyorbbecsdk as ob
    except ImportError as exc:
        raise RuntimeError("无法加载 SDK v1，请先 source scripts/activate.sh") from exc
    return ob


def _intrinsic(value, distortion) -> Intrinsics:
    k = Intrinsics(
        fx=float(value.fx), fy=float(value.fy), cx=float(value.cx), cy=float(value.cy),
        width=int(value.width), height=int(value.height),
        distortion=np.array([getattr(distortion, n) for n in
                             ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], dtype=np.float64),
    )
    if (k.width <= 0 or k.height <= 0 or k.fx <= 0 or k.fy <= 0
            or not np.isfinite([k.fx, k.fy, k.cx, k.cy]).all()
            or not np.isfinite(k.distortion).all()):
        raise RuntimeError("DC1 返回无效内参/畸变，停止采集")
    return k


class DaBaiDC1Camera:
    """与现有 open_camera 接口一致；只接受经过硬件 D2C 的完整 RGB-D 帧。

    DC1 没有 SDK 帧同步；接收时间差只用于过滤陈旧帧，不代表同步曝光。
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        self._context = None
        self._device = None
        self._pipeline = None
        self._opened = False
        self._color_k = None
        self._depth_k = None
        self._extra = None
        self._device_info: dict = {}
        self._profiles: dict = {}
        self._laser_previous = None
        self._laser_property = None
        self._last_pair = None

    def __enter__(self) -> "DaBaiDC1Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def device_info(self) -> dict:
        return dict(self._device_info)

    def open(self) -> None:
        if self._opened:
            return
        if str(self.cfg.get_path("camera.align_mode", "HW")).upper() != "HW":
            raise ValueError("DC1 后端要求 camera.align_mode=HW，不回退为图像缩放")
        ob = _import_sdk_v1()
        try:
            self._context = ob.Context()
            devices = self._context.query_devices()
            serial = str(self.cfg.get_path("camera.serial", "") or "")
            indices = [i for i in range(devices.get_count())
                       if devices.get_device_pid_by_index(i) == DABAI_DC1_DEPTH_PID]
            if len(indices) > 1 and not serial:
                raise RuntimeError("检测到多台 DC1，请配置 camera.serial")
            # OpenNI 枚举阶段的序列号可能为空，必须打开设备后再次核验。
            for i in indices:
                candidate = devices.get_device_by_index(i)
                info = candidate.get_device_info()
                if not serial or info.get_serial_number() == serial:
                    self._device = candidate
                    break
            if self._device is None:
                raise RuntimeError(f"未找到 DC1（serial={serial or '未指定'}），请检查连接并关闭 Viewer")
            info = self._device.get_device_info()
            self._device_info = {
                "name": info.get_name(), "pid": int(info.get_pid()), "vid": int(info.get_vid()),
                "serial": info.get_serial_number(), "firmware": info.get_firmware_version(),
            }
            self._laser_property = ob.OBPropertyID.OB_PROP_LASER_BOOL
            self._laser_previous = bool(self._device.get_bool_property(self._laser_property))
            # 发射器关闭会产生全零深度；只在采集期间开启，退出恢复原状态。
            self._device.set_bool_property(self._laser_property, bool(self.cfg.get_path("camera.laser", True)))
            self._pipeline = ob.Pipeline(self._device)
            config = ob.Config()
            for name, sensor, defaults in (
                ("color", ob.OBSensorType.COLOR_SENSOR, (640, 480, "MJPG")),
                ("depth", ob.OBSensorType.DEPTH_SENSOR, (640, 400, "Y11")),
            ):
                w = int(self.cfg.get_path(f"camera.{name}_width", defaults[0]))
                h = int(self.cfg.get_path(f"camera.{name}_height", defaults[1]))
                fps = int(self.cfg.get_path(f"camera.{name}_fps", 30))
                fmt = str(self.cfg.get_path(f"camera.{name}_format", defaults[2])).upper()
                profiles = self._pipeline.get_stream_profile_list(sensor)
                profile = profiles.get_video_stream_profile(w, h, getattr(ob.OBFormat, fmt), fps)
                config.enable_stream(profile)
                self._profiles[name] = {"width": w, "height": h, "fps": fps, "format": fmt}
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
            config.set_depth_scale_require(True)
            # 不调用 DC1 不支持的 enable_frame_sync；只筛选接收时间接近的完整帧。
            self._pipeline.start(config)
            if not self._device.get_bool_property(ob.OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL):
                raise RuntimeError("DC1 硬件 D2C 开关未生效，停止采集")
            param = self._pipeline.get_camera_param()
            self._color_k = _intrinsic(param.rgb_intrinsic, param.rgb_distortion)
            self._depth_k = _intrinsic(param.depth_intrinsic, param.depth_distortion)
            color_profile = self._profiles["color"]
            if (self._color_k.width, self._color_k.height) != (color_profile["width"], color_profile["height"]):
                raise RuntimeError("DC1 彩色内参尺寸与选定流配置不符")
            self._extra = Extrinsics(
                np.asarray(param.transform.rot, dtype=np.float64).reshape(3, 3),
                np.asarray(param.transform.transform, dtype=np.float64).reshape(3) / 1000.0,
            )
            # D2C 后参数应已处于彩色坐标系，禁止再套原始 Depth→RGB 外参。
            if (not np.allclose(self._color_k.as_matrix(), self._depth_k.as_matrix(), atol=1e-4)
                    or (self._color_k.width, self._color_k.height) != (self._depth_k.width, self._depth_k.height)
                    or not np.allclose(self._extra.rotation, np.eye(3), atol=1e-5)
                    or not np.allclose(self._extra.translation, 0, atol=1e-6)):
                raise RuntimeError("DC1 硬件 D2C 参数未映射到彩色坐标系，拒绝生成点云")
            startup_ms = int(self.cfg.get_path("camera.startup_timeout_ms", 10000))
            self.read(timeout_ms=startup_ms)
            for _ in range(int(self.cfg.get_path("camera.warmup_frames", 8))):
                self.read()
            self._opened = True
            LOGGER.info("DC1 已打开 %s；D2C=HW；RGB-D 不同步曝光", self._device_info)
        except BaseException:
            self.close()
            raise

    def get_intrinsics(self) -> Intrinsics:
        if self._color_k is None:
            raise RuntimeError("相机未打开")
        return self._color_k

    def read(self, timeout_ms: Optional[int] = None) -> RGBDFrame:
        if self._pipeline is None:
            raise RuntimeError("相机未打开")
        timeout_ms = int(timeout_ms if timeout_ms is not None else self.cfg.get_path("camera.timeout_ms", 2000))
        max_delta = float(self.cfg.get_path("camera.max_receive_delta_ms", 100))
        if timeout_ms <= 0 or not np.isfinite(max_delta) or max_delta <= 0:
            raise ValueError("DC1 timeout_ms 和 max_receive_delta_ms 必须为正数")
        deadline = time.monotonic() + timeout_ms / 1000.0
        reason = "未收到完整 RGB-D 帧"
        while time.monotonic() < deadline:
            frames = self._pipeline.wait_for_frames(max(1, min(400, int((deadline - time.monotonic()) * 1000))))
            if frames is None:
                continue
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue
            # 两路设备时间戳不共时钟，不能拿它们的差当曝光误差。
            d_us, c_us = int(depth.get_system_timestamp_us()), int(color.get_system_timestamp_us())
            delta_ms = abs(d_us - c_us) / 1000.0
            if d_us <= 0 or c_us <= 0 or delta_ms > max_delta:
                reason = f"RGB-D 主机接收差 {delta_ms:.1f} ms 超限（不是曝光同步误差）"
                continue
            if self._last_pair is not None and (d_us <= self._last_pair[0] or c_us <= self._last_pair[1]):
                reason = "RGB 或深度帧尚未更新"
                continue
            frame = self._decode_pair(depth, color)
            if not np.any(frame.depth_m > 0):
                reason = "有效深度全部为零，请检查遮挡、工作距离和激光发射器"
                continue
            frame.timestamp_ns = d_us * 1000
            frame.extra.update({"color_receive_timestamp_us": c_us, "depth_receive_timestamp_us": d_us,
                                "receive_delta_ms": delta_ms})
            self._last_pair = (d_us, c_us)
            return frame
        raise TimeoutError(f"DC1 取流超时：{reason}")

    def _decode_pair(self, depth, color) -> RGBDFrame:
        color_bgr = frame_to_bgr(color)
        if color_bgr is None:
            raise RuntimeError("DC1 彩色帧解码失败")
        k = self.get_intrinsics()
        shape = (k.height, k.width)
        if color_bgr.shape[:2] != shape or (depth.get_height(), depth.get_width()) != shape:
            raise RuntimeError("DC1 D2C 输出尺寸与彩色内参不符，拒绝缩放深度图")
        scale = float(depth.get_depth_scale())
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("DC1 深度缩放系数无效")
        # 请求 Y11，SDK 实际解包为 uint16/Y16；单位取 SDK scale，不按 11 位移位。
        if str(depth.get_format()).split(".")[-1] != "Y16":
            raise RuntimeError(f"DC1 需要解包后的 Y16 深度，实际为 {depth.get_format()}")
        data = depth.get_data()
        if memoryview(data).nbytes != k.width * k.height * 2:
            raise RuntimeError("DC1 深度缓冲区长度异常")
        raw = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        depth_m = raw.astype(np.float32) * (scale / 1000.0)
        zmin = float(self.cfg.get_path("camera.min_depth_m", 0.30))
        zmax = float(self.cfg.get_path("camera.max_depth_m", 2.50))
        depth_m = np.where((depth_m >= zmin) & (depth_m <= zmax), depth_m, 0.0).astype(np.float32)
        return RGBDFrame(
            color_bgr=color_bgr, depth_m=depth_m,
            depth_u16=np.rint(np.clip(depth_m * 1000, 0, 65535)).astype(np.uint16),
            color_intrinsics=k, depth_intrinsics=self._depth_k, depth_to_color=self._extra,
            aligned=True, depth_scale=scale, backend="dabai_dc1",
            extra={"device": self.device_info, "align_mode": "HW", "frame_sync": False,
                   "depth_geometry": "sdk_hardware_d2c_color_frame", "geometry_verified": False,
                   "profiles": self._profiles, "depth_actual_format": "Y16"},
        )

    def frames(self) -> Iterator[RGBDFrame]:
        while True:
            yield self.read()

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                LOGGER.warning("DC1 停流异常: %s", exc)
        if self._device is not None and self._laser_previous is not None:
            try:
                self._device.set_bool_property(self._laser_property, self._laser_previous)
            except Exception as exc:
                LOGGER.warning("DC1 恢复发射器开关失败: %s", exc)
        self._laser_previous = self._laser_property = None
        self._device = None
        self._context = None
        self._opened = False
        self._color_k = self._depth_k = self._extra = None
        self._last_pair = None
