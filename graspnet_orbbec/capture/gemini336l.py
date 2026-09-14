"""Gemini 336L 取流（PyOrbbecSDK v2.x）。

与 AstraCamera 同接口：open / close / read / frames / get_intrinsics / 上下文管理器。
默认 1280x800 软件 AlignFilter（固件 1.3.70 硬件 D2C 最高 640x480）。帧同步 + 激光开。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import yaml

from capture.color_convert import frame_to_bgr
from capture.sdk_v2 import import_sdk_v2
from capture.types import Extrinsics, Intrinsics, RGBDFrame
from capture.usb_detect import GEMINI_336L_PID, ORBBEC_VID
from common.config import Config as AppConfig
from common.config import load_config
from common.paths import CONFIG_DIR

LOGGER = logging.getLogger("graspnet_orbbec.capture.gemini336l")

# 336L 验收阈值（严于 Astra 的 10 px / 5 mm）
ALIGN_PASS_PX = 3.0
BLOCK_PASS_MM = 3.0
MIN_FPS = 15.0

# Gemini 330：0=关 1=开 2=自动
_LASER_OFF, _LASER_ON, _LASER_AUTO = 0, 1, 2


def _fmt_name(fmt) -> str:
    return str(fmt).split(".")[-1].upper()


def _intrinsic_from_ob(intr, distortion=None) -> Intrinsics:
    dist = None
    if distortion is not None:
        try:
            dist = np.array(
                [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3],
                dtype=np.float64,
            )
        except Exception:
            dist = None
    return Intrinsics(
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.cx),
        cy=float(intr.cy),
        width=int(intr.width),
        height=int(intr.height),
        distortion=dist,
    )


def _extrinsic_from_ob(transform) -> Extrinsics:
    rot = np.array(transform.rot, dtype=np.float64).reshape(3, 3)
    t = np.array(transform.transform, dtype=np.float64).reshape(3)
    # 官方文档：v2 外参平移单位为毫米
    if float(np.linalg.norm(t)) > 2.0:
        t = t / 1000.0
    return Extrinsics(rotation=rot, translation=t)


def _format_penalty(fmt) -> int:
    name = _fmt_name(fmt)
    order = {"RGB": 0, "BGR": 1, "YUYV": 2, "YUY2": 2, "MJPG": 3, "MJPEG": 3}
    return order.get(name, 8)


def _pick_hw_d2c_profiles(pipeline, ob, width: int, height: int, fps: int):
    """在硬件 D2C 可用的 color/depth 组合里，挑最接近目标分辨率的一对。"""
    color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    best = None
    best_score = 1e18
    for i in range(len(color_list)):
        color_profile = color_list[i]
        try:
            cw, ch, cf = color_profile.get_width(), color_profile.get_height(), color_profile.get_fps()
            cfmt = color_profile.get_format()
        except Exception:
            continue
        if _format_penalty(cfmt) > 3:
            continue
        try:
            hw_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
        except Exception:
            continue
        if hw_list is None or len(hw_list) == 0:
            continue
        depth_profile = _pick_matching_depth(hw_list, cw, ch, cf if cf else fps)
        score = abs(cw - width) + abs(ch - height) + 0.15 * abs(cf - fps) + 8 * _format_penalty(cfmt)
        if cf < 15:
            score += 80
        if score < best_score:
            best_score = score
            best = (color_profile, depth_profile)
    return best


def _pick_stream_profile(profile_list, width: int, height: int, fps: int, formats: tuple[str, ...]):
    """按分辨率/帧率/格式打分挑选流。formats 如 ("RGB", "BGR")。"""
    best = None
    best_score = 1e18
    for i in range(len(profile_list)):
        p = profile_list[i]
        try:
            w, h, f = p.get_width(), p.get_height(), p.get_fps()
            name = _fmt_name(p.get_format())
        except Exception:
            continue
        if name not in formats:
            continue
        score = abs(w - width) + abs(h - height) + 0.15 * abs(f - fps) + 8 * _format_penalty(p.get_format())
        if f < 15:
            score += 80
        if score < best_score:
            best_score = score
            best = p
    return best


def _pick_matching_depth(hw_list, width: int, height: int, fps: int):
    scored = []
    for j in range(len(hw_list)):
        dp = hw_list[j]
        try:
            dw, dh, df = dp.get_width(), dp.get_height(), dp.get_fps()
        except Exception:
            continue
        # 同分辨率同帧率最稳；硬件 D2C 输出会被拉到彩色图尺寸
        score = abs(dw - width) + abs(dh - height) + 0.2 * abs(df - fps)
        scored.append((score, dp))
    if not scored:
        return hw_list[0]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _set_laser(device, ob, enable: bool) -> dict:
    """336L 不支持 OB_PROP_LASER_BOOL，用 LASER_CONTROL_INT。"""
    info = {"requested": bool(enable), "applied": None, "property": None}
    permission = ob.OBPermissionType.PERMISSION_READ_WRITE
    control_id = ob.OBPropertyID.OB_PROP_LASER_CONTROL_INT
    bool_id = ob.OBPropertyID.OB_PROP_LASER_BOOL
    target = _LASER_ON if enable else _LASER_OFF
    try:
        if device.is_property_supported(control_id, permission):
            device.set_int_property(control_id, target)
            info["property"] = "OB_PROP_LASER_CONTROL_INT"
            info["applied"] = int(device.get_int_property(control_id))
            return info
    except Exception as e:
        LOGGER.warning("设置 LASER_CONTROL_INT 失败: %s", e)
    try:
        if device.is_property_supported(bool_id, permission):
            device.set_bool_property(bool_id, bool(enable))
            info["property"] = "OB_PROP_LASER_BOOL"
            info["applied"] = bool(device.get_bool_property(bool_id))
            return info
    except Exception as e:
        LOGGER.warning("设置 LASER_BOOL 失败: %s", e)
    LOGGER.warning("设备不支持激光开关属性")
    return info


def _select_device(ob, serial: Optional[str] = None):
    ctx = ob.Context()
    device_list = ctx.query_devices()
    n = int(device_list.get_count())
    if n <= 0:
        raise RuntimeError("未找到 Orbbec 设备。请检查 USB 与 udev 规则。")
    chosen = None
    for i in range(n):
        dev = device_list.get_device_by_index(i)
        info = dev.get_device_info()
        sn = info.get_serial_number() if hasattr(info, "get_serial_number") else ""
        pid = int(info.get_pid())
        if serial and sn == serial:
            chosen = dev
            break
        if chosen is None and pid == GEMINI_336L_PID:
            chosen = dev
        if chosen is None:
            chosen = dev
    return ctx, chosen


class Gemini336LCamera:
    """Gemini 336L 统一取流接口（对齐后的 RGB-D）。"""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()
        self._ob = None
        self._ctx = None
        self._pipeline = None
        self._opened = False
        self._align_mode = str(self.cfg.get_path("camera.align_mode", "HW")).upper()
        self._color_k: Optional[Intrinsics] = None
        self._depth_k: Optional[Intrinsics] = None
        self._extra: Optional[Extrinsics] = None
        self._device_info: dict = {}
        self._profile_info: dict = {}
        self._laser_info: dict = {}
        self._align_filter = None
        self._frame_index = 0
        self._intrinsics_path = Path(
            str(self.cfg.get_path("camera.intrinsics_file", CONFIG_DIR / "gemini336l_intrinsics.yaml"))
        )
        if not self._intrinsics_path.is_absolute():
            from common.paths import ROOT

            self._intrinsics_path = ROOT / self._intrinsics_path

    def __enter__(self) -> "Gemini336LCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def device_info(self) -> dict:
        return dict(self._device_info)

    def open(self) -> None:
        if self._opened:
            return
        ob = import_sdk_v2()
        self._ob = ob
        serial = self.cfg.get_path("camera.serial", None)
        self._ctx, device = _select_device(ob, serial=serial)
        info = device.get_device_info()
        pid = int(info.get_pid())
        self._device_info = {
            "name": info.get_name() if hasattr(info, "get_name") else "Gemini 336L",
            "pid": pid,
            "vid": int(info.get_vid()) if hasattr(info, "get_vid") else ORBBEC_VID,
            "serial": info.get_serial_number() if hasattr(info, "get_serial_number") else "",
            "firmware": info.get_firmware_version() if hasattr(info, "get_firmware_version") else "",
            "sdk": getattr(ob, "__version__", "2.x"),
        }
        LOGGER.info("打开设备: %s", self._device_info)
        if pid != GEMINI_336L_PID:
            LOGGER.warning("当前 PID=0x%04X，模块按 Gemini 330 系列硬件 D2C 路径工作", pid)

        self._laser_info = _set_laser(device, ob, bool(self.cfg.get_path("camera.laser", True)))
        LOGGER.info("激光: %s", self._laser_info)

        pipeline = ob.Pipeline(device)
        want_w = int(self.cfg.get_path("camera.color_width", 1280))
        want_h = int(self.cfg.get_path("camera.color_height", 800))
        want_fps = int(self.cfg.get_path("camera.color_fps", 30))
        want_dw = int(self.cfg.get_path("camera.depth_width", want_w))
        want_dh = int(self.cfg.get_path("camera.depth_height", want_h))
        color_profile = None
        depth_profile = None
        use_hw = self._align_mode == "HW"

        if use_hw:
            picked = _pick_hw_d2c_profiles(pipeline, ob, want_w, want_h, want_fps)
            if picked is not None:
                color_profile, depth_profile = picked
                # 请求 1280x800 时硬件 D2C 只有 640x480，视为不匹配，改走软件对齐
                mismatch = abs(color_profile.get_width() - want_w) + abs(color_profile.get_height() - want_h)
                if mismatch > 160:
                    LOGGER.info(
                        "硬件 D2C 最接近 %dx%d，与请求 %dx%d 差距大，改用软件 AlignFilter",
                        color_profile.get_width(),
                        color_profile.get_height(),
                        want_w,
                        want_h,
                    )
                    color_profile = depth_profile = None
                    use_hw = False
            else:
                LOGGER.info("无硬件 D2C 组合，改用软件对齐")
                use_hw = False

        if not use_hw:
            self._align_mode = "SW"
            color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            depth_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            color_profile = _pick_stream_profile(color_list, want_w, want_h, want_fps, ("RGB", "BGR", "YUYV", "MJPG"))
            depth_profile = _pick_stream_profile(depth_list, want_dw, want_dh, want_fps, ("Y16",))
            if color_profile is None or depth_profile is None:
                raise RuntimeError(f"无法启用彩色/深度流 {want_w}x{want_h} + {want_dw}x{want_dh}")

        config = ob.Config()
        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        if use_hw:
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
        else:
            try:
                config.set_align_mode(ob.OBAlignMode.SW_MODE)
            except Exception as e:
                LOGGER.warning("set_align_mode(SW) 失败: %s", e)
        try:
            config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            LOGGER.warning("set_frame_aggregate_output_mode 失败（可忽略）: %s", e)
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            LOGGER.warning("enable_frame_sync 失败（可忽略）: %s", e)

        if use_hw:
            try:
                if device.is_property_supported(
                    ob.OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL,
                    ob.OBPermissionType.PERMISSION_READ_WRITE,
                ):
                    device.set_bool_property(ob.OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, True)
            except Exception as e:
                LOGGER.warning("OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL 设置失败（已走 Config HW_MODE）: %s", e)
        else:
            try:
                self._align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            except Exception as e:
                LOGGER.warning("AlignFilter 不可用: %s", e)
                self._align_filter = None

        pipeline.start(config)
        self._pipeline = pipeline
        self._profile_info = {
            "color": f"{color_profile.get_width()}x{color_profile.get_height()}"
            f"@{color_profile.get_fps()} {_fmt_name(color_profile.get_format())}",
            "depth": f"{depth_profile.get_width()}x{depth_profile.get_height()}"
            f"@{depth_profile.get_fps()} {_fmt_name(depth_profile.get_format())}",
            "align": self._align_mode,
        }
        LOGGER.info("流配置 %s", self._profile_info)
        time.sleep(0.35)

        last_err = None
        warmup = int(self.cfg.get_path("camera.warmup_frames", 8))
        for _ in range(max(3, warmup)):
            try:
                self._refresh_calibration()
                self.read()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.08)
        if last_err is not None:
            LOGGER.warning("预热未拿到完整帧，继续尝试: %s", last_err)
            self._refresh_calibration()
        self._save_intrinsics_file()
        self._opened = True

    def _refresh_calibration(self) -> None:
        param = self._pipeline.get_camera_param()
        rgb_dist = getattr(param, "rgb_distortion", None)
        depth_dist = getattr(param, "depth_distortion", None)
        self._color_k = _intrinsic_from_ob(param.rgb_intrinsic, rgb_dist)
        self._depth_k = _intrinsic_from_ob(param.depth_intrinsic, depth_dist)
        try:
            self._extra = _extrinsic_from_ob(param.transform)
        except Exception as e:
            LOGGER.warning("读取 depth→color 外参失败: %s", e)
            self._extra = Extrinsics(np.eye(3), np.zeros(3))
        LOGGER.info("彩色内参 %s", self._color_k)
        LOGGER.info("深度内参 %s", self._depth_k)
        if self._extra is not None:
            LOGGER.info("D2C 外参 t(m)=%s", self._extra.translation.tolist())

    def _save_intrinsics_file(self) -> None:
        if self._color_k is None or self._depth_k is None:
            return
        payload = {
            "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "device": self._device_info,
            "profile": self._profile_info,
            "laser": self._laser_info,
            "align_mode": self._align_mode,
            "color_intrinsics": self._color_k.to_dict(),
            "depth_intrinsics": self._depth_k.to_dict(),
        }
        if self._extra is not None:
            payload["depth_to_color"] = self._extra.to_dict()
            payload["depth_to_color_note"] = "P_color = R @ P_depth + t，t 单位米"
        self._intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        self._intrinsics_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        LOGGER.info("内参已写入 %s", self._intrinsics_path)

    def get_intrinsics(self) -> Intrinsics:
        if self._color_k is None:
            raise RuntimeError("相机未打开")
        return self._color_k

    def read(self, timeout_ms: Optional[int] = None) -> RGBDFrame:
        if self._pipeline is None:
            raise RuntimeError("相机未打开")
        timeout_ms = timeout_ms or int(self.cfg.get_path("camera.timeout_ms", 2000))
        frames = None
        depth_frame = None
        color_frame = None
        deadline = time.time() + max(timeout_ms / 1000.0, 0.2)
        while time.time() < deadline:
            frames = self._pipeline.wait_for_frames(min(timeout_ms, 400))
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is not None and color_frame is not None:
                break
            time.sleep(0.01)
        if frames is None or depth_frame is None or color_frame is None:
            raise TimeoutError("等待 RGB-D 帧超时")

        if self._align_filter is not None:
            try:
                aligned_set = self._align_filter.process(frames)
                if aligned_set is not None:
                    frames = aligned_set
                    depth_frame = frames.get_depth_frame() or depth_frame
                    color_frame = frames.get_color_frame() or color_frame
            except Exception as e:
                if self._frame_index < 3:
                    LOGGER.warning("AlignFilter.process 失败: %s", e)

        color_bgr = frame_to_bgr(color_frame)
        if color_bgr is None:
            raise RuntimeError("彩色帧无法解码")

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        scale = float(depth_frame.get_depth_scale())
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw).copy()
        depth_m = depth_raw.astype(np.float32) * scale / 1000.0
        raw_valid = depth_m > 1e-6
        self._frame_index += 1
        if np.any(raw_valid):
            raw_mm = depth_m[raw_valid] * 1000.0
            if self._frame_index <= 5 or self._frame_index % 120 == 0:
                LOGGER.info(
                    "深度 raw valid=%d/%d scale=%.4f mm_min=%.1f mm_med=%.1f mm_max=%.1f",
                    int(raw_valid.sum()),
                    depth_raw.size,
                    scale,
                    float(raw_mm.min()),
                    float(np.median(raw_mm)),
                    float(raw_mm.max()),
                )
        elif self._frame_index <= 5:
            LOGGER.warning("深度 raw 全零")

        min_z = float(self.cfg.get_path("camera.min_depth_m", 0.15))
        max_z = float(self.cfg.get_path("camera.max_depth_m", 2.0))
        depth_m = np.where((depth_m >= min_z) & (depth_m <= max_z), depth_m, 0.0).astype(np.float32)
        kept = int((depth_m > 0).sum())
        if kept < max(1000, depth_m.size // 50):
            LOGGER.warning(
                "裁剪后有效深度过少 kept=%d 范围=[%.2f, %.2f] m；请把相机放到工作距（约 0.5 m）或放宽 camera.max_depth_m",
                kept,
                min_z,
                max_z,
            )

        color_k = self._color_k
        depth_k = self._depth_k
        assert color_k is not None and depth_k is not None

        ch, cw = color_bgr.shape[:2]
        aligned = depth_m.shape == (ch, cw)
        if not aligned:
            # 硬件 D2C 正常时应同尺寸；若固件偶发不一致，最近邻拉到彩色图，并标 aligned=False
            LOGGER.warning("D2C 后尺寸不一致 depth=%s color=%s，最近邻缩放到彩色", depth_m.shape, (ch, cw))
            depth_m = cv2.resize(depth_m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            aligned = False
        if color_k.width != cw or color_k.height != ch:
            sx = cw / max(color_k.width, 1)
            sy = ch / max(color_k.height, 1)
            color_k = Intrinsics(
                fx=color_k.fx * sx,
                fy=color_k.fy * sy,
                cx=color_k.cx * sx,
                cy=color_k.cy * sy,
                width=cw,
                height=ch,
                distortion=color_k.distortion,
            )

        depth_u16 = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        try:
            ts = int(depth_frame.get_timestamp())
        except Exception:
            ts = time.time_ns()

        return RGBDFrame(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_m=depth_m.astype(np.float32),
            color_intrinsics=color_k,
            depth_intrinsics=depth_k,
            aligned=aligned,
            timestamp_ns=ts,
            depth_scale=scale,
            depth_to_color=self._extra,
            backend="gemini336l",
            extra={
                "device": self._device_info,
                "align_mode": self._align_mode,
                "profile": self._profile_info,
                "laser": self._laser_info,
                "intrinsics_file": str(self._intrinsics_path),
            },
        )

    def frames(self) -> Iterator[RGBDFrame]:
        while True:
            yield self.read()

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align_filter = None
        self._ctx = None
        self._opened = False
