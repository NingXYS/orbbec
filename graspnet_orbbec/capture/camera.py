"""Astra Pro Plus 取流。

优先 PyOrbbecSDK v1（Astra Pro Plus 在 v1 为 limited maintenance，v2 不支持）。
彩色传感器若 SDK 打不开，则回退到 OpenCV UVC（PID 0x050F），再用内参/外参做软件 D2C。
离线回放走 FileCamera。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from capture.color_convert import frame_to_bgr
from capture.types import Extrinsics, Intrinsics, RGBDFrame
from common.config import Config, load_config

LOGGER = logging.getLogger("graspnet_orbbec.capture")

ORBBEC_VID = 0x2BC5
ASTRA_PRO_PLUS_DEPTH_PID = 0x060F
ASTRA_PRO_PLUS_COLOR_PID = 0x050F


def _intrinsic_from_ob(intr) -> Intrinsics:
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
    return Extrinsics(rotation=rot, translation=t)


def _pick_video_profile(profile_list, width: int, height: int, fps: int):
    """按目标分辨率挑选流配置，找不到则退回默认。"""
    default = profile_list.get_default_video_stream_profile()
    best = default
    try:
        count = profile_list.get_count()
    except Exception:
        return default
    scored = []
    for i in range(count):
        p = profile_list.get_stream_profile_by_index(i)
        try:
            w, h, f = p.get_width(), p.get_height(), p.get_fps()
        except Exception:
            continue
        score = abs(w - width) + abs(h - height) + 0.1 * abs(f - fps)
        scored.append((score, p, w, h, f, str(p.get_format())))
    if not scored:
        return default
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    LOGGER.info(
        "选用流配置 %dx%d@%s %s",
        scored[0][2],
        scored[0][3],
        scored[0][4],
        scored[0][5],
    )
    return best


def find_orbbec_rgb_v4l_index() -> Optional[int]:
    """按 USB VID/PID 定位 Astra Pro Plus 的 UVC 彩色相机，避开笔记本内置摄像头。"""
    sysfs = Path("/sys/class/video4linux")
    if not sysfs.exists():
        return None
    candidates = []
    for node in sorted(sysfs.iterdir()):
        if not node.name.startswith("video"):
            continue
        try:
            index = int(node.name.replace("video", ""))
        except ValueError:
            continue
        # 向上找 USB 设备
        device = (node / "device").resolve()
        vid, pid = None, None
        cur = device
        for _ in range(8):
            vendor = cur / "idVendor"
            product = cur / "idProduct"
            if vendor.exists() and product.exists():
                try:
                    vid = int(vendor.read_text().strip(), 16)
                    pid = int(product.read_text().strip(), 16)
                except ValueError:
                    vid, pid = None, None
                break
            cur = cur.parent
        if vid == ORBBEC_VID and pid == ASTRA_PRO_PLUS_COLOR_PID:
            name = ""
            name_file = node / "name"
            if name_file.exists():
                name = name_file.read_text().strip()
            candidates.append((index, name))
    # 优先能打开的 capture 节点（偶数号常见）
    for index, name in candidates:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                LOGGER.info("UVC 彩色相机 /dev/video%d (%s)", index, name)
                return index
        else:
            cap.release()
    return candidates[0][0] if candidates else None


def align_depth_to_color(
    depth_m: np.ndarray,
    depth_k: Intrinsics,
    color_k: Intrinsics,
    extra: Extrinsics,
) -> np.ndarray:
    """软件 D2C：把深度图变到彩色图像坐标系，Z 取彩色相机坐标系下的深度（米）。"""
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    valid = z > 1e-6
    x = (us - np.float32(depth_k.cx)) * z / np.float32(depth_k.fx)
    y = (vs - np.float32(depth_k.cy)) * z / np.float32(depth_k.fy)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    mask = valid.reshape(-1)
    if not np.any(mask):
        return np.zeros((color_k.height, color_k.width), dtype=np.float32)

    R = extra.rotation.astype(np.float32)
    t = extra.translation.astype(np.float32)
    # 出厂外参平移偶发以毫米给出，超过 2 m 视为毫米
    if float(np.linalg.norm(t)) > 2.0:
        t = t / 1000.0
    pts_c = pts[mask] @ R.T + t
    zc = pts_c[:, 2]
    good = zc > 1e-6
    pts_c = pts_c[good]
    zc = zc[good]
    uc = np.rint(color_k.fx * pts_c[:, 0] / zc + color_k.cx).astype(np.int32)
    vc = np.rint(color_k.fy * pts_c[:, 1] / zc + color_k.cy).astype(np.int32)
    inb = (uc >= 0) & (uc < color_k.width) & (vc >= 0) & (vc < color_k.height)
    aligned = np.zeros((color_k.height, color_k.width), dtype=np.float32)
    if not np.any(inb):
        return aligned
    # 远到近写入，近处覆盖远处（简易 z-buffer）
    order = np.argsort(-zc[inb])
    aligned[vc[inb][order], uc[inb][order]] = zc[inb][order]
    return aligned


class AstraCamera:
    """Astra Pro Plus 统一取流接口。"""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        self._pipeline = None
        self._align_filter = None
        self._uvc = None
        self._opened = False
        self._has_sdk_color = False
        self._align_mode = str(self.cfg.get_path("camera.align_mode", "SW")).upper()
        self._color_k: Optional[Intrinsics] = None
        self._depth_k: Optional[Intrinsics] = None
        self._extra: Optional[Extrinsics] = None
        self._device_info: dict = {}
        self._use_numpy_align = False

    def __enter__(self) -> "AstraCamera":
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
        try:
            from pyorbbecsdk import (
                AlignFilter,
                Config,
                OBAlignMode,
                OBSensorType,
                OBStreamType,
                Pipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "未安装 pyorbbecsdk。请使用项目内 conda 环境并执行 bash scripts/setup_env.sh"
            ) from e

        pipeline = Pipeline()
        device = pipeline.get_device()
        info = device.get_device_info()
        self._device_info = {
            "name": getattr(info, "get_name", lambda: "")(),
            "pid": int(info.get_pid()),
            "vid": int(info.get_vid()) if hasattr(info, "get_vid") else ORBBEC_VID,
            "serial": info.get_serial_number() if hasattr(info, "get_serial_number") else "",
            "firmware": info.get_firmware_version() if hasattr(info, "get_firmware_version") else "",
        }
        LOGGER.info("打开设备: %s", self._device_info)

        config = Config()
        depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_list is None:
            raise RuntimeError("未找到深度传感器，请检查 USB 权限与 udev 规则")
        depth_profile = _pick_video_profile(
            depth_list,
            int(self.cfg.get_path("camera.depth_width", 640)),
            int(self.cfg.get_path("camera.depth_height", 480)),
            int(self.cfg.get_path("camera.depth_fps", 30)),
        )
        config.enable_stream(depth_profile)

        self._has_sdk_color = False
        try:
            color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_list is not None:
                color_profile = _pick_video_profile(
                    color_list,
                    int(self.cfg.get_path("camera.color_width", 1280)),
                    int(self.cfg.get_path("camera.color_height", 720)),
                    int(self.cfg.get_path("camera.color_fps", 30)),
                )
                config.enable_stream(color_profile)
                self._has_sdk_color = True
        except Exception as e:
            LOGGER.warning("SDK 未枚举到彩色传感器: %s", e)

        # Astra Pro Plus (0x060f) 的 Pipeline SW_MODE 经常导致深度流起不来。
        # 深度与彩色分开取，再用出厂外参做 numpy D2C。
        pid = int(self._device_info.get("pid", 0))
        if pid == ASTRA_PRO_PLUS_DEPTH_PID:
            self._align_mode = "NONE"
            self._use_numpy_align = True
            LOGGER.info("Astra Pro Plus：关闭 SDK D2C，改用 numpy 软件配准")

        if self._has_sdk_color and self._align_mode == "HW":
            config.set_align_mode(OBAlignMode.HW_MODE)
        elif self._has_sdk_color and self._align_mode == "SW":
            config.set_align_mode(OBAlignMode.SW_MODE)
        else:
            config.set_align_mode(OBAlignMode.DISABLE)
            self._align_mode = "NONE"

        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            LOGGER.warning("enable_frame_sync 失败（可忽略）: %s", e)

        pipeline.start(config)
        self._pipeline = pipeline
        # 深度传感器上电后前几帧常为空，稍等再取
        time.sleep(0.4)

        # Astra Pro Plus：用 Pipeline 的 SW_MODE 做 D2C。
        # AlignFilter 是 Gemini 330 路径，在这台相机上会返回空帧。
        self._align_filter = None
        if self._has_sdk_color and self._align_mode not in ("SW", "HW"):
            try:
                self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            except Exception as e:
                LOGGER.warning("AlignFilter 不可用，将使用 numpy 软件 D2C: %s", e)
                self._use_numpy_align = True
        if not self._has_sdk_color:
            idx = find_orbbec_rgb_v4l_index()
            if idx is None:
                raise RuntimeError(
                    "SDK 无彩色流，且未找到 UVC 彩色相机 (VID=2bc5 PID=050f)。"
                    "请确认相机已插入，并安装 udev 规则。"
                )
            self._uvc = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            self._uvc.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get_path("camera.color_width", 1280)))
            self._uvc.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get_path("camera.color_height", 720)))
            self._uvc.set(cv2.CAP_PROP_FPS, int(self.cfg.get_path("camera.color_fps", 30)))
            if not self._uvc.isOpened():
                raise RuntimeError(f"无法打开 UVC 彩色相机 /dev/video{idx}")
            self._use_numpy_align = True
            LOGGER.info("彩色走 UVC 回退，深度走 PyOrbbecSDK")

        self._refresh_calibration()
        last_err = None
        warmup = int(self.cfg.get_path("camera.warmup_frames", 8))
        for i in range(max(3, warmup)):
            try:
                self.read()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.08)
        if last_err is not None:
            LOGGER.warning("预热未拿到完整帧，继续尝试: %s", last_err)
        self._opened = True

    def _refresh_calibration(self) -> None:
        param = self._pipeline.get_camera_param()
        self._depth_k = _intrinsic_from_ob(param.depth_intrinsic)
        self._color_k = _intrinsic_from_ob(param.rgb_intrinsic)
        try:
            self._extra = _extrinsic_from_ob(param.transform)
        except Exception as e:
            LOGGER.warning("读取 depth→color 外参失败: %s", e)
            self._extra = Extrinsics(np.eye(3), np.zeros(3))
        # UVC 实际分辨率可能与 SDK 内参不一致，用实际帧修正宽高
        if self._uvc is not None:
            w = int(self._uvc.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._uvc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                # 若 SDK 内参宽高与 UVC 不同，按比例缩放 fx/fy/cx/cy
                if self._color_k.width > 0 and self._color_k.height > 0:
                    sx = w / float(self._color_k.width)
                    sy = h / float(self._color_k.height)
                    if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3:
                        self._color_k = Intrinsics(
                            fx=self._color_k.fx * sx,
                            fy=self._color_k.fy * sy,
                            cx=self._color_k.cx * sx,
                            cy=self._color_k.cy * sy,
                            width=w,
                            height=h,
                            distortion=self._color_k.distortion,
                        )
                else:
                    self._color_k.width = w
                    self._color_k.height = h
        LOGGER.info("彩色内参 %s", self._color_k)
        LOGGER.info("深度内参 %s", self._depth_k)

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
            color_frame = frames.get_color_frame() if self._has_sdk_color else None
            if depth_frame is not None and (color_frame is not None or not self._has_sdk_color or self._uvc is not None):
                break
            time.sleep(0.02)
        if frames is None or depth_frame is None:
            raise TimeoutError("等待 RGB-D 帧超时（深度未到）")

        if self._has_sdk_color and self._align_filter is not None and not self._use_numpy_align:
            try:
                aligned_set = self._align_filter.process(frames)
                if aligned_set is not None:
                    frames = aligned_set
                    depth_frame = frames.get_depth_frame() or depth_frame
                    color_frame = frames.get_color_frame() or color_frame
            except Exception as e:
                LOGGER.warning("AlignFilter 失败，改用 numpy D2C: %s", e)
                self._use_numpy_align = True

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        scale = float(depth_frame.get_depth_scale())  # 通常 1.0，单位 mm
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw).copy()
        depth_mm = depth_raw.astype(np.float32) * scale
        depth_m = depth_mm / 1000.0

        color_bgr = None
        aligned = False
        if color_frame is not None:
            color_bgr = frame_to_bgr(color_frame)
        if color_bgr is None and self._uvc is not None:
            ok, color_bgr = self._uvc.read()
            if not ok:
                raise RuntimeError("UVC 彩色帧读取失败")

        if color_bgr is None:
            raise RuntimeError("未获得彩色图像")

        raw_valid = int((depth_raw > 0).sum())
        LOGGER.info(
            "深度 raw valid=%d/%d scale=%.4f mm_min=%.1f mm_max=%.1f",
            raw_valid,
            depth_raw.size,
            scale,
            float(depth_mm[depth_mm > 0].min()) if np.any(depth_mm > 0) else 0.0,
            float(depth_mm.max()) if depth_mm.size else 0.0,
        )
        min_z = float(self.cfg.get_path("camera.min_depth_m", 0.35))
        max_z = float(self.cfg.get_path("camera.max_depth_m", 1.5))
        depth_m = np.where((depth_m >= min_z) & (depth_m <= max_z), depth_m, 0.0).astype(np.float32)

        color_k = self._color_k
        depth_k = self._depth_k
        assert color_k is not None and depth_k is not None

        # SDK 软件/硬件对齐成功时，深度图尺寸应等于彩色图
        if (
            not self._use_numpy_align
            and self._has_sdk_color
            and depth_m.shape[0] == color_bgr.shape[0]
            and depth_m.shape[1] == color_bgr.shape[1]
        ):
            aligned_depth = depth_m
            aligned = True
            color_k = Intrinsics(
                fx=color_k.fx,
                fy=color_k.fy,
                cx=color_k.cx,
                cy=color_k.cy,
                width=color_bgr.shape[1],
                height=color_bgr.shape[0],
                distortion=color_k.distortion,
            )
        else:
            extra = self._extra or Extrinsics(np.eye(3), np.zeros(3))
            LOGGER.info("D2C 外参 R=\n%s t=%s", extra.rotation, extra.translation)
            # 用当前深度图真实宽高校准深度内参
            if depth_k.width != dw or depth_k.height != dh:
                sx = dw / max(depth_k.width, 1)
                sy = dh / max(depth_k.height, 1)
                depth_k = Intrinsics(
                    fx=depth_k.fx * sx,
                    fy=depth_k.fy * sy,
                    cx=depth_k.cx * sx,
                    cy=depth_k.cy * sy,
                    width=dw,
                    height=dh,
                    distortion=depth_k.distortion,
                )
            if color_k.width != color_bgr.shape[1] or color_k.height != color_bgr.shape[0]:
                sx = color_bgr.shape[1] / max(color_k.width, 1)
                sy = color_bgr.shape[0] / max(color_k.height, 1)
                color_k = Intrinsics(
                    fx=color_k.fx * sx,
                    fy=color_k.fy * sy,
                    cx=color_k.cx * sx,
                    cy=color_k.cy * sy,
                    width=color_bgr.shape[1],
                    height=color_bgr.shape[0],
                    distortion=color_k.distortion,
                )
            aligned_depth = align_depth_to_color(depth_m, depth_k, color_k, extra)
            aligned = True
            n_aligned = int((aligned_depth > 0).sum())
            LOGGER.info("numpy D2C 有效点 %d", n_aligned)
            if n_aligned < 1000:
                LOGGER.warning("D2C 有效点过少，回退为深度图缩放到彩色分辨率（几何用深度内参）")
                aligned_depth = cv2.resize(
                    depth_m, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                sx = color_bgr.shape[1] / max(dw, 1)
                sy = color_bgr.shape[0] / max(dh, 1)
                color_k = Intrinsics(
                    fx=depth_k.fx * sx,
                    fy=depth_k.fy * sy,
                    cx=depth_k.cx * sx,
                    cy=depth_k.cy * sy,
                    width=color_bgr.shape[1],
                    height=color_bgr.shape[0],
                    distortion=depth_k.distortion,
                )
                aligned = False

        depth_u16 = np.clip(aligned_depth * 1000.0, 0, 65535).astype(np.uint16)
        ts = 0
        try:
            ts = int(depth_frame.get_timestamp())
        except Exception:
            ts = time.time_ns()

        return RGBDFrame(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_m=aligned_depth.astype(np.float32),
            color_intrinsics=color_k,
            depth_intrinsics=depth_k,
            aligned=aligned,
            timestamp_ns=ts,
            depth_scale=scale,
            depth_to_color=self._extra,
            backend="pyorbbec+uvc" if self._uvc is not None else "pyorbbec",
            extra={"device": self._device_info, "align_mode": self._align_mode},
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
        if self._uvc is not None:
            self._uvc.release()
            self._uvc = None
        self._opened = False


class FileCamera:
    """离线回放：支持 GraspNet example_data 与本项目 dump 目录。"""

    def __init__(self, data_dir: str | Path, loop: bool = False):
        self.data_dir = Path(data_dir)
        self.loop = loop
        self._index = 0
        self._items = self._discover()
        if not self._items:
            raise FileNotFoundError(f"目录中没有可用的 RGB-D 数据: {self.data_dir}")

    def _discover(self) -> list[Path]:
        # 单帧 example_data
        if (self.data_dir / "color.png").exists() and (self.data_dir / "depth.png").exists():
            return [self.data_dir]
        # 多帧 dump：每子目录一帧，或 color_*.png
        sub = sorted([p for p in self.data_dir.iterdir() if p.is_dir() and (p / "color.png").exists()])
        if sub:
            return sub
        colors = sorted(self.data_dir.glob("color*.png"))
        return [c.parent for c in colors]

    def __enter__(self) -> "FileCamera":
        return self

    def __exit__(self, *args) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_intrinsics(self) -> Intrinsics:
        return self.read().color_intrinsics

    def read(self) -> RGBDFrame:
        if self._index >= len(self._items):
            if self.loop:
                self._index = 0
            else:
                raise StopIteration("离线数据已读完")
        item = self._items[self._index]
        self._index += 1
        return load_rgbd_dir(item)

    def frames(self) -> Iterator[RGBDFrame]:
        while True:
            try:
                yield self.read()
            except StopIteration:
                return


def load_rgbd_dir(data_dir: str | Path) -> RGBDFrame:
    """读取一个目录中的 color/depth/intrinsics。兼容 GraspNet example_data。"""
    data_dir = Path(data_dir)
    color_path = data_dir / "color.png"
    if not color_path.exists():
        color_path = data_dir / "rgb.png"
    depth_path = data_dir / "depth.png"
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"无法读取彩色图 {color_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图 {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.uint16)

    factor = 1000.0
    fx = fy = cx = cy = None
    json_path = data_dir / "intrinsics.json"
    mat_path = data_dir / "meta.mat"
    if json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        k = meta.get("color_intrinsics", meta)
        fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
        factor = float(meta.get("factor_depth", meta.get("depth_scale_inv", 1000.0)))
    elif mat_path.exists():
        import scipy.io as scio

        meta = scio.loadmat(str(mat_path))
        intrinsic = meta["intrinsic_matrix"]
        fx, fy = float(intrinsic[0][0]), float(intrinsic[1][1])
        cx, cy = float(intrinsic[0][2]), float(intrinsic[1][2])
        factor = float(np.array(meta["factor_depth"]).squeeze())
    else:
        # 缺省：按 640x480 典型 Astra 内参
        h, w = color_bgr.shape[:2]
        fx = fy = 0.9 * w
        cx, cy = w / 2.0, h / 2.0

    h, w = color_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    k = Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=w, height=h)
    depth_m = depth.astype(np.float32) / float(factor)
    return RGBDFrame(
        color_bgr=color_bgr,
        depth_u16=depth,
        depth_m=depth_m,
        color_intrinsics=k,
        depth_intrinsics=k,
        aligned=True,
        timestamp_ns=0,
        depth_scale=1000.0 / float(factor),
        backend="file",
        extra={"path": str(data_dir)},
    )


def save_rgbd_dir(frame: RGBDFrame, out_dir: str | Path) -> Path:
    """把一帧存成可回放目录。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "color.png"), frame.color_bgr)
    cv2.imwrite(str(out_dir / "depth.png"), frame.depth_u16)
    meta = {
        "color_intrinsics": frame.color_intrinsics.to_dict(),
        "depth_intrinsics": frame.depth_intrinsics.to_dict(),
        "factor_depth": 1000.0,
        "aligned": frame.aligned,
        "backend": frame.backend,
        "timestamp_ns": frame.timestamp_ns,
        "device": frame.extra.get("device", {}),
    }
    if frame.depth_to_color is not None:
        meta["depth_to_color"] = frame.depth_to_color.to_dict()
    (out_dir / "intrinsics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def open_camera(cfg: Optional[Config] = None, data_dir: Optional[str | Path] = None):
    """按 camera.backend 打开相机或离线目录。

    backend: auto | gemini336l | astra | pyorbbec | file
    auto 按 USB PID 选择（336L / Gemini 330 优先）。切相机只改这一项。
    """
    cfg = cfg or load_config()
    backend = str(cfg.get_path("camera.backend", "auto")).lower()
    if data_dir is not None or backend == "file":
        if data_dir is None:
            raise ValueError("file 后端需要 data_dir")
        return FileCamera(data_dir)
    if backend == "auto":
        from capture.usb_detect import detect_capture_backend

        backend = detect_capture_backend()
        LOGGER.info("camera.backend=auto，检测到 %s", backend)
    if backend in ("gemini336l", "gemini", "gemini330", "gemini336"):
        from capture.gemini336l import Gemini336LCamera

        return Gemini336LCamera(cfg)
    if backend in ("astra", "pyorbbec", "astra_pro_plus"):
        return AstraCamera(cfg)
    raise ValueError(f"未知 camera.backend={backend}，可选 auto|gemini336l|astra|file")
