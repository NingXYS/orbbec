"""bbox → 3D 质心（离线，不接实时流）。

取对齐深度图中 bbox 内的有效点，用中位数深度附近的点估计质心，
避免单像素落在空洞或背景上。有效点不足时抛出明确错误。
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from capture.types import Intrinsics, RGBDFrame
from common.config import Config, load_config

LOGGER = logging.getLogger("graspnet_orbbec.locator")

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BBox = Sequence[float]


class InsufficientDepthError(ValueError):
    """bbox 内有效深度点太少，无法估计质心。"""


@dataclass
class LocateResult:
    """相机坐标系下的目标质心（单位米）。"""

    xyz_m: list[float]  # [x, y, z]，相机系
    uv: list[int]  # 质心投影到 RGB 的像素 [u, v]
    bbox_xyxy: list[int]  # 裁剪后的输入框
    roi_xyxy: list[int]  # 收缩后实际用的框
    n_roi: int
    n_valid: int
    n_inlier: int
    valid_ratio: float
    z_median_m: float
    depth_band_m: float
    elapsed_ms: float
    target: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["xyz_m"] = [float(v) for v in self.xyz_m]
        return d


def clip_bbox(bbox: BBox, width: int, height: int) -> list[int]:
    """把框裁进图像，返回包含两端像素的 [x1,y1,x2,y2]。"""
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox 必须是 [x1,y1,x2,y2]")
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox 裁剪后为空: {[x1, y1, x2, y2]} 图像 {width}x{height}")
    return [x1, y1, x2, y2]


def _shrink_bbox(xyxy: list[int], ratio: float) -> list[int]:
    """向中心收缩，减少框边带到的背景；过小则不收缩。"""
    if ratio <= 0:
        return list(xyxy)
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return list(xyxy)
    dx = int(round(bw * ratio / 2.0))
    dy = int(round(bh * ratio / 2.0))
    sx1, sy1, sx2, sy2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    if sx2 - sx1 < 4 or sy2 - sy1 < 4:
        return list(xyxy)
    return [sx1, sy1, sx2, sy2]


def _locator_params(cfg: Optional[Config]) -> dict:
    cfg = cfg or load_config()
    return {
        "min_depth_m": float(cfg.get_path("locator.min_depth_m", cfg.get_path("camera.min_depth_m", 0.12))),
        "max_depth_m": float(cfg.get_path("locator.max_depth_m", cfg.get_path("camera.max_depth_m", 3.50))),
        "min_valid_points": int(cfg.get_path("locator.min_valid_points", 50)),
        "bbox_shrink": float(cfg.get_path("locator.bbox_shrink", 0.08)),
        "depth_band_m": float(cfg.get_path("locator.depth_band_m", 0.06)),
    }


def locate_from_depth(
    depth_m: np.ndarray,
    K: Intrinsics,
    bbox: BBox,
    *,
    min_depth_m: float = 0.12,
    max_depth_m: float = 3.50,
    min_valid_points: int = 50,
    bbox_shrink: float = 0.08,
    depth_band_m: float = 0.06,
    target: str = "",
) -> LocateResult:
    """用 bbox 内有效深度的中位数邻域估计 3D 质心。全程向量化。"""
    t0 = time.perf_counter()
    if depth_m.ndim != 2:
        raise ValueError(f"depth 必须是 HxW，实际 {depth_m.shape}")
    h, w = depth_m.shape
    if K.width != w or K.height != h:
        raise ValueError(f"内参 {K.width}x{K.height} 与深度 {w}x{h} 不一致，对齐帧才能定位")
    xyxy = clip_bbox(bbox, w, h)
    roi_xyxy = _shrink_bbox(xyxy, bbox_shrink)
    x1, y1, x2, y2 = roi_xyxy
    # 框是闭区间，切片右端 +1
    roi = np.asarray(depth_m[y1 : y2 + 1, x1 : x2 + 1], dtype=np.float32)
    valid = np.isfinite(roi) & (roi > min_depth_m) & (roi < max_depth_m)
    n_roi = int(roi.size)
    n_valid = int(valid.sum())
    if n_valid < min_valid_points:
        pct = 100.0 * n_valid / max(n_roi, 1)
        raise InsufficientDepthError(
            f"bbox {xyxy} 目标={target or '-'} 内有效深度点 {n_valid} < 阈值 {min_valid_points}"
            f"（ROI {x2 - x1 + 1}x{y2 - y1 + 1}，有效率 {pct:.1f}%），无法估计 3D 质心"
        )
    z = roi[valid]
    z_med = float(np.median(z))
    band = float(max(1e-4, depth_band_m))
    # 只保留中位数深度附近的点，挡住框里的前景遮挡和远背景
    inlier = valid & (np.abs(roi - np.float32(z_med)) <= np.float32(band))
    n_inlier = int(inlier.sum())
    if n_inlier < min_valid_points:
        inlier = valid
        n_inlier = n_valid
        LOGGER.warning(
            "中位数带宽 ±%.3fm 内仅 %d 点，回退为全部有效点 target=%s",
            band,
            int((valid & (np.abs(roi - np.float32(z_med)) <= np.float32(band))).sum()),
            target or "-",
        )
    ys, xs = np.nonzero(inlier)
    zz = roi[ys, xs].astype(np.float32)
    uu = (xs + x1).astype(np.float32)
    vv = (ys + y1).astype(np.float32)
    xx = (uu - np.float32(K.cx)) * zz / np.float32(K.fx)
    yy = (vv - np.float32(K.cy)) * zz / np.float32(K.fy)
    # 各轴中位数：抗单点噪声，且不是 bbox 中心那一个像素
    xyz = np.array([np.median(xx), np.median(yy), np.median(zz)], dtype=np.float64)
    if not np.isfinite(xyz).all() or xyz[2] <= 1e-6:
        raise InsufficientDepthError(f"质心深度非法 z={xyz[2]} target={target or '-'}")
    u_proj = int(round(float(K.fx) * float(xyz[0]) / float(xyz[2]) + float(K.cx)))
    v_proj = int(round(float(K.fy) * float(xyz[1]) / float(xyz[2]) + float(K.cy)))
    u_proj = max(0, min(w - 1, u_proj))
    v_proj = max(0, min(h - 1, v_proj))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LocateResult(
        xyz_m=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
        uv=[u_proj, v_proj],
        bbox_xyxy=xyxy,
        roi_xyxy=roi_xyxy,
        n_roi=n_roi,
        n_valid=n_valid,
        n_inlier=n_inlier,
        valid_ratio=float(n_valid) / float(max(n_roi, 1)),
        z_median_m=z_med,
        depth_band_m=band,
        elapsed_ms=float(elapsed_ms),
        target=target,
        extra={
            "uv_median": [int(round(float(np.median(uu)))), int(round(float(np.median(vv))))],
            "z_mad_m": float(np.median(np.abs(z - z_med))),
        },
    )


def locate_target(
    frame: RGBDFrame,
    bbox: BBox,
    cfg: Optional[Config] = None,
    target: str = "",
) -> LocateResult:
    """对一帧对齐 RGB-D 做 bbox→质心。"""
    if frame.color_bgr.shape[:2] != frame.depth_m.shape:
        raise ValueError(
            f"RGB {frame.color_bgr.shape[:2][::-1]} 与 depth {frame.depth_m.shape[::-1]} 尺寸不同，需要对齐帧"
        )
    params = _locator_params(cfg)
    return locate_from_depth(frame.depth_m, frame.color_intrinsics, bbox, target=target, **params)


def put_cjk(bgr: np.ndarray, text: str, org, color=(0, 255, 0), size: int = 20) -> None:
    x, y = org
    try:
        from PIL import Image, ImageDraw, ImageFont

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(_CJK_FONT, size)
        draw.text((x, max(0, y - size)), text, font=font, fill=(color[2], color[1], color[0]))
        bgr[:, :, :] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            bgr,
            text.encode("ascii", "replace").decode("ascii"),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_locate_overlay(
    color_bgr: np.ndarray,
    result: LocateResult,
    *,
    depth_m: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """叠加 bbox、收缩 ROI、质心十字，以及 ROI 内空洞提示。"""
    vis = color_bgr.copy()
    x1, y1, x2, y2 = result.bbox_xyxy
    rx1, ry1, rx2, ry2 = result.roi_xyxy
    if depth_m is not None:
        roi = depth_m[ry1 : ry2 + 1, rx1 : rx2 + 1]
        hole = (~np.isfinite(roi)) | (roi <= 0)
        tint = vis[ry1 : ry2 + 1, rx1 : rx2 + 1]
        red = np.zeros_like(tint)
        red[:, :, 2] = 255
        tint[hole] = (0.70 * tint[hole] + 0.30 * red[hole]).astype(np.uint8)
        vis[ry1 : ry2 + 1, rx1 : rx2 + 1] = tint
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
    u, v = result.uv
    cv2.drawMarker(vis, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
    cv2.circle(vis, (u, v), 10, (0, 0, 255), 2)
    cv2.circle(vis, (u, v), 3, (0, 255, 255), -1)
    name = label or result.target or "target"
    x, y, z = result.xyz_m
    put_cjk(vis, f"{name}", (x1, max(22, y1 - 6)), (0, 255, 0), 20)
    put_cjk(
        vis,
        f"xyz=({x:.3f},{y:.3f},{z:.3f})m  uv=({u},{v})  valid={result.n_valid}/{result.n_roi}  {result.elapsed_ms:.1f}ms",
        (8, 28),
        (0, 255, 255),
        18,
    )
    return vis


def draw_depth_error(color_bgr: np.ndarray, bbox: BBox, message: str) -> np.ndarray:
    """空洞失败时的可视化：红框 + 报错原文。"""
    vis = color_bgr.copy()
    h, w = vis.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    put_cjk(vis, message, (8, 28), (0, 0, 255), 18)
    return vis


def centroid_in_bbox(result: LocateResult, margin: int = 0) -> bool:
    x1, y1, x2, y2 = result.bbox_xyxy
    u, v = result.uv
    return (x1 - margin) <= u <= (x2 + margin) and (y1 - margin) <= v <= (y2 + margin)
