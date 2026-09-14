"""RGB / 深度边缘叠加，用来目视与量化检验 D2C 对齐。不改 viz 模块。

度量只比较画面内部的几何边缘：硬件 D2C 后深度有效区四周一圈没有深度，
若把这条视场边框当成深度边缘，平面场景会得到上百像素的假偏差。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# 互相对应搜索半径；超出则视为 RGB 纹理边（深度上没有几何边）
_MUTUAL_PX = 15.0


def colorize_depth(depth_m: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    vis = np.clip(depth_m / max(max_m, 1e-6), 0, 1)
    vis = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def overlay_rgbd(color_bgr: np.ndarray, depth_m: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    depth_c = colorize_depth(depth_m)
    if depth_c.shape[:2] != color_bgr.shape[:2]:
        depth_c = cv2.resize(depth_c, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (depth_m > 0)[:, :, None]
    blend = cv2.addWeighted(color_bgr, 1.0 - alpha, depth_c, alpha, 0)
    return np.where(mask, blend, color_bgr).astype(np.uint8)


def _interior_mask(valid: np.ndarray, border: int = 21) -> np.ndarray:
    kernel = np.ones((border, border), np.uint8)
    return cv2.erode(valid.astype(np.uint8), kernel, iterations=1) > 0


def _depth_edges(depth_m: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """去倾角后的局部残差 Canny：保留物体/缝的几何边，去掉平面斜坡和视场边框。"""
    valid = depth_m > 0
    fill = depth_m.copy()
    if np.any(valid):
        fill[~valid] = float(np.median(depth_m[valid]))
    residual = np.abs(fill - cv2.GaussianBlur(fill, (0, 0), 5))
    residual[~interior] = 0
    r8 = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (cv2.Canny(r8, 30, 90) > 0) & interior


def measure_edge_alignment_px(color_bgr: np.ndarray, depth_m: np.ndarray) -> Optional[float]:
    """RGB 几何边到深度边的中位距离（像素）。只统计 15 px 内能对上的边。"""
    if color_bgr.shape[:2] != depth_m.shape[:2]:
        return None
    valid = depth_m > 0
    if int(valid.sum()) < 1000:
        return None
    interior = _interior_mask(valid)
    if int(interior.sum()) < 1000:
        return None
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    e_rgb = (cv2.Canny(gray, 40, 120) > 0) & interior
    e_d = _depth_edges(depth_m, interior)
    if int(e_rgb.sum()) < 200 or int(e_d.sum()) < 50:
        return None
    dist = cv2.distanceTransform((~e_d).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    sample = dist[e_rgb]
    near = sample[sample <= _MUTUAL_PX]
    if near.size < 150:
        return None
    return float(np.median(near))


def edge_overlay(color_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """红=深度边缘，绿=RGB 边缘，黄=重合。返回 (叠加图, 中位偏差 px)。"""
    h, w = color_bgr.shape[:2]
    if depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = depth_m > 0
    interior = _interior_mask(valid)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    e_rgb = (cv2.Canny(gray, 40, 120) > 0) & interior
    e_d = _depth_edges(depth_m, interior)
    overlay = color_bgr.copy()
    overlay[e_d] = (0, 0, 255)
    overlay[e_rgb] = (0, 255, 0)
    overlay[e_d & e_rgb] = (0, 255, 255)
    err = measure_edge_alignment_px(color_bgr, depth_m)
    return overlay, err


def hole_overlay(color_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, dict]:
    """无效深度涂品红，返回 (叠加图, 统计)。用于新旧相机空洞对比。"""
    h, w = color_bgr.shape[:2]
    if depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = depth_m > 0
    overlay = color_bgr.copy()
    overlay[~valid] = (0.55 * overlay[~valid] + np.array([180, 0, 180])).astype(np.uint8)
    n = int(valid.size)
    n_ok = int(valid.sum())
    stats = {
        "valid": n_ok,
        "total": n,
        "valid_ratio": n_ok / max(n, 1),
        "hole_ratio": 1.0 - n_ok / max(n, 1),
    }
    label = f"holes {stats['hole_ratio']*100:.1f}%  valid {stats['valid_ratio']*100:.1f}%"
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return overlay, stats


def make_align_panel(color_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    """左 RGB、中伪彩深度、右边缘叠加。"""
    overlay, err = edge_overlay(color_bgr, depth_m)
    finite = depth_m[depth_m > 0]
    max_m = float(np.percentile(finite, 95)) if finite.size else 2.0
    depth_c = colorize_depth(depth_m, max_m=max(max_m, 0.3))
    if depth_c.shape[:2] != color_bgr.shape[:2]:
        depth_c = cv2.resize(depth_c, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    panel = np.hstack([color_bgr, depth_c, overlay])
    label = f"edge median={err:.2f}px" if err is not None else "edge n/a"
    cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return panel, err
