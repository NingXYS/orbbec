"""离屏渲染：RGB、着色点云、叠加抓取框的点云。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def save_rgb(color_bgr: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color_bgr)
    return path


def _lookat_front(pcd):
    """相机从斜上方看向点云，按主体包围盒取景（去掉最远离群点）。"""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None
    z = pts[:, 2]
    lo, hi = np.percentile(z, [2, 96])
    core = pts[(z >= lo) & (z <= hi)]
    if core.shape[0] < 10:
        core = pts
    center = core.mean(axis=0)
    extent = float(np.max(core.max(axis=0) - core.min(axis=0)))
    dist = max(extent * 0.72, 0.28)
    # 相机系：X 右、Y 下、Z 前。从 -Y（画面上方）略抬、沿 -Z 看向桌面。
    eye = center + np.array([0.0, -0.40 * dist, -0.90 * dist])
    return center, eye


def render_geometries(geoms: Sequence, path: str | Path, width: int = 1280, height: int = 720) -> Path:
    """优先 OffscreenRenderer，失败则 Visualizer 截屏。"""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Filament 离屏
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])
        for i, g in enumerate(geoms):
            name = f"g{i}"
            g_mat = mat
            if not isinstance(g, o3d.geometry.PointCloud):
                g_mat = o3d.visualization.rendering.MaterialRecord()
                g_mat.shader = "defaultLit"
            scene.add_geometry(name, g, g_mat)
        # 取点云定相机
        pcd = next((g for g in geoms if isinstance(g, o3d.geometry.PointCloud)), None)
        if pcd is not None and np.asarray(pcd.points).size:
            center, eye = _lookat_front(pcd)
            renderer.setup_camera(42.0, center, eye, [0, -1, 0])
        img = renderer.render_to_image()
        o3d.io.write_image(str(path), img)
        return path
    except Exception:
        pass

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()
    return path


def render_pointcloud(pcd, path: str | Path, width: int = 1280, height: int = 720) -> Path:
    return render_geometries([pcd], path, width, height)


def _project_pts(K, pts: np.ndarray):
    """相机系 3D 点 → 像素。z<=0 的点返回 None。"""
    out = []
    for p in pts:
        if p[2] <= 1e-6:
            out.append(None)
            continue
        u = int(K.fx * p[0] / p[2] + K.cx)
        v = int(K.fy * p[1] / p[2] + K.cy)
        out.append((u, v))
    return out


def _score_bgr(score: float, best: bool) -> tuple[int, int, int]:
    """与 graspnetAPI plot_gripper_pro_max 一致：高分偏红，低分偏蓝。"""
    s = float(np.clip(score, 0.0, 1.0))
    if best:
        return (0, 0, 255)  # BGR 纯红，突出最优
    return (int(255 * (1.0 - s)), 0, int(255 * s))


def overlay_grasps_on_rgb(
    color_bgr: np.ndarray,
    gg,
    K,
    top_k: int = 20,
    highlight_index: Optional[int] = None,
) -> np.ndarray:
    """把 6-DoF 夹爪投影成 RGB 上的抓取框（宽边 + 两指 + 接近方向）。

    highlight_index 为点名匹配到的候选：加粗黄框画在最上层。
    """
    vis = color_bgr.copy()
    if gg is None or len(gg) == 0:
        return vis
    n_all = len(gg)
    n = min(int(top_k), n_all)
    T = np.asarray(gg.translations)
    R = np.asarray(gg.rotation_matrices)
    W = np.asarray(gg.widths)
    D = np.asarray(gg.depths) if hasattr(gg, "depths") else np.full(n_all, 0.02)
    S = np.asarray(gg.scores)
    order = list(range(n))
    hi = int(highlight_index) if highlight_index is not None else None
    if hi is not None and 0 <= hi < n_all and hi not in order:
        order.append(hi)
    h, w = vis.shape[:2]
    for i in order:
        t, r, width, depth, score = T[i], R[i], float(W[i]), float(D[i]), float(S[i])
        x_axis, z_axis = r[:, 0], r[:, 2]
        hw, finger = width / 2.0, max(depth, 0.02)
        pts3 = np.stack(
            [
                t + x_axis * hw,
                t - x_axis * hw,
                t + x_axis * hw + z_axis * finger,
                t - x_axis * hw + z_axis * finger,
                t - z_axis * 0.04,
                t,
            ],
            axis=0,
        )
        uv = _project_pts(K, pts3)
        if any(p is None for p in uv[:4]):
            continue
        is_hi = hi is not None and i == hi
        if is_hi:
            color, thick = (0, 255, 255), 3
        else:
            color = _score_bgr(score, best=(hi is None and i == 0))
            thick = 3 if (hi is None and i == 0) else 1
        pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for a, b in pairs:
            pa, pb = uv[a], uv[b]
            if pa is None or pb is None:
                continue
            if not (0 <= pa[0] < w and 0 <= pa[1] < h) and not (0 <= pb[0] < w and 0 <= pb[1] < h):
                continue
            cv2.line(vis, pa, pb, color, thick, cv2.LINE_AA)
        if uv[4] is not None and uv[5] is not None:
            cv2.arrowedLine(vis, uv[4], uv[5], color, thick, cv2.LINE_AA, tipLength=0.25)
        if uv[5] is not None and 0 <= uv[5][0] < w and 0 <= uv[5][1] < h:
            cv2.circle(vis, uv[5], 6 if is_hi else 4, color, -1)
            if is_hi or i < 5:
                tag = f"HIT {score:.2f}" if is_hi else f"{i}:{score:.2f}"
                cv2.putText(
                    vis,
                    tag,
                    (uv[5][0] + 6, uv[5][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 if is_hi else 0.45,
                    color,
                    2 if is_hi else 1,
                    cv2.LINE_AA,
                )
    return vis


def save_triplet(frame, pcd, gg, out_dir: str | Path, top_k: int = 50) -> dict:
    """验收要求的三张图：原始 RGB、着色点云、叠加抓取框点云。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_rgb = save_rgb(frame.color_bgr, out_dir / "01_rgb.png")
    p_cloud = render_pointcloud(pcd, out_dir / "02_cloud.png")
    geoms = [pcd]
    if gg is not None and len(gg) > 0:
        show = gg[: min(int(top_k), len(gg))]
        geoms.extend(show.to_open3d_geometry_list())
    p_grasp = render_geometries(geoms, out_dir / "03_cloud_grasps.png")
    overlay = overlay_grasps_on_rgb(frame.color_bgr, gg, frame.color_intrinsics, top_k=min(20, top_k))
    p_ov = save_rgb(overlay, out_dir / "04_rgb_grasp_centers.png")
    return {"rgb": p_rgb, "cloud": p_cloud, "grasps": p_grasp, "overlay": p_ov}
