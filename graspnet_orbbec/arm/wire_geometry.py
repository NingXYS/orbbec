"""原始RGB射线、支撑平面与法兰工具变换；此模块不访问机械臂。"""

import cv2
import numpy as np


def fk(mdh, joints, *, frames=False):
    """按已核实的SDK名义MDH计算几何；结果不代表实测位姿。"""
    table, joints = np.asarray(mdh, float), np.asarray(joints, float)
    if table.shape != (6, 4) or joints.shape != (6,) or not np.isfinite(table).all() or not np.isfinite(joints).all():
        raise ValueError('需要SDK六轴MDH和有限关节弧度')
    total, chain = np.eye(4), []
    for (d, a, alpha, offset), angle in zip(table, joints):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(angle+offset), np.sin(angle+offset)
        link = np.array([[ct, -st, 0, a], [ca*st, ca*ct, -sa, -sa*d],
                         [sa*st, sa*ct, ca, ca*d], [0, 0, 0, 1]])
        total = total @ link
        chain.append(total.copy())
    return chain if frames else total


def rigid_transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-6):
        raise ValueError('需要米制、右手系4×4刚体变换')
    return t.copy()


def flange_transform(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError('法兰反馈须为[x,y,z,roll,pitch,yaw]，单位m/rad')
    x, y, z = p[3:]
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rz @ ry @ rx, p[:3]
    return t


def board_plane(image, k, d, board, max_reprojection_px=1.0):
    """法向统一朝相机，避免棋盘角点顺序翻转改变线材高度的正方向。"""
    size = tuple(board['inner_corners'])
    points = np.zeros((size[0] * size[1], 3), np.float32)
    points[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2) * board['square_size_m']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, size)
    detector = 'classic_subpixel'
    if found:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    else:
        # 项目隔离环境OpenCV 4.10对倾斜棋盘可能漏检；SB仍须完整40角点和相同RMS门槛。
        found, corners = cv2.findChessboardCornersSB(gray, size,
                                                     cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        detector = 'sector_based'
    if not found:
        raise ValueError('当前原图未完整看到棋盘；平面模式需要棋盘与线材同时可见')
    ok, rvec, tvec = cv2.solvePnP(points, corners, k, d)
    if not ok:
        raise ValueError('棋盘PnP失败')
    rotation = cv2.Rodrigues(rvec)[0]
    xyz = points @ rotation.T + tvec.reshape(3)
    if np.any(xyz[:, 2] <= 0):
        raise ValueError('棋盘PnP落在相机后方')
    projected = cv2.projectPoints(points, rvec, tvec, k, d)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected-corners)**2, axis=2))))
    if not np.isfinite(rms) or rms > max_reprojection_px:
        raise ValueError(f'棋盘重投影RMS过大：{rms:.3f}px')
    normal = rotation[:, 2]
    origin = tvec.reshape(3)
    if normal @ origin > 0:
        normal = -normal
    return {'normal_camera': normal.tolist(), 'origin_camera_m': origin.tolist(),
            'reprojection_rms_px': rms, 'corner_count': len(corners), 'detector': detector,
            'source': 'current_raw_rgb_chessboard_pnp', 'depth_used': False}


def pixels_on_plane(pixels, k, d, plane, height_m=0.0):
    uv = np.asarray(pixels, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
        raise ValueError('路径必须为有限Nx2原图像素')
    height_m = float(height_m)
    if not np.isfinite(height_m):
        raise ValueError('相对棋盘平面的有符号高度必须为有限米数')
    rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), k, d).reshape(-1, 2)
    rays = np.c_[rays, np.ones(len(rays))]
    n = np.asarray(plane['normal_camera'], dtype=float)
    origin = np.asarray(plane['origin_camera_m'], dtype=float)
    if (n.shape != (3,) or origin.shape != (3,) or not np.isfinite(n).all()
            or not np.isfinite(origin).all() or abs(np.linalg.norm(n)-1) > 1e-6):
        raise ValueError('支撑平面需要单位法向与有限原点')
    denominator = rays @ n
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError('射线与平面近似平行')
    scale = (origin @ n + height_m) / denominator
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('射线平面交点位于相机后方或无效')
    return rays * scale[:, None]


def grasp_transforms(plan, t_base_rgb, t_flange_grasp=None, clearance_m=0.03):
    """工具坐标定义：X为闭合轴、Z为朝支撑面的接近轴，Y补成右手系。"""
    base_rgb = rigid_transform(t_base_rgb)
    center = np.asarray(plan['grasp_camera_m'], float)
    closing = np.asarray(plan['closing_direction_camera'], float)
    outward = np.asarray(plan['support_plane_normal'], float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (center, closing, outward)):
        raise ValueError('缺少完整的三维夹持中心和方向')
    x = closing / np.linalg.norm(closing)
    z = -outward / np.linalg.norm(outward)
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('闭合方向与接近轴退化')
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    rgb_grasp = np.eye(4)
    rgb_grasp[:3, :3] = np.column_stack([x, y, z])
    rgb_grasp[:3, 3] = center
    base_grasp = rigid_transform(base_rgb @ rgb_grasp)
    result = {'T_base_grasp': base_grasp.tolist(),
              'grasp_base_m': base_grasp[:3, 3].tolist(),
              'closing_direction_base': (base_rgb[:3, :3] @ closing).tolist(),
              'approach_direction_base': base_grasp[:3, 2].tolist(),
              'tcp_compensation_applied': False}
    if t_flange_grasp is not None:
        if not np.isfinite(clearance_m) or clearance_m <= 0:
            raise ValueError('预抓取净空必须为正数，单位米')
        tool = rigid_transform(t_flange_grasp)
        pregrasp = base_grasp.copy()
        pregrasp[:3, 3] -= pregrasp[:3, 2] * clearance_m
        result.update(T_base_flange_grasp=(base_grasp @ np.linalg.inv(tool)).tolist(),
                      T_base_flange_pregrasp=(pregrasp @ np.linalg.inv(tool)).tolist(),
                      tcp_compensation_applied=True)
    return result


def fit_pivot(transforms):
    """同一刚性测点、多朝向最小二乘；只能求位置，不能据此推断工具轴向。"""
    ts = [rigid_transform(t) for t in transforms]
    if len(ts) < 6:
        raise ValueError('至少6个不同朝向的同点接触姿态')
    a = np.vstack([np.c_[t[:3, :3], -np.eye(3)] for t in ts])
    b = np.concatenate([-t[:3, 3] for t in ts])
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank != 6 or singular[-1] / singular[0] < 0.03:
        raise ValueError('朝向激励不足，TCP求解退化；需要绕不同轴改变姿态')
    residuals = np.linalg.norm((a @ solution-b).reshape(-1, 3), axis=1)
    return {'tcp_translation_flange_m': solution[:3].tolist(),
            'fixed_point_base_m': solution[3:].tolist(),
            'rms_m': float(np.sqrt(np.mean(residuals**2))),
            'max_m': float(residuals.max()), 'sample_count': len(ts),
            'orientation_calibrated': False, 'verified': False,
            'scope': 'measured_rigid_point_only_not_automatically_grasp_center'}
