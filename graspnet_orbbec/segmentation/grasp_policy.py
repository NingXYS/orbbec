"""端子/绝缘交界后的夹持中心策略；位置与工具TCP补偿分别处理。"""

import copy

import numpy as np


def plan_candidate(candidate, boundary_xy, *, offset_mm=10, path_xyz_m=None, plane_normal=None):
    """沿已定向中心线按米制弧长后退；三维路径须与inward_path_xy逐点对应。"""
    offset_mm = float(offset_mm)
    if not np.isfinite(offset_mm) or offset_mm <= 0:
        raise ValueError('夹持中心后退距离必须为正数，单位mm')
    plan = {
        'candidate_id': candidate['id'], 'anchor': 'terminal_insulation_boundary',
        'offset_mm': offset_mm, 'closing_rule': 'perpendicular_to_local_wire_tangent',
        'boundary_confirmed': False, 'boundary_xy': None, 'grasp_xy_preview': None,
        'grasp_camera_m': None, 'closing_direction_camera': None,
        'robot_pose_valid': False, 'tcp_compensation_applied': False,
    }
    if boundary_xy is None:
        return dict(plan, status='boundary_required')
    path = np.asarray(candidate.get('inward_path_xy', []), dtype=float)
    point = np.asarray(boundary_xy, dtype=float)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2 or not np.isfinite(path).all():
        return dict(plan, status='path_unavailable')
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('交界位置必须是有限原图像素坐标')
    distance = np.linalg.norm(path - point, axis=1)
    index = int(distance.argmin())
    if distance[index] > 12:
        return dict(plan, status='boundary_outside_path')
    plan.update(boundary_confirmed=True, boundary_xy=path[index].tolist(),
                boundary_input_xy=point.tolist(), boundary_path_index=index)
    if candidate.get('path_ambiguous', False):
        return dict(plan, status='wire_instance_ambiguous')
    if path_xyz_m is None:
        return dict(plan, status='metric_path_required')
    xyz = np.asarray(path_xyz_m, dtype=float)
    if xyz.shape != (len(path), 3) or not np.isfinite(xyz).all() or np.any(xyz[:,2] <= 0):
        raise ValueError('米制相机路径必须为与像素路径对应的Nx3有限坐标，且Z为正')
    tail = xyz[index:]
    if len(tail) < 2:
        return dict(plan, status='insufficient_wire_length')
    lengths = np.linalg.norm(np.diff(tail, axis=0), axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    required = offset_mm / 1000
    if cumulative[-1] + 1e-12 < required:
        return dict(plan, status='insufficient_wire_length')
    j = min(int(np.searchsorted(cumulative, required, side='left')), len(tail)-1)
    if j == 0 or lengths[j-1] <= 0:
        return dict(plan, status='degenerate_metric_path')
    fraction = float((required - cumulative[j-1]) / lengths[j-1])
    center = tail[j-1] + fraction * (tail[j] - tail[j-1])
    uv = path[index+j-1] + fraction * (path[index+j] - path[index+j-1])
    tangent = tail[j] - tail[j-1]
    tangent /= np.linalg.norm(tangent)
    plan.update(grasp_camera_m=center.tolist(), grasp_xy_preview=uv.tolist(),
                wire_tangent_camera=tangent.tolist(), status='support_plane_required')
    if plane_normal is not None:
        normal = np.array(plane_normal, dtype=float, copy=True)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            raise ValueError('支撑面法向必须是有限非零三维向量')
        normal /= np.linalg.norm(normal)
        closing = np.cross(normal, tangent)
        if np.linalg.norm(closing) < 1e-6:
            return dict(plan, status='wire_direction_inconsistent_with_plane')
        closing /= np.linalg.norm(closing)
        plan.update(closing_direction_camera=closing.tolist(), support_plane_normal=normal.tolist(),
                    status='camera_frame_preview_ready')
    return plan


def apply_grasp_policy(result, cfg, boundaries=None, metric_paths=None):
    """没有交界/真实三维路径时只记录待定位状态，不以像素冒充毫米。"""
    offset = float(cfg.get_path('wire.grasp_policy.offset_mm', 10))
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError('夹持中心后退距离必须为正数，单位mm')
    boundaries, metric_paths = boundaries or {}, metric_paths or {}
    unknown = set(boundaries) - {c['id'] for c in result.candidates}
    if unknown:
        raise ValueError(f'交界点对应的候选不存在：{sorted(unknown)}')
    plans = []
    for candidate in result.candidates:
        data = metric_paths.get(candidate['id'], {})
        plans.append(plan_candidate(candidate, boundaries.get(candidate['id']), offset_mm=offset,
                                    path_xyz_m=data.get('xyz_m'), plane_normal=data.get('plane_normal')))
    result.grasp_policy = {
        'anchor': 'terminal_insulation_boundary', 'offset_mm': offset,
        'closing_rule': 'perpendicular_to_local_wire_tangent',
        'boundary_confirmation': 'manual_per_snapshot',
        'metric_source': 'aligned_camera_path_m',
        'robot_pose_valid': False, 'tcp_compensation_applied': False,
    }
    result.grasp_plans = plans
    return result


def confirm_boundary(result, point, cfg):
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('交界点必须为有限像素坐标')
    distances = []
    for candidate in result.candidates:
        path = np.asarray(candidate.get('inward_path_xy', []), dtype=float)
        if path.ndim == 2 and len(path):
            lateral = np.linalg.norm(path-point, axis=1)
            index = int(lateral.argmin())
            arc = float(np.linalg.norm(np.diff(path[:index+1], axis=0), axis=1).sum())
            distances.append((float(lateral[index]), arc, candidate['id']))
    distances.sort()
    if not distances or distances[0][0] > 12:
        raise ValueError('交界点离端部中心线太远，请点选端子侧的金属/绝缘交界')
    # 同根线两端共享中心线，按沿线距端点的弧长归属，不能只比较横向距离。
    close = sorted((row for row in distances if row[0] <= distances[0][0]+2), key=lambda row: row[1])
    if len(close) > 1 and close[1][1] - close[0][1] < 4:
        raise ValueError('该位置接近多个端部路径，请先分开线材或缩小分割区域')
    chosen = close[0][2]
    boundaries = {p['candidate_id']: p['boundary_input_xy'] for p in result.grasp_plans if p.get('boundary_confirmed')}
    boundaries[chosen] = point.tolist()
    # 先完成校验再替换结果，点击错误时保留上次确认。
    updated = copy.copy(result)
    apply_grasp_policy(updated, cfg, boundaries)
    result.grasp_policy, result.grasp_plans = updated.grasp_policy, updated.grasp_plans
    return chosen


def confirm_body_boundary(result, point, cfg):
    """人工确认线身掩码与金属的交界；不把自动排除的断口冒充物理线头。"""
    from segmentation.wire import _end_region
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('交界点必须为有限像素坐标')
    updated = copy.copy(result)
    updated.candidates = copy.deepcopy(result.candidates)
    updated.end_regions = list(result.end_regions)
    for rejected in result.automation.get('rejected_geometric_ends', []):
        if (rejected.get('rejection_reason') != 'image_line_continues_past_mask_tip'
                or rejected.get('path_ambiguous')
                or np.linalg.norm(np.asarray(rejected['tip_xy'])-point) > 12):
            continue
        if any(c.get('tip_xy') == rejected['tip_xy'] for c in updated.candidates):
            continue
        candidate = copy.deepcopy(rejected)
        candidate.update(id=f'J{len(updated.candidates)+1}',
                         kind='manually_confirmed_body_boundary_path',
                         terminal_status='manual_boundary_not_terminal_tip',
                         terminal_model_evidence=[], confirmed=False)
        updated.candidates.append(candidate)
        updated.end_regions.append(_end_region(result.mask, candidate['tip_xy'], 24))
    chosen = confirm_boundary(updated, point, cfg)
    result.candidates, result.end_regions = updated.candidates, updated.end_regions
    result.grasp_policy, result.grasp_plans = updated.grasp_policy, updated.grasp_plans
    return chosen
