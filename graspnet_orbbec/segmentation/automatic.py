"""本地自动线材候选与端子模型辅助识别；不依赖手工框或云端服务。"""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import cv2
import numpy as np

from common.config import Config, load_config
from common.paths import ROOT
from segmentation.wire import (Sam2WireSegmenter, SegmentationError, WireResult,
                               _end_region, analyze_mask, analyze_instance_masks)


def line_contrast(image, cfg, search_mask=None):
    """局部明暗对比；可选掩码只限定阈值统计，不改动输入图像。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = max(9, int(min(gray.shape) * 0.035) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    contrast = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
                          cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
    values = contrast if search_mask is None else contrast[search_mask]
    threshold = max(float(cfg.get_path('wire.auto.min_contrast', 12)),
                    float(np.percentile(values, 85)) if values.size else 0.0)
    return contrast, threshold


def seed_search_mask(shape, cfg, roi_xyxy=None):
    """将配置中的半开矩形投到ROI；返回仅供正点采样使用的搜索掩码。"""
    h, w = shape[:2]
    roi = [0, 0, w, h] if roi_xyxy is None else list(roi_xyxy)
    if (len(roi) != 4 or any(type(v) is not int for v in roi)
            or roi[0] < 0 or roi[1] < 0 or roi[2] - roi[0] != w or roi[3] - roi[1] != h):
        raise ValueError('种子ROI必须是与局部图像尺寸一致的原图整数xyxy')
    enabled = cfg.get_path('wire.auto.seed_filter.enabled', False)
    if type(enabled) is not bool:
        raise ValueError('seed_filter.enabled必须为布尔值')
    info = {'filter_enabled': enabled, 'roi_xyxy_original_rgb': roi,
            'search_regions_roi_xyxy': None, 'exclude_regions_roi_xyxy': []}
    if not enabled:
        return np.ones((h, w), bool), info
    frame = cfg.get_path('wire.auto.seed_filter.coordinate_space', 'original_rgb')
    if frame not in ('original_rgb', 'roi'):
        raise ValueError('种子区域coordinate_space只支持original_rgb或roi')

    def rectangles(value, *, allow_null=False):
        if value is None and allow_null:
            return None
        if not isinstance(value, (list, tuple)):
            raise ValueError('种子区域必须是整数xyxy矩形列表')
        output = []
        dx, dy = roi[:2] if frame == 'original_rgb' else (0, 0)
        for rect in value:
            if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                    or any(type(v) is not int for v in rect)
                    or rect[0] > rect[2] or rect[1] > rect[3]):
                raise ValueError('种子区域必须是有序的整数xyxy矩形')
            x1, y1, x2, y2 = rect
            clipped = [max(0, min(w, x1-dx)), max(0, min(h, y1-dy)),
                       max(0, min(w, x2-dx)), max(0, min(h, y2-dy))]
            output.append(clipped)
        return output

    search = rectangles(cfg.get_path('wire.auto.seed_filter.search_regions', None), allow_null=True)
    excluded = rectangles(cfg.get_path('wire.auto.seed_filter.exclude_regions', []))
    allowed = np.ones((h, w), bool) if search is None else np.zeros((h, w), bool)
    for x1, y1, x2, y2 in search or []:
        allowed[y1:y2, x1:x2] = True
    before = int(allowed.sum())
    for x1, y1, x2, y2 in excluded:
        allowed[y1:y2, x1:x2] = False
    info.update(region_coordinate_space=frame, search_regions_roi_xyxy=search,
                exclude_regions_roi_xyxy=excluded, search_pixels_before_exclusion=before)
    return allowed, info


def automatic_line_seeds(image, cfg, extra_points=(), *, roi_xyxy=None, diagnostics=None):
    """生成ROI局部正点；关闭区域过滤时保留原采样，诊断同时提供完整RGB坐标。"""
    from skimage.morphology import skeletonize
    allowed, info = seed_search_mask(image.shape, cfg, roi_xyxy)
    enabled = info['filter_enabled']
    contrast, threshold = line_contrast(image, cfg, allowed if enabled else None)
    binary = (contrast >= threshold).astype(np.uint8)
    before = int(binary.sum())
    if enabled:
        binary[~allowed] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 16]
    skeleton = skeletonize(np.isin(labels, valid))
    parts = [np.argwhere(skeleton & (labels == i)) for i in valid]
    parts = sorted([p for p in parts if len(p) >= 8], key=len, reverse=True)
    maximum = int(cfg.get_path('wire.auto.max_seeds', 64))
    spacing = float(cfg.get_path('wire.auto.seed_spacing_px', 32))
    if maximum < 1 or spacing <= 0:
        raise ValueError('max_seeds 和 seed_spacing_px 必须为正数')
    seeds = [list(map(int, p)) for p in extra_points]
    extra_count = len(seeds)
    if enabled:
        seeds = [p for p in seeds if 0 <= p[0] < image.shape[1]
                 and 0 <= p[1] < image.shape[0] and allowed[p[1], p[0]]]
    extra_kept = len(seeds)
    seeds = seeds[:maximum]
    exhausted = set()
    rounds = 0
    while len(seeds) < maximum and len(exhausted) < len(parts):
        for i, part in enumerate(parts):
            if i in exhausted or len(seeds) >= maximum:
                continue
            if rounds == 0:
                idx = len(part) // 2
            else:
                distances = np.min(((part[:, None, :] - np.asarray(seeds)[None, :, ::-1]) ** 2).sum(2), axis=1)
                idx = int(distances.argmax())
                if distances[idx] < spacing ** 2:
                    exhausted.add(i)
                    continue
            point = part[idx, ::-1].tolist()
            if point not in seeds:
                seeds.append(point)
        rounds += 1
    points = np.asarray(seeds, np.float32).reshape(-1, 2)
    capped = len(seeds) >= maximum
    if diagnostics is not None:
        origin = np.array(info['roi_xyxy_original_rgb'][:2])
        info.update(allowed_pixels=int(allowed.sum()), roi_pixels=int(allowed.size),
                    contrast_threshold=threshold, contrast_pixels_before_filter=before,
                    contrast_pixels_after_filter=int(binary.sum()),
                    contrast_pixels_filtered=before-int(binary.sum()),
                    components_kept=len(valid), sampling_parts=len(parts),
                    extra_points_input=extra_count, extra_points_filtered=extra_count-extra_kept,
                    seed_count=len(points), seed_budget_reached=capped,
                    seed_points_roi_xy=points.tolist(),
                    seed_points_rgb_xy=(points+origin).tolist())
        diagnostics.update(info)
    return points, capped


def draw_seed_debug(image, diagnostics):
    """在局部图上显示搜索/排除区域与实际正点，不改动图像或分割掩码。"""
    vis = image.copy()
    if diagnostics['filter_enabled']:
        allowed = np.ones(image.shape[:2], bool)
        search = diagnostics['search_regions_roi_xyxy']
        if search is not None:
            allowed[:] = False
            for x1, y1, x2, y2 in search:
                allowed[y1:y2, x1:x2] = True
        vis[~allowed] = (vis[~allowed] * 0.35).astype(np.uint8)
        for x1, y1, x2, y2 in diagnostics['exclude_regions_roi_xyxy']:
            patch = vis[y1:y2, x1:x2]
            patch[:] = (patch * .5 + np.array([0, 0, 255]) * .5).astype(np.uint8)
    for point in diagnostics['seed_points_roi_xy']:
        cv2.circle(vis, tuple(map(int, point)), 3, (0, 255, 0), -1)
    return vis


def automatic_background_points(image, points, cfg, roi_xyxy=None):
    """只在操作者确认的背景区内，为每个正点独立选择局部负点；无证据就回退。"""
    enabled = cfg.get_path('wire.auto.negative_prompts.enabled', False)
    if type(enabled) is not bool:
        raise ValueError('negative_prompts.enabled必须为布尔值')
    origin = np.asarray([0, 0] if roi_xyxy is None else roi_xyxy[:2])
    groups = [[] for _ in points]
    info = {'enabled': enabled, 'coordinate_space': 'original_rgb_pixels', 'candidates': [],
            'mixed_candidates': 0, 'positive_only_candidates': len(points)}
    if enabled:
        def number(name, default, *, integer=False):
            value = cfg.get_path(f'wire.auto.negative_prompts.{name}', default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0 or (integer and type(value) is not int)):
                raise ValueError(f'negative_prompts.{name}必须为正的有限数值')
            return value

        maximum = number('max_points', 2, integer=True)
        minimum = number('min_distance_px', 12)
        radius = number('max_distance_px', 80)
        clearance = number('wire_clearance_px', 6)
        max_contrast = number('max_contrast', 6)
        patch_radius = number('patch_radius_px', 2, integer=True)
        patch_range = number('max_patch_range', 8)
        spacing = number('spacing_px', 12)
        if radius <= minimum or maximum > 8 or patch_radius > 32:
            raise ValueError('负点半径范围无效，max_points上限8，patch_radius_px上限32')
        region_cfg = Config({'wire': {'auto': {'seed_filter': {
            'enabled': True,
            'coordinate_space': cfg.get_path('wire.auto.negative_prompts.coordinate_space', 'original_rgb'),
            'search_regions': cfg.get_path('wire.auto.negative_prompts.trusted_regions', []),
        }}}})
        # null在正点搜索中表示全图，但不能在负点中隐式授权全图为可信背景。
        if region_cfg['wire.auto.seed_filter.search_regions'] is None:
            raise ValueError('负点trusted_regions必须显式给出背景矩形，不能为null')
        trusted, regions = seed_search_mask(image.shape, region_cfg, roi_xyxy)
        kernel = np.ones((2*patch_radius+1, 2*patch_radius+1), np.uint8)
        trusted = cv2.erode(trusted.astype(np.uint8), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        contrast, _ = line_contrast(image, cfg)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local_range = cv2.dilate(gray, kernel).astype(float) - cv2.erode(gray, kernel)
        risk = (contrast >= max_contrast).astype(np.uint8)
        for point in points:
            cv2.circle(risk, tuple(map(int, point)), int(np.ceil(clearance)), 1, -1)
        distance = cv2.distanceTransform(1-risk, cv2.DIST_L2, 5)
        safe = trusted & (distance > clearance) & (local_range <= patch_range)
        info.update(trusted_regions_roi_xyxy=regions['search_regions_roi_xyxy'],
                    trusted_pixels=int(trusted.sum()), safe_pixels=int(safe.sum()))
        h, w = safe.shape
        for index, (px, py) in enumerate(points):
            x1, x2 = max(0, int(px-radius)), min(w, int(px+radius)+1)
            y1, y2 = max(0, int(py-radius)), min(h, int(py+radius)+1)
            y, x = np.where(safe[y1:y2, x1:x2])
            candidates = np.c_[x+x1, y+y1]
            distances = ((candidates-[px, py])**2).sum(axis=1)
            keep = (distances >= minimum**2) & (distances <= radius**2)
            for _ in range(maximum):
                indices = np.flatnonzero(keep)
                if not len(indices):
                    break
                chosen = int(indices[np.argmin(distances[indices])])
                point = candidates[chosen]
                groups[index].append(point.tolist())
                keep &= ((candidates-point)**2).sum(axis=1) >= spacing**2
                keep[chosen] = False
    for index, (positive, negatives) in enumerate(zip(points, groups)):
        info['candidates'].append({
            'seed_index': index, 'positive_roi_xy': positive.tolist(),
            'positive_rgb_xy': (positive+origin).tolist(),
            'negative_roi_xy': negatives,
            'negative_rgb_xy': (np.asarray(negatives).reshape(-1, 2)+origin).tolist(),
            'point_labels': [1]+[0]*len(negatives),
            'final_mode': 'positive_negative' if negatives else 'positive_only',
            'fallback_reason': None if negatives or not enabled else 'no_reliable_background',
        })
    return groups, info


def predict_automatic_prompts(predictor, image_shape, points, negatives, batch, min_score, info):
    """按点数分批，保持BxNx2/BxN；负点不满足输出约束时仅重试对应候选。"""
    records = [[] for _ in points]

    def predict(indices, use_negative):
        coordinates = np.asarray([[points[i].tolist()]+(negatives[i] if use_negative else [])
                                  for i in indices], np.float32)
        labels = np.zeros(coordinates.shape[:2], np.int32)
        labels[:, 0] = 1
        masks, scores, _ = predictor.predict(
            point_coords=coordinates, point_labels=labels, multimask_output=True)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if masks.ndim == 3:
            masks, scores = masks[None], scores[None]
        if (masks.ndim != 4 or masks.shape[0] != len(indices)
                or masks.shape[2:] != tuple(image_shape[:2]) or scores.shape != masks.shape[:2]):
            raise ValueError('SAM2返回的提示批次/掩码尺寸不匹配')
        for i, alternatives, qualities in zip(indices, masks, scores):
            x, y = map(int, points[i])
            for mask, score in zip(alternatives, qualities):
                if not np.isfinite(score) or float(score) < min_score or not mask[y, x]:
                    continue
                if use_negative and any(mask[ny, nx] for nx, ny in negatives[i]):
                    continue
                records[i].append({'segmentation': mask, 'predicted_iou': float(score)})

    # 不填充伪点、不广播共享负点；不同负点数量的候选使用独立批次。
    by_count = {}
    for i, group in enumerate(negatives):
        by_count.setdefault(len(group), []).append(i)
    for count, indices in by_count.items():
        for offset in range(0, len(indices), batch):
            chosen = indices[offset:offset+batch]
            try:
                predict(chosen, bool(count))
            except ValueError:
                if not count:
                    raise
                for i in chosen:
                    records[i] = []
                    info['candidates'][i]['fallback_reason'] = 'mixed_prompt_rejected'
            if count:
                retry = [i for i in chosen if not records[i]]
                if retry:
                    for i in retry:
                        item = info['candidates'][i]
                        item['final_mode'] = 'positive_only'
                        item['fallback_reason'] = item['fallback_reason'] or 'no_mask_satisfies_mixed_prompt'
                    predict(retry, False)
    info['mixed_candidates'] = sum(c['final_mode'] == 'positive_negative' for c in info['candidates'])
    info['positive_only_candidates'] = len(points)-info['mixed_candidates']
    for candidate in info['candidates']:
        candidate['final_point_labels'] = (
            candidate['point_labels'] if candidate['final_mode'] == 'positive_negative' else [1])
    return [record for group in records for record in group]


def draw_prompt_debug(image, info):
    """每个候选单独画正点和负点，避免将不同候选的点误读为同一组提示。"""
    vis = image.copy()
    cv2.circle(vis, tuple(map(int, info['positive_roi_xy'])), 4, (0, 255, 0), -1)
    for point in info['negative_roi_xy']:
        x, y = map(int, point)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 2)
    cv2.putText(vis, info['final_mode'], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return vis


def save_prompt_debug(directory, image, info):
    (directory/'prompts.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    for candidate in info['candidates']:
        path = directory/f"prompt_{candidate['seed_index']:03d}.png"
        if not cv2.imwrite(str(path), draw_prompt_debug(image, candidate)):
            raise OSError('正负点调试图保存失败')


def prune_continuing_endpoints(result, cfg):
    """原图线条仍向外延伸时，掩码断口不能当作物理线头；保留拒绝原因供复核。"""
    contrast, threshold = line_contrast(result.image_bgr, cfg)
    h, w = contrast.shape
    candidates, regions, rejected = [], [], []
    for candidate, region in zip(result.candidates, result.end_regions):
        tip = np.asarray(candidate['tip_xy'])
        axis = np.asarray(candidate['tangent_xy'])
        hits = []
        for distance in range(4, 17, 2):
            x, y = np.rint(tip - axis * distance).astype(int)
            if 2 <= x < w - 2 and 2 <= y < h - 2:
                hits.append(bool(contrast[y-2:y+3,x-2:x+3].max() >= threshold))
        if not candidate.get('terminal_model_evidence') and len(hits) >= 3 and np.mean(hits) >= .6:
            rejected.append(dict(candidate, rejection_reason='image_line_continues_past_mask_tip'))
            continue
        if candidate.get('kind') != 'terminal_model_guided_boundary_path':
            candidate['id'] = chr(65+len(candidates)) if len(candidates)<26 else f'E{len(candidates)+1}'
        candidates.append(candidate)
        regions.append(region)
    result.candidates, result.end_regions = candidates, regions
    result.automation['rejected_geometric_ends'] = rejected
    if rejected:
        result.warnings.append(f'原图线条仍在延伸，已排除{len(rejected)}个疑似掩码断口候选')


def select_wire_masks(image, records, cfg, reference_shape=None):
    """筛选线状前景，保留没有可靠端点的线身；几何筛选不赋予类别语义。"""
    h, w = image.shape[:2]
    scale = min(reference_shape or (h, w))
    choices = []
    seen = []
    for record in sorted(records, key=lambda r: float(r['predicted_iou']), reverse=True):
        mask = np.asarray(record['segmentation'], dtype=bool)
        if mask.shape != (h, w):
            continue
        score = float(record['predicted_iou'])
        if not np.isfinite(score) or score < float(cfg.get_path('wire.auto.min_score', 0.5)):
            continue
        if mask.sum() < 12 or mask.mean() > float(cfg.get_path('wire.auto.max_area_fraction', 0.20)):
            continue
        if any(np.logical_and(mask, previous).sum() / np.logical_or(mask, previous).sum() > 0.90 for previous in seen):
            continue
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask)
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        width = 2 * float(np.median(dt[skeleton])) if skeleton.any() else 0
        if not width or width > scale * float(cfg.get_path('wire.auto.max_width_fraction', 0.02)):
            continue
        geometry = analyze_mask(image, mask, inset_px=float(cfg.get_path('wire.inset_px', 12)))
        length = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in geometry.paths_xy)
        if not length:
            length = float(skeleton.sum())
        slenderness = length / max(width, 1)
        if (length < scale * float(cfg.get_path('wire.auto.min_length_fraction', 0.06))
                or slenderness < float(cfg.get_path('wire.auto.min_slenderness', 12))):
            continue
        seen.append(mask)
        choices.append((score * slenderness, mask, {
            'sam_score': score, 'median_width_px': width, 'centerline_length_px': length,
            'slenderness': slenderness,
        }))
    selected = []
    for _, mask, info in sorted(choices, key=lambda row: row[0], reverse=True):
        # 同一根线的包含/重复掩码只保留一份，避免重复输出端部。
        if any(np.logical_and(mask, previous).sum() / np.logical_or(mask, previous).sum() > 0.90
               for previous, _ in selected):
            continue
        selected.append((mask, info))
        if len(selected) >= int(cfg.get_path('wire.auto.max_masks', 32)):
            break
    return selected


def _terminal_pair_evidence(result, detection, terminal_mask, instance, cfg, cache):
    """对一个端子和一个实例评估几何；分数不能覆盖硬拒绝条件。"""
    maximum = float(cfg.get_path('wire.auto.terminal_path_max_distance_px', 12))
    gap_limit = float(cfg.get_path('wire.auto.terminal_match.max_gap_px', 6))
    end_zone = float(cfg.get_path('wire.auto.terminal_match.endpoint_zone_px', 24))
    angle_limit = float(cfg.get_path('wire.auto.terminal_match.max_axis_angle_deg', 35))
    evidence = {'instance_id': instance.instance_id, 'score': 0.0, 'eligible': False,
                'rejection_reasons': [], 'features': {}}
    reasons, features = evidence['rejection_reasons'], evidence['features']
    if instance.ambiguous:
        reasons.append('wire_instance_ambiguous')
    if not instance.paths_xy or len(instance.paths_xy[0]) < 9:
        reasons.append('instance_centerline_unavailable')
        return evidence
    path = np.asarray(instance.paths_xy[0], float)
    ty, tx = np.where(terminal_mask)
    terminal_points = np.c_[tx, ty].astype(float)
    centered = terminal_points-terminal_points.mean(0)
    _, axes = np.linalg.eigh(centered.T @ centered/max(len(centered), 1))
    axis = axes[:, -1]
    extent = np.ptp(centered @ axes, axis=0)+1
    aspect = float(extent[-1]/max(extent[0], 1))
    if aspect < float(cfg.get_path('wire.auto.terminal_match.min_aspect_ratio', 1.5)):
        reasons.append('terminal_axis_unreliable')
    if instance.instance_id not in cache:
        raster = np.zeros(result.mask.shape, np.uint8)
        pixels = np.rint(path).astype(int)
        raster[pixels[:, 1], pixels[:, 0]] = 1
        cache[instance.instance_id] = (
            cv2.distanceTransform(1-raster, cv2.DIST_L2, 5),
            cv2.distanceTransform((~instance.mask).astype(np.uint8), cv2.DIST_L2, 5))
    path_distance, mask_distance = cache[instance.instance_id]
    closest_terminal = terminal_points[int(path_distance[ty, tx].argmin())]
    distances = np.linalg.norm(path-closest_terminal, axis=1)
    index = int(distances.argmin())
    distance = float(distances[index])
    gap = float(mask_distance[ty, tx].min())
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    endpoint = 0 if arc[index] <= arc[-1]-arc[index] else 1
    endpoint_arc = float(min(arc[index], arc[-1]-arc[index]))
    lo, hi = max(0, index-8), min(len(path)-1, index+8)
    tangent = path[hi]-path[lo]
    norm = np.linalg.norm(tangent)
    alignment = abs(float(axis @ tangent/norm)) if norm else 0.0
    angle = float(np.degrees(np.arccos(np.clip(alignment, 0, 1))))
    width = max(float(instance.geometry.get('median_width_px', 0)), 1)
    width_ratio = float(extent[0]/width)
    length_ratio = float(extent[-1]/width)
    if distance > maximum:
        reasons.append('terminal_too_far_from_instance_path')
    if gap > gap_limit:
        reasons.append('no_local_connection_support')
    if endpoint_arc > end_zone:
        reasons.append('terminal_not_near_wire_end')
    if angle > angle_limit:
        reasons.append('terminal_axis_not_aligned_with_wire')
    if not .3 <= width_ratio <= 6 or not 1 <= length_ratio <= 20:
        reasons.append('terminal_wire_geometry_implausible')
    # 只检查本实例到端子的局部连接；其他实例接触该区域时不作最近距离裁决。
    route = terminal_mask.astype(np.uint8).copy()
    cv2.line(route, tuple(np.rint(closest_terminal).astype(int)), tuple(np.rint(path[index]).astype(int)), 1, 1)
    corridor = cv2.dilate(route, np.ones((3, 3), np.uint8)) > 0
    rivals = [other.instance_id for other in result.instances
              if other.instance_id != instance.instance_id and np.any(corridor & other.mask)]
    if rivals:
        reasons.append('competing_instance_local_contact')
    direct = bool(np.any(terminal_mask & instance.mask))
    try:
        confidence = float(detection.get('confidence', 0))
    except (TypeError, ValueError):
        confidence = 0.0
        reasons.append('terminal_confidence_invalid')
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        reasons.append('terminal_confidence_invalid')
        confidence = 0.0
    if confidence < float(cfg.get_path('wire.auto.terminal_confidence', .4)):
        reasons.append('terminal_confidence_below_threshold')
    score = (.25*max(0., 1-distance/maximum)+.30*alignment
             +.25*(1. if direct else max(0., 1-gap/gap_limit))
             +.15*max(0., 1-endpoint_arc/end_zone)+.05*np.clip(confidence, 0, 1))
    features.update(path_distance_px=distance, mask_gap_px=gap, endpoint_arc_px=endpoint_arc,
                    tangent_xy=(tangent/norm).tolist() if norm else None,
                    terminal_axis_xy=axis.tolist(), axis_angle_deg=angle, aspect_ratio=aspect,
                    terminal_width_wire_ratio=width_ratio, terminal_length_wire_ratio=length_ratio,
                    direct_mask_overlap=direct,
                    connection_kind=('overlap' if direct else 'clear_short_gap' if gap <= gap_limit and not rivals
                                     else 'unsupported_or_conflicting_gap'),
                    competing_instance_ids=rivals)
    evidence.update(score=float(score), endpoint_index=endpoint, path_index=index,
                    boundary_xy=path[index].tolist(), eligible=not reasons)
    return evidence


def associate_terminal_instances(result, cfg):
    """语义证据按实例匹配；保留所有评分、拒绝原因及竞争关系，不赋予抓取许可。"""
    minimum = float(cfg.get_path('wire.auto.terminal_match.min_score', .55))
    margin = float(cfg.get_path('wire.auto.terminal_match.ambiguity_margin', .15))
    positive_settings = [cfg.get_path('wire.auto.terminal_path_max_distance_px', 12),
                         cfg.get_path('wire.auto.terminal_match.max_gap_px', 6),
                         cfg.get_path('wire.auto.terminal_match.endpoint_zone_px', 24),
                         cfg.get_path('wire.auto.terminal_match.max_axis_angle_deg', 35),
                         cfg.get_path('wire.auto.terminal_match.min_aspect_ratio', 1.5)]
    if (any(not np.isfinite(float(v)) or float(v) <= 0 for v in positive_settings)
            or not np.isfinite(minimum) or not 0 <= minimum <= 1
            or not np.isfinite(margin) or not 0 <= margin <= 1):
        raise ValueError('端子实例匹配阈值无效')
    cache, reports = {}, []
    ids = [d.get('id') for d in result.terminal_detections]
    for index, detection in enumerate(result.terminal_detections):
        report = {'terminal_id': detection.get('id'), 'status': 'unmatched', 'instance_id': None,
                  'score': None, 'semantic_evidence_only': True, 'rejection_reasons': [], 'matches': []}
        detection['wire_association'] = report
        # 检测关联不写confirmed，保留已存在的人工确认状态。
        if ids.count(detection.get('id')) != 1:
            report['rejection_reasons'].append('duplicate_terminal_id')
        elif index >= len(result.terminal_masks):
            report['rejection_reasons'].append('terminal_mask_missing')
        else:
            mask = np.asarray(result.terminal_masks[index], bool)
            if mask.shape != result.mask.shape or mask.sum() < 6:
                report['rejection_reasons'].append('terminal_mask_invalid')
            else:
                colors = result.image_bgr[mask].astype(float)
                report['appearance_auxiliary'] = {
                    'mean_brightness': float(colors.mean()),
                    'bright_achromatic_fraction': float(np.mean((colors.mean(1) > 160) & (np.ptp(colors, axis=1) < 30))),
                    'used_for_match_decision': False}
                count, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
                report['terminal_connected_components'] = count-1
                report['matches'] = [_terminal_pair_evidence(result, detection, mask, owner, cfg, cache)
                                     for owner in result.instances]
                if count > 2:
                    report['rejection_reasons'].append('terminal_mask_fragmented')
                if not result.instances:
                    report['rejection_reasons'].append('wire_instances_unavailable')
                eligible = sorted([m for m in report['matches'] if m['eligible'] and m['score'] >= minimum],
                                  key=lambda m: m['score'], reverse=True)
                uncertain = any(m['features'].get('path_distance_px', np.inf) <= float(positive_settings[0])
                                and (count > 2 or any(r in m['rejection_reasons'] for r in
                                        ('wire_instance_ambiguous', 'competing_instance_local_contact',
                                         'terminal_axis_unreliable')))
                                for m in report['matches'])
                if uncertain:
                    report.update(status='ambiguous')
                    report['rejection_reasons'].append('wire_identity_or_local_connection_ambiguous')
                elif len(eligible) > 1 and eligible[0]['score']-eligible[1]['score'] <= margin:
                    report.update(status='ambiguous')
                    report['rejection_reasons'].append('multiple_instances_with_similar_support')
                elif eligible and not report['rejection_reasons']:
                    best = eligible[0]
                    report.update(status='matched', instance_id=best['instance_id'], score=best['score'],
                                  endpoint_index=best['endpoint_index'], path_index=best['path_index'],
                                  boundary_xy=best['boundary_xy'])
                elif not report['rejection_reasons']:
                    report['rejection_reasons'].append('no_geometrically_supported_instance')
        reports.append(report)
    # 两个检测竞争同一实例同一端部时拒绝二选一；不同端部可独立匹配。
    assignments = {}
    for report in reports:
        if report['status'] == 'matched':
            assignments.setdefault((report['instance_id'], report['endpoint_index']), []).append(report)
    for group in assignments.values():
        if len(group) > 1:
            for report in group:
                report.update(status='ambiguous', instance_id=None, endpoint_index=None,
                              path_index=None, boundary_xy=None)
                report['rejection_reasons'].append('multiple_terminals_compete_for_same_wire_end')
    result.automation['terminal_association'] = {
        'schema_version': 1, 'coordinate_space': 'original_rgb_pixels',
        'semantic_evidence_only': True, 'reports': reports}
    return reports


def attach_terminals(result, detections, masks, cfg=None):
    """仅把无歧义实例匹配结果作为端部语义证据，不确认交界或生成执行许可。"""
    result.terminal_detections, result.terminal_masks = detections, masks
    settings = cfg if cfg is not None else Config()
    reports = associate_terminal_instances(result, settings)
    owners = {instance.instance_id: instance for instance in result.instances}
    for candidate in result.candidates:
        evidence = []
        owner = owners.get(candidate.get('instance_id'))
        if owner is not None and owner.paths_xy:
            path = owner.paths_xy[0]
            side = int(np.linalg.norm(path[-1]-candidate['tip_xy']) < np.linalg.norm(path[0]-candidate['tip_xy']))
            for detection, report in zip(detections, reports):
                if (report['status'] == 'matched' and report['instance_id'] == owner.instance_id
                        and report['endpoint_index'] == side
                        and np.linalg.norm(np.asarray(candidate['tip_xy'])-report['boundary_xy']) <=
                        float(settings.get_path('wire.auto.terminal_match.endpoint_zone_px', 24))):
                    evidence.append({'id': detection['id'], 'confidence': detection['confidence'],
                                     'instance_id': owner.instance_id, 'match_score': report['score'],
                                     'semantic_evidence_only': True})
        candidate['terminal_model_evidence'] = evidence
        candidate['terminal_status'] = 'model_detected' if evidence else 'geometry_only'
    if not detections:
        result.warnings.append('端子模型未检出端子；当前只有线材几何端部候选')
    elif any(report['status'] == 'ambiguous' for report in reports):
        result.warnings.append('端子与线材实例关联存在歧义，不能据此确认交界')
    elif any(not item['terminal_model_evidence'] for item in result.candidates):
        result.warnings.append('部分几何端部没有可靠的端子实例证据，仍需人工确认')


def attach_terminal_guided_boundaries(result, cfg):
    """可选实验只产生模型交界预测；仍须人工确认和任务5的独立执行复核。"""
    suggestions = {}
    result.automation['terminal_boundary_suggestions'] = suggestions
    result.automation['terminal_guided_candidates'] = []
    if not cfg.get_path('wire.auto.terminal_auto_boundary', False):
        return suggestions
    reports = associate_terminal_instances(result, cfg)
    owners = {instance.instance_id: instance for instance in result.instances}
    inset = float(cfg.get_path('wire.inset_px', 12))
    for detection, report in zip(result.terminal_detections, reports):
        if report['status'] != 'matched':
            continue
        owner = owners[report['instance_id']]
        path, index = owner.paths_xy[0], report['path_index']
        boundary = np.asarray(report['boundary_xy'], float)
        if any(c.get('instance_id') == owner.instance_id
               and np.linalg.norm(np.asarray(c['tip_xy'])-boundary) <= 6 for c in result.candidates):
            continue
        inward = path[index:] if report['endpoint_index'] == 0 else path[:index+1][::-1]
        if len(inward) < inset+5:
            continue
        arc = np.r_[0., np.linalg.norm(np.diff(inward, axis=0), axis=1).cumsum()]
        near = min(int(np.searchsorted(arc, inset)), len(inward)-1)
        tangent = inward[min(len(inward)-1, near+5)]-inward[max(0, near-5)]
        norm = float(np.linalg.norm(tangent))
        if norm <= 0:
            continue
        sequence = 1
        while any(c['id'] == f'Y{sequence}' for c in result.candidates):
            sequence += 1
        cid = f'Y{sequence}'
        evidence = {'id': detection['id'], 'confidence': detection['confidence'],
                    'instance_id': owner.instance_id, 'match_score': report['score'], 'semantic_evidence_only': True}
        region = _end_region(owner.mask, np.rint(boundary).astype(int), 2*inset)
        result.candidates.append({
            'id': cid, 'instance_id': owner.instance_id, 'component_id': int(owner.instance_id[1:]),
            'tip_xy': np.rint(boundary).astype(int).tolist(), 'near_tip_xy': inward[near].astype(int).tolist(),
            'tangent_xy': (tangent/norm).tolist(), 'inward_path_xy': inward.astype(int).tolist(),
            'path_ambiguous': owner.ambiguous, 'inset_px': float(arc[near]),
            'end_region_pixels': int(region.sum()), 'end_region_extent_px': 2*inset,
            'kind': 'terminal_model_guided_boundary_path', 'confirmed': False,
            'terminal_model_evidence': [evidence], 'terminal_status': 'model_detected',
            'boundary_confirmation_source': 'model_prediction',
            'boundary_confirmation': 'instance_geometry_model_prediction'})
        result.end_regions.append(region)
        suggestions[cid] = boundary.tolist()
        detection['guided_candidate_id'], detection['boundary_xy'] = cid, boundary.tolist()
    result.automation['terminal_guided_candidates'] = list(suggestions)
    return suggestions


class AutomaticWireRecognizer:
    def __init__(self, cfg=None, segmenter=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.segmenter = segmenter or Sam2WireSegmenter(self.cfg)
        self.generator = None
        self.terminal_model = None
        self.terminal_model_class = None
        self.terminal_runtime_error = None
        self.generation_stats = {}
        self.terminal_seed_points = []
        self.seed_roi_xyxy = None

    def prepare_terminal_runtime(self):
        """主线程在取流/窗口/推理线程启动前调用，防止库导入修改OpenCV全局线程池。"""
        if (not self.cfg.get_path('wire.auto.terminal_enabled', True)
                or self.terminal_model_class is not None or self.terminal_runtime_error is not None):
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('请先在主线程调用prepare_terminal_runtime，再启动彩色采集和推理线程')
        try:
            runtime = str(ROOT / '.yolo-deps')
            if runtime not in sys.path:
                sys.path.insert(0, runtime)
            os.environ.setdefault('YOLO_OFFLINE', 'true')
            os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
            os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'output/yolo_config'))
            from ultralytics import YOLO
            self.terminal_model_class = YOLO
        except Exception as exc:
            # 缓存失败，不得在窗口已运行后重试首次导入。
            self.terminal_runtime_error = str(exc)

    def _generate(self, image):
        import torch
        method = self.cfg.get_path('wire.auto.method', 'line_prompts')
        debug_enabled = self.cfg.get_path('wire.auto.seed_debug.enabled', False)
        if type(debug_enabled) is not bool:
            raise ValueError('seed_debug.enabled必须为布尔值')
        if method == 'dense' and (debug_enabled or self.cfg.get_path('wire.auto.seed_filter.enabled', False)
                                  or self.cfg.get_path('wire.auto.negative_prompts.enabled', False)):
            raise ValueError('种子过滤/调试/正负点仅支持line_prompts，dense模式请关闭相应开关')
        if method != 'dense':
            extra_points = (
                self.terminal_seed_points
                if self.cfg.get_path('wire.auto.terminal_use_seeds', False)
                else []
            )
            diagnostics = {}
            points, capped = automatic_line_seeds(
                image, self.cfg, extra_points, roi_xyxy=self.seed_roi_xyxy, diagnostics=diagnostics)
            self.generation_stats = {'method': 'line_prompts', 'seed_count': len(points),
                                     'seed_budget_reached': capped, 'seed_points_xy': points.tolist(),
                                     'seed_filter': diagnostics}
            negatives, prompt_info = automatic_background_points(image, points, self.cfg, self.seed_roi_xyxy)
            self.generation_stats['automatic_prompts'] = prompt_info
            if debug_enabled:
                directory = Path(self.cfg.get_path('wire.auto.seed_debug.directory', 'output/wire_seeds'))
                if not directory.is_absolute():
                    directory = ROOT / directory
                # 每次推理独占目录，避免GUI重复刷新覆盖A/B证据。
                directory = directory / f'seeds_{uuid.uuid4().hex}'
                directory.mkdir(parents=True, exist_ok=False)
                if not cv2.imwrite(str(directory/'seeds_roi.png'), draw_seed_debug(image, diagnostics)):
                    raise OSError('种子调试图保存失败')
                (directory/'seeds.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
                self.generation_stats['seed_debug_directory'] = str(directory)
            if not len(points):
                if debug_enabled:
                    save_prompt_debug(directory, image, prompt_info)
                return []
            self.segmenter._load()
            amp = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if self.segmenter.device == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext())
            batch = int(self.cfg.get_path('wire.auto.prompt_batch', 8))
            if batch < 1:
                raise ValueError('prompt_batch 必须为正数')
            with torch.inference_mode(), amp:
                self.segmenter.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                records = predict_automatic_prompts(
                    self.segmenter.predictor, image.shape, points, negatives, batch,
                    float(self.cfg.get_path('wire.auto.min_score', 0.5)), prompt_info)
            if debug_enabled:
                save_prompt_debug(directory, image, prompt_info)
            return records
        self.segmenter._load()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if self.segmenter.device == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext())
        if self.generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self.generator = SAM2AutomaticMaskGenerator(
                self.segmenter.predictor.model,
                points_per_side=int(self.cfg.get_path('wire.auto.points_per_side', 32)),
                points_per_batch=int(self.cfg.get_path('wire.auto.points_per_batch', 32)),
                pred_iou_thresh=float(self.cfg.get_path('wire.auto.min_score', 0.5)),
                stability_score_thresh=float(self.cfg.get_path('wire.auto.stability_score', 0.8)),
                min_mask_region_area=0,
            )
        self.generation_stats = {'method': 'dense', 'seed_count': int(self.cfg.get_path('wire.auto.points_per_side', 32)) ** 2}
        with torch.inference_mode(), amp:
            return self.generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _detect_terminals(self, image):
        import torch
        weights = Path(self.cfg.get_path('wire.auto.terminal_weights', '../best.pt'))
        if not weights.is_absolute():
            weights = ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f'端子权重不存在：{weights}；没有下载或替换模型')
        if self.terminal_model is None:
            self.prepare_terminal_runtime()
            if self.terminal_runtime_error is not None:
                raise RuntimeError(self.terminal_runtime_error)
            self.terminal_model = self.terminal_model_class(str(weights.resolve()))
            if self.terminal_model.task != 'segment':
                raise ValueError('端子模型必须是分割模型，不能把检测框当成端子掩码')
        result = self.terminal_model.predict(
            image, conf=float(self.cfg.get_path('wire.auto.terminal_confidence', 0.4)),
            imgsz=640, device='0' if torch.cuda.is_available() else 'cpu',
            retina_masks=True, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        if result.masks is None or tuple(result.masks.data.shape[1:]) != image.shape[:2]:
            raise ValueError('端子模型未返回与原图同尺寸的像素掩码')
        detections, masks = [], []
        allowed = self.cfg.get_path('wire.auto.terminal_class_ids', [0])
        for box, tensor in zip(result.boxes.cpu(), result.masks.data.cpu()):
            cls = int(box.cls.item())
            if cls not in allowed:
                continue
            name = f'T{len(detections) + 1}'
            detections.append({
                'id': name, 'class_id': cls, 'raw_class_name': result.names[cls],
                'label': self.cfg.get_path('wire.auto.terminal_label', '线材金属端子'),
                'confidence': float(box.conf.item()), 'bbox_xyxy': box.xyxy[0].tolist(),
                'kind': 'terminal_model_detection', 'confirmed': False,
                'mask_file': f'terminal_{name}.png',
            })
            masks.append(tensor.numpy() > 0.5)
        return detections, masks

    def recognize(self, image):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise SegmentationError('自动识别需要原始BGR彩色图像')
        full_image = image
        h, w = image.shape[:2]
        roi = self.cfg.get_path('wire.auto.roi_xyxy')
        reference_size = self.cfg.get_path('wire.auto.roi_reference_size_wh')
        if roi is not None and reference_size is not None:
            if (not isinstance(reference_size, (list, tuple)) or len(reference_size) != 2
                    or any(type(v) is not int or v <= 0 for v in reference_size)
                    or list(reference_size) != [w, h]):
                raise ValueError('固定视角ROI的原图尺寸不匹配，请重新配置ROI；禁止自动缩放像素')
        if roi is None:
            roi = [0, 0, w, h]
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(v) is not int for v in roi)):
            raise ValueError('自动识别工作区必须是原图整数像素xyxy')
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and min(x2-x1, y2-y1) >= 32):
            raise ValueError('自动识别工作区越界或小于32像素')
        image = image[y1:y2, x1:x2]
        self.seed_roi_xyxy = list(roi)
        self.generation_stats = {}
        start = time.perf_counter()
        detections, terminal_masks, terminal_error = [], [], None
        terminal_enabled = self.cfg.get_path('wire.auto.terminal_enabled', True)
        if terminal_enabled:
            try:
                detections, terminal_masks = self._detect_terminals(image)
            except Exception as exc:
                terminal_error = str(exc)
        self.terminal_seed_points = []
        if self.cfg.get_path('wire.auto.terminal_use_seeds', False):
            for mask in terminal_masks:
                dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                if dt.any():
                    y, x = np.unravel_index(dt.argmax(), dt.shape)
                    self.terminal_seed_points.append([int(x), int(y)])
        terminal_s = time.perf_counter() - start
        records = self._generate(image)
        sam_s = time.perf_counter() - start - terminal_s
        selected = select_wire_masks(image, records, self.cfg, reference_shape=full_image.shape[:2])
        if 'seed_points_xy' in self.generation_stats:
            self.generation_stats['seed_points_xy'] = (
                np.asarray(self.generation_stats['seed_points_xy']).reshape(-1, 2)+[x1, y1]).tolist()
        self.generation_stats['coordinate_space'] = 'original_rgb_pixels'
        # 推理可限于工作区，但所有保存像素、交界和手眼射线必须恢复到原始RGB坐标。
        def restore(mask):
            output = np.zeros((h, w), bool)
            output[y1:y2, x1:x2] = mask
            return output
        terminal_masks = [restore(mask) for mask in terminal_masks]
        for detection in detections:
            box = np.asarray(detection['bbox_xyxy']) + [x1, y1, x1, y1]
            detection['bbox_xyxy'] = box.tolist()
        if selected:
            result = analyze_instance_masks(
                full_image, [restore(m) for m, _ in selected], crop_xyxy=roi,
                scores=[info['sam_score'] for _, info in selected],
                inset_px=float(self.cfg.get_path('wire.inset_px', 12)))
        else:
            blank = np.zeros(full_image.shape[:2], bool)
            result = WireResult(full_image.copy(), blank, blank.copy(), np.empty((0, 2), int), [], 0, 0,
                                list(roi), ['未分出线身；端子检测单独保留，可按R重新取景'])
        instance_debug = result.automation.get('duplicate_wire_hypotheses', [])
        result.automation = {
            'duplicate_wire_hypotheses': instance_debug,
            'method': self.generation_stats.get('method', 'line_prompts'), 'manual_prompts': False,
            'generated_masks': len(records), 'selected_masks': [info for _, info in selected],
            'sam_score_aggregation': 'mean_predicted_iou_of_selected_masks',
            'external_api_calls': 0,
            'roi_xyxy_original_rgb': list(roi),
            'sampling': self.generation_stats,
            'timing_s': {'terminal': terminal_s, 'sam': sam_s},
        }
        result.automation['terminal_boundary_suggestions'] = {}
        result.automation['terminal_guided_candidates'] = []
        result.warnings.append('自动线状候选不代表已确认线材类别；遮挡/粘连处不保证分实例')
        if self.generation_stats.get('seed_budget_reached'):
            result.warnings.append('本帧达到自动提示预算，可能漏检；可补点或增加max_seeds')
        for candidate in result.candidates:
            candidate['terminal_model_evidence'] = []
            candidate['terminal_status'] = 'geometry_only'
        if terminal_enabled:
            if terminal_error is None:
                attach_terminals(result, detections, terminal_masks, self.cfg)
                if self.cfg.get_path('wire.auto.terminal_auto_boundary', False):
                    attach_terminal_guided_boundaries(result, self.cfg)
                result.automation['terminal_status'] = 'ok'
            else:
                result.automation['terminal_status'] = 'unavailable'
                result.warnings.append(f'端子模型不可用：{terminal_error}；仅显示几何候选')
        else:
            result.automation['terminal_status'] = 'disabled'
            result.warnings.append('端子模型已关闭；仅显示几何端部候选')
        prune_continuing_endpoints(result, self.cfg)
        result.elapsed_s = time.perf_counter() - start
        return result
