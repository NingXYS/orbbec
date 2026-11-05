"""静止RGB序列的离线稳定性证据；统计框不成为任何帧的几何输入。"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from arm.handeye_bridge import read_json, sha256
from arm.stationary_sequence import verify_stationary_sequence
from common.config import Config
from segmentation.automatic import AutomaticWireRecognizer
from segmentation.grasp_policy import apply_grasp_policy
from segmentation.wire import save_result


@dataclass
class StabilityConfig:
    """全部为实验筛选初值，既非定位精度，也不授予执行资格。"""
    min_detection_ratio: float = .7
    max_center_deviation_px: float = 4.
    max_boundary_deviation_px: float = 5.
    match_center_gate_px: float = 12.
    match_boundary_gate_px: float = 20.
    max_size_ratio: float = 2.
    max_gray_mean_change: float = 20.
    max_wire_deviation_px: float = 4.
    wire_check_frames: int = 1

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{key}必须是有限数值')
            if key == 'wire_check_frames':
                if type(value) is not int or not 0 <= value <= 2:
                    raise ValueError('wire_check_frames只支持0、1或2')
            elif value <= 0:
                raise ValueError(f'{key}必须大于0')
        if self.min_detection_ratio > 1 or self.max_size_ratio <= 1:
            raise ValueError('检出比例或尺寸比例范围无效')


def _box(detection):
    box = np.asarray(detection['bbox_xyxy'], float)
    confidence = detection.get('confidence', 0)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
            or not np.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError('端子检测框/置信度无效')
    return box


def _center(box):
    return (box[:2]+box[2:])/2


def _box_evidence(a, b, cfg):
    x, y = _box(a), _box(b)
    center = float(np.linalg.norm(_center(x)-_center(y)))
    edge = float(np.max(np.abs(x-y)))
    ratio = float(np.max(np.maximum((x[2:]-x[:2])/(y[2:]-y[:2]),
                                    (y[2:]-y[:2])/(x[2:]-x[:2]))))
    eligible = (a.get('class_id', 0) == b.get('class_id', 0)
                and center <= cfg.match_center_gate_px and edge <= cfg.match_boundary_gate_px
                and ratio <= cfg.max_size_ratio)
    return eligible, {'center_distance_px': center, 'max_edge_delta_px': edge, 'size_ratio': ratio}


def _unique_pairs(edges):
    """只接受双方均唯一的边；竞争关联保留歧义，不靠分数强选。"""
    rows, cols = edges.sum(1), edges.sum(0)
    pairs, ambiguous = [], []
    for i, j in np.argwhere(edges):
        (pairs if rows[i] == 1 and cols[j] == 1 else ambiguous).append((int(i), int(j)))
    return pairs, ambiguous


def analyze_detections(frames, reference_id, cfg=None):
    """原图框的保守跨帧关联与统计；输入每帧已独立检测，不修改原框。"""
    cfg = cfg or StabilityConfig()
    if len(frames) < 2 or len({f['frame_id'] for f in frames}) != len(frames):
        raise ValueError('至少需要两张唯一帧')
    reference = next((f for f in frames if f['frame_id'] == reference_id), None)
    if reference is None:
        raise ValueError('参考帧不在序列中')
    stamps = [int(f['host_receive_timestamp_us']) for f in frames]
    if stamps[0] <= 0 or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('帧时间戳缺失或未严格递增')
    tracks, events = [], []

    def birth(frame, index):
        detection = deepcopy(frame['detections'][index])
        _box(detection)
        track = {'track_id': f'TR{len(tracks)+1:03d}', 'anchor': detection,
                 'anchor_frame_id': frame['frame_id'],
                 'reference_detection_index': index if frame is reference else None,
                 'observations': [{'frame_id': frame['frame_id'], 'detection_index': index,
                                   'detection': detection}], 'ambiguous_frames': []}
        tracks.append(track)

    if not reference.get('error'):
        for index in range(len(reference['detections'])):
            birth(reference, index)
    for frame in frames:
        if frame is reference or frame.get('error'):
            continue
        detections = frame['detections']
        edges = np.zeros((len(tracks), len(detections)), bool)
        evidence = {}
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                edges[i, j], evidence[i, j] = _box_evidence(track['anchor'], detection, cfg)
        pairs, ambiguous = _unique_pairs(edges)
        for i, j in pairs:
            tracks[i]['observations'].append({'frame_id': frame['frame_id'], 'detection_index': j,
                                             'detection': deepcopy(detections[j])})
        for i, j in ambiguous:
            if frame['frame_id'] not in tracks[i]['ambiguous_frames']:
                tracks[i]['ambiguous_frames'].append(frame['frame_id'])
        events.append({'frame_id': frame['frame_id'],
                       'matched': [{'track_id': tracks[i]['track_id'], 'detection_index': j,
                                    **evidence[i, j]} for i, j in pairs],
                       'ambiguous_edges': [{'track_id': tracks[i]['track_id'], 'detection_index': j,
                                            **evidence[i, j]} for i, j in ambiguous]})
        used = {j for _, j in pairs+ambiguous}
        for j in range(len(detections)):
            if j not in used:
                birth(frame, j)
    reasons = []
    if any(f.get('error') for f in frames):
        reasons.append('per_frame_detection_failed')
    if any(f.get('gray_mean_delta', 0) > cfg.max_gray_mean_change for f in frames):
        reasons.append('illumination_inconsistent')
    for track in tracks:
        boxes = np.array([_box(o['detection']) for o in track['observations']])
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        anchor = _box(track['anchor'])
        delta = np.linalg.norm(centers-_center(anchor), axis=1)
        ratio = len(boxes)/len(frames)
        failures = []
        if track['reference_detection_index'] is None:
            failures.append('absent_in_reference')
        if ratio < cfg.min_detection_ratio:
            failures.append('low_detection_ratio')
        if track['ambiguous_frames']:
            failures.append('matching_ambiguous')
        if delta.max() > cfg.max_center_deviation_px:
            failures.append('center_unstable')
        if np.abs(boxes-anchor).max() > cfg.max_boundary_deviation_px:
            failures.append('boundary_unstable')
        track.update(detection_ratio=ratio, matched_frames=len(boxes), total_frames=len(frames),
                     center_std_xy_px=centers.std(0).tolist(),
                     max_center_deviation_px=float(delta.max()),
                     boundary_std_xyxy_px=boxes.std(0).tolist(),
                     boundary_range_xyxy_px=np.ptp(boxes, axis=0).tolist(),
                     max_boundary_deviation_px=float(np.abs(boxes-anchor).max()),
                     status='experimental_stable' if not failures else 'manual_confirmation_required',
                     rejection_reasons=failures)
        if failures:
            reasons.append('terminal_tracks_require_manual_confirmation')
    if not tracks:
        reasons.append('no_terminal_tracks')
    return {'tracks': tracks, 'matching_events': events, 'rejection_reasons': sorted(set(reasons)),
            'unmatched_boxes_averaged': False, 'statistics_used_for_geometry': False}


class ReferenceRecognizer(AutomaticWireRecognizer):
    """仅复用这张原图的YOLO结果；ROI仍由现有recognize恢复到原图。"""
    def __init__(self, cfg, image, detections, masks, error=None):
        super().__init__(cfg)
        self.source_digest = hashlib.sha256(image.tobytes()).hexdigest()
        self.cached_detections, self.cached_masks, self.cached_error = detections, masks, error
        self.full_shape = image.shape[:2]

    def recognize(self, image):
        if hashlib.sha256(image.tobytes()).hexdigest() != self.source_digest or image.shape[:2] != self.full_shape:
            raise ValueError('缓存端子与SAM2输入原图不对应')
        return super().recognize(image)

    def _detect_terminals(self, image):
        if self.cached_error:
            raise ValueError(self.cached_error)
        x1, y1, x2, y2 = self.seed_roi_xyxy
        detections, masks = [], []
        for detection, mask in zip(self.cached_detections, self.cached_masks):
            patch = mask[y1:y2, x1:x2]
            if not patch.any():
                continue
            d = deepcopy(detection)
            box = _box(d)
            box[:2] = np.maximum(box[:2], [x1, y1])
            box[2:] = np.minimum(box[2:], [x2, y2])
            d['bbox_xyxy'] = (box-[x1, y1, x1, y1]).tolist()
            detections.append(d)
            masks.append(patch.copy())
        return detections, masks


def compare_wire_instances(reference, other, cfg):
    """逐图实例独立验收；只验证同坐标下连续性，不拼掩码或延长路径。"""
    def path(instance):
        if instance.ambiguous or len(instance.paths_xy) != 1:
            return None
        points = np.asarray(instance.paths_xy[0], float)
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2
                or not np.isfinite(points).all() or np.linalg.norm(np.diff(points, axis=0), axis=1).max() > 1.5):
            return None
        return points

    aa, bb = reference.instances, other.instances
    edges = np.zeros((len(aa), len(bb)), bool)
    evidence = {}
    for i, a in enumerate(aa):
        p = path(a)
        for j, b in enumerate(bb):
            q = path(b)
            if p is None or q is None or a.mask.shape != b.mask.shape:
                continue
            # 距离图保留各自实例，不把合并掩码连通视为同一根线。
            distances = []
            for source, target in ((p, q), (q, p)):
                canvas = np.ones(a.mask.shape, np.uint8)
                xy = np.rint(target).astype(int)
                if np.any(xy < 0) or np.any(xy >= np.array(canvas.shape[::-1])):
                    raise ValueError('实例路径越出RGB边界')
                canvas[xy[:, 1], xy[:, 0]] = 0
                dt = cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                xy = np.rint(source).astype(int)
                if np.any(xy < 0) or np.any(xy >= np.array(canvas.shape[::-1])):
                    raise ValueError('实例路径越出RGB边界')
                distances.extend(dt[xy[:, 1], xy[:, 0]].tolist())
            displacement = float(max(distances))
            endpoint = min(max(np.linalg.norm(p[0]-q[0]), np.linalg.norm(p[-1]-q[-1])),
                           max(np.linalg.norm(p[0]-q[-1]), np.linalg.norm(p[-1]-q[0])))
            u, v = p[-1]-p[0], q[-1]-q[0]
            cosine = abs(float(u@v))/(np.linalg.norm(u)*np.linalg.norm(v)+1e-12)
            width_ratio = (a.mask.sum()/len(p))/max(b.mask.sum()/len(q), 1e-12)
            edges[i, j] = (displacement <= cfg.max_wire_deviation_px
                           and endpoint <= cfg.max_wire_deviation_px and cosine >= np.cos(np.deg2rad(20))
                           and .8 <= len(p)/len(q) <= 1.25 and 1/1.5 <= width_ratio <= 1.5)
            evidence[i, j] = {'max_path_displacement_px': displacement,
                              'endpoint_displacement_px': float(endpoint), 'width_ratio': float(width_ratio)}
    pairs, ambiguous = _unique_pairs(edges)
    passed = bool(aa) and len(pairs) == len(aa) == len(bb) and not ambiguous
    return {'status': 'consistent' if passed else 'manual_confirmation_required',
            'matches': [{'reference_instance_id': aa[i].instance_id, 'other_instance_id': bb[j].instance_id,
                         **evidence[i, j]} for i, j in pairs],
            'ambiguous_pairs': [[aa[i].instance_id, bb[j].instance_id] for i, j in ambiguous],
            'rejection_reasons': [] if passed else ['wire_instance_missing_ambiguous_or_discontinuous']}


def analyze_sequence(bundle, sequence_dir, output, vision_cfg, *, stability=None, reference_id=None,
                     allow_simulated=False, _detector=None, _recognizer=None, _geometry=None):
    """离线完整性→独立YOLO→关联→参考SAM2/少量挑战帧；失败返回人工确认报告。"""
    cfg = stability or StabilityConfig()
    root, out = Path(sequence_dir).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'kind': 'offline_multiframe_wire_analysis',
              'sequence': str(root), 'experimental_thresholds': asdict(cfg), 'safety_standard': False,
              'robot_motion_commands_sent': 0, 'executable': False, 'planning_allowed': False,
              'status': 'manual_confirmation_required', 'rejection_reasons': [],
              'analysis_engine': 'injected_test_predictor' if _detector or _recognizer else 'local_yolo_sam2'}
    started = time.perf_counter()

    def finish():
        report['elapsed_s'] = time.perf_counter()-started
        (out/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return report

    try:
        verify_stationary_sequence(bundle, root, allow_simulated=allow_simulated)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(rejection_reasons=['sequence_validation_failed'], validation_error=str(exc))
        return finish()
    manifest = read_json(root/'manifest.json')
    report['data_origin'] = manifest['data_origin']
    entries = manifest['frames']
    reference_requested = reference_id is not None
    reference_id = reference_id or entries[0]['frame_id']
    reference = next((e for e in entries if e['frame_id'] == reference_id), None)
    if reference is None:
        report['rejection_reasons'] = ['reference_frame_not_in_sequence']
        return finish()
    manifest_digest = sha256(root/'manifest.json')
    vision_cfg = Config(deepcopy(vision_cfg))
    auto = vision_cfg.setdefault('wire', {}).setdefault('auto', {})
    auto.update(terminal_auto_boundary=False, terminal_enabled=True)
    report['vision_wire_config'] = deepcopy(vision_cfg['wire'])
    detector = None if _detector else AutomaticWireRecognizer(vision_cfg)
    runtime_start = time.perf_counter()
    if detector:
        detector.prepare_terminal_runtime()
    runtime_prepare_s = time.perf_counter()-runtime_start
    cache, frames = {}, []
    reference_image = cv2.imread(str(root/reference['rgb']))
    if reference_image is None:
        report['rejection_reasons'] = ['reference_rgb_unreadable']
        return finish()
    gray_ref = float(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY).mean())
    for entry in entries:
        image = cv2.imread(str(root/entry['rgb']))
        start = time.perf_counter()
        detections, masks, error = [], [], None
        try:
            if (image is None or sha256(root/entry['rgb']) != entry['sha256']['capture/rgb.png']
                    or hashlib.sha256(image.tobytes()).hexdigest() != entry['rgb_pixels_sha256']):
                raise ValueError('分析期间原图发生变化')
            detections, masks = (_detector(image) if _detector else detector._detect_terminals(image))
            if len(detections) != len(masks):
                raise ValueError('YOLO框与掩码数量不对应')
            for detection, mask in zip(detections, masks):
                box = _box(detection)
                if (np.any(box < 0) or np.any(box > np.array(image.shape[1::-1]*2))
                        or mask.shape != image.shape[:2] or mask.dtype != bool or not mask.any()):
                    raise ValueError('YOLO输出未使用完整RGB坐标')
        except Exception as exc:
            detections, masks, error = [], [], str(exc)
        elapsed = time.perf_counter()-start
        frame = {'frame_id': entry['frame_id'], 'host_receive_timestamp_us': entry['host_receive_timestamp_us'],
                 'rgb': entry['rgb'], 'detections': deepcopy(detections), 'error': error, 'yolo_elapsed_s': elapsed,
                 'gray_mean_delta': abs(float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())-gray_ref) if image is not None else 0}
        frames.append(frame)
        cache[entry['frame_id']] = (image, detections, masks, error)
        frame_dir = out/'yolo'/f'frame_{len(frames)-1:03d}'
        frame_dir.mkdir(parents=True)
        for i, (d, mask) in enumerate(zip(frame['detections'], masks)):
            filename = frame_dir/f'terminal_{i:03d}.png'
            if not cv2.imwrite(str(filename), mask.astype(np.uint8)*255):
                raise OSError('逐帧端子掩码保存失败')
            d['analysis_mask_file'] = str(filename.relative_to(out))
    report['frames'] = frames
    matched = analyze_detections(frames, reference_id, cfg)
    report.update(matched)
    for track in report['tracks']:
        observations = track['observations']
        first = observations[0]
        anchor_mask = cache[first['frame_id']][2][first['detection_index']]
        contour = anchor_mask.astype(np.uint8)-cv2.erode(anchor_mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
        changes = []
        for observation in observations:
            mask = cache[observation['frame_id']][2][observation['detection_index']]
            other_contour = mask.astype(np.uint8)-cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
            a = cv2.distanceTransform(1-contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            b = cv2.distanceTransform(1-other_contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            changes.append({'frame_id': observation['frame_id'],
                            'mask_iou_to_anchor': float((mask & anchor_mask).sum()/(mask | anchor_mask).sum()),
                            'mask_boundary_max_deviation_px': float(max(a[other_contour > 0].max(), b[contour > 0].max()))})
        track['mask_boundary_changes'] = changes
        if max(c['mask_boundary_max_deviation_px'] for c in changes) > cfg.max_boundary_deviation_px:
            track['status'] = 'manual_confirmation_required'
            track['rejection_reasons'].append('mask_boundary_unstable')
            report['rejection_reasons'].append('terminal_tracks_require_manual_confirmation')
    report['reference'] = {'frame_id': reference_id, 'selection': 'explicit_id' if reference_requested else 'first_manifest_frame',
                           'snapshot': str(root/reference['snapshot']), 'rgb': reference['rgb'],
                           'rgb_sha256': reference['sha256']['capture/rgb.png'],
                           'host_receive_timestamp_us': reference['host_receive_timestamp_us'],
                           'device_timestamp_us': reference['device_timestamp_us'],
                           'calibration_sha256': reference['calibration_sha256'], 'manifest_sha256': manifest_digest,
                           'pair_sha256': reference['sha256']['pair.json'],
                           'pixel_source_frame_id': reference_id, 'pose_source_frame_id': reference_id,
                           'cross_frame_pixels_used_for_geometry': False}
    sam_times, results = {}, {}

    def sam(frame_id):
        image, detections, masks, error = cache[frame_id]
        entry = next(e for e in entries if e['frame_id'] == frame_id)
        if image is None or hashlib.sha256(image.tobytes()).hexdigest() != entry['rgb_pixels_sha256']:
            raise ValueError('推理输入像素与原始快照不对应')
        start = time.perf_counter()
        try:
            result = (_recognizer(frame_id, image, detections, masks) if _recognizer else
                      ReferenceRecognizer(vision_cfg, image, detections, masks, error).recognize(image))
        finally:
            sam_times[frame_id] = time.perf_counter()-start
        if not np.array_equal(result.image_bgr, image):
            raise ValueError('SAM2输出原图与指定参考快照不对应')
        results[frame_id] = result
        return result

    result = None
    try:
        result = sam(reference_id)
        report['reference_terminal_instance_associations'] = [
            {'terminal_id': d['id'], 'wire_association': d.get('wire_association')}
            for d in result.terminal_detections]
        if (result.automation.get('terminal_status') == 'unavailable'
                or len(result.terminal_detections) != len(cache[reference_id][1])
                or any(d.get('wire_association', {}).get('status') != 'matched'
                       for d in result.terminal_detections)):
            report['rejection_reasons'].append('reference_terminal_instance_requires_manual_confirmation')
        # 选最具挑战的少量帧，不要求每张都跑SAM2。
        displacement = {f['frame_id']: f['gray_mean_delta']/max(cfg.max_gray_mean_change, 1e-12) for f in frames}
        for event in matched['matching_events']:
            displacement[event['frame_id']] += max((m['center_distance_px'] for m in event['matched']), default=0)
        challenges = sorted((f for f in frames if f['frame_id'] != reference_id),
                            key=lambda f: (displacement[f['frame_id']], int(f['host_receive_timestamp_us'])), reverse=True)
        checks = []
        for frame in challenges[:cfg.wire_check_frames]:
            other = sam(frame['frame_id'])
            checks.append({'frame_id': frame['frame_id'], **compare_wire_instances(result, other, cfg)})
        report['wire_checks'] = checks
        if not checks or any(c['status'] != 'consistent' for c in checks):
            report['rejection_reasons'].append('cross_frame_wire_validation_required')
        # 在任何交界/三维处理前重验快照，防止文件在推理期间被替换。
        verify_stationary_sequence(bundle, root, allow_simulated=allow_simulated)
        if sha256(root/'manifest.json') != manifest_digest:
            raise ValueError('分析期间manifest被替换')
        apply_grasp_policy(result, vision_cfg)
        if _geometry:
            if manifest['data_origin'] != 'physical_camera':
                raise ValueError('模拟序列禁止三维手眼处理')
            _geometry(result, root/reference['snapshot'])
        result.automation['multiframe_reference'] = deepcopy(report['reference'])
        result.grasp_policy.update(executable=False, planning_allowed=False)
        result.grasp_policy.setdefault('readiness_blockers', []).append('OFFLINE_MULTIFRAME_ANALYSIS_ONLY')
        for plan in result.grasp_plans:
            plan.update(executable=False, planning_allowed=False)
            plan.setdefault('readiness_blockers', []).append('OFFLINE_MULTIFRAME_ANALYSIS_ONLY')
        save_result(result, out/'reference')
        report['reference_recognition'] = 'reference/recognition.json'
    except Exception as exc:
        report['rejection_reasons'].append('reference_or_wire_analysis_failed')
        report['sam_or_geometry_error'] = str(exc)
    report['timing_s'] = {'runtime_prepare': runtime_prepare_s,
                          'yolo_per_frame': {f['frame_id']: f['yolo_elapsed_s'] for f in frames},
                          'yolo_total': sum(f['yolo_elapsed_s'] for f in frames), 'sam_and_instances_per_frame': sam_times}
    stable = [t for t in report['tracks'] if t['status'] == 'experimental_stable']
    ref_frame = next(f for f in frames if f['frame_id'] == reference_id)
    report['ab_comparison'] = {
        'same_reference_frame_id': reference_id, 'shared_reference_sam_inference': True,
        'single_frame': {'terminal_count': len(ref_frame['detections']), 'temporal_evidence': 'unavailable',
                         'yolo_elapsed_s': ref_frame['yolo_elapsed_s'],
                         'sam_and_instances_elapsed_s': sam_times.get(reference_id)},
        'multi_frame': {'frames': len(frames), 'experimental_stable_tracks': len(stable),
                        'manual_review_tracks': len(report['tracks'])-len(stable),
                        'yolo_elapsed_s': report['timing_s']['yolo_total'], 'sam_frames': len(sam_times)},
        'accuracy_measured': False, 'ground_truth': 'not_provided',
        'timing_interpretation': 'observed costs within one run; shared reference SAM; not independent cold-start trials',
        'interpretation': '同一参考帧比较单帧证据与跨帧稳定性证据；不以稳定性冒充检测准确率或执行许可'}
    report['rejection_reasons'] = sorted(set(report['rejection_reasons']))
    report['manual_confirmation'] = {'snapshot': report['reference']['snapshot'],
                                      'action': 'review this reference image; Shift+left-click its boundary in existing preview',
                                      'automatic_boundary_enabled': False, 'automatic_execution_enabled': False}
    if not report['rejection_reasons']:
        report['status'] = 'experimental_consistent_manual_boundary_required'
    return finish()
