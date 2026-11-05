"""局部14mm视觉路径验收；独立复算证据，不修改弧长、手眼或TCP算法。"""

import hashlib

import cv2
import numpy as np


PREFIX = 'visual_path_'


def instance_mask_digest(mask):
    array = np.asarray(mask, np.uint8)
    return hashlib.sha256(np.asarray(array.shape, dtype=np.int64).tobytes()+array.tobytes()).hexdigest()


def manual_mask_review(result):
    """手动替换自动目标假设后，必须单独复核掩码；交界确认不能代替该步骤。"""
    if result.automation.get('method') != 'manual_prompts':
        return {'required': False, 'passed': True, 'rejection_reasons': []}
    record = result.automation.get('manual_mask_review') or {}
    if not isinstance(record, dict):
        record = {}
    passed = (len(result.instances) == 1 and not result.instances[0].ambiguous
              and record.get('schema_version') == 1 and record.get('human_reviewed') is True
              and record.get('instance_id') == result.instances[0].instance_id
              and record.get('operator_contract') == 'target_body_mask_and_visible_no_touch_crossing_reviewed'
              and record.get('image_rgb_sha256') == hashlib.sha256(result.image_bgr.tobytes()).hexdigest()
              and record.get('mask_sha256') == instance_mask_digest(result.instances[0].mask))
    return {'required': True, 'passed': bool(passed),
            'rejection_reasons': [] if passed else [PREFIX+'manual_mask_review_required']}


def encode_instance_evidence(result):
    """独立保存所有实例的原图二值游程，供最终层核对局部证据是否漏掉其他实例。"""
    masks = []
    for instance in result.instances:
        flat = instance.mask.reshape(-1).astype(np.uint8)
        edges = np.diff(np.r_[0, flat, 0].astype(np.int16))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        masks.append({'instance_id': instance.instance_id,
                      'runs': [[int(a), int(b-a)] for a, b in zip(starts, ends)]})
    return {'schema_version': 1, 'shape_hw': list(result.mask.shape), 'masks': masks}


def _check_instance_patch(recognition, candidate, evidence):
    package = recognition['grasp_policy']['instance_mask_evidence']
    shape = package['shape_hw']
    if (package.get('schema_version') != 1 or not isinstance(shape, list) or len(shape) != 2
            or any(type(v) is not int or v <= 0 or v > 4096 for v in shape)):
        return False
    records = package['masks']
    ids = [i['instance_id'] for i in records]
    expected = [i['instance_id'] for i in recognition['instances']]
    if len(set(ids)) != len(ids) or len(set(expected)) != len(expected) or set(ids) != set(expected):
        return False
    owner, others = _mask(evidence['owner_mask_rows']), _mask(evidence['other_mask_rows'])
    x, y = evidence['patch_origin_xy']
    h, w = owner.shape
    if x < 0 or y < 0 or x+w > shape[1] or y+h > shape[0]:
        return False
    other = np.zeros_like(others)
    found = False
    for instance in records:
        flat = np.zeros(shape[0]*shape[1], bool)
        previous = 0
        for start, length in instance['runs']:
            if (type(start) is not int or type(length) is not int or length <= 0
                    or start < previous or start+length > len(flat)):
                return False
            flat[start:start+length] = True
            previous = start+length
        patch = flat.reshape(shape)[y:y+h, x:x+w]
        if instance['instance_id'] == candidate['instance_id']:
            if not np.array_equal(patch, owner):
                return False
            found = True
        else:
            other |= patch
    return found and np.array_equal(other, others)


def _array(value, dimensions):
    data = np.asarray(value, float)
    if data.ndim != 2 or data.shape[1] != dimensions or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('path_array')
    return data


def _rows(mask):
    return [''.join('1' if v else '0' for v in row) for row in mask]


def _mask(rows):
    if (not isinstance(rows, list) or not rows or not isinstance(rows[0], str)
            or not 0 < len(rows[0]) <= 2048 or len(rows) > 2048):
        raise ValueError('mask_rows')
    if any(not isinstance(row, str) or len(row) != len(rows[0]) or set(row)-{'0', '1'} for row in rows):
        raise ValueError('mask_rows')
    return np.array([[v == '1' for v in row] for row in rows], bool)


def _thin_mask(mask):
    """局部二值细化仅依赖NumPy，避免官方系统Python依赖视觉Conda的skimage。"""
    data = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            center = data[1:-1, 1:-1]
            p = [data[:-2, 1:-1], data[:-2, 2:], data[1:-1, 2:], data[2:, 2:],
                 data[2:, 1:-1], data[2:, :-2], data[1:-1, :-2], data[:-2, :-2]]
            neighbors = np.stack(p).sum(axis=0)
            transitions = np.stack([(a == 0) & (b == 1) for a, b in zip(p, p[1:]+p[:1])]).sum(axis=0)
            first, second = ((p[0]*p[2]*p[4], p[2]*p[4]*p[6]) if phase == 0 else
                             (p[0]*p[2]*p[6], p[0]*p[4]*p[6]))
            remove = (center > 0) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & (first == 0) & (second == 0)
            if remove.any():
                center[remove] = 0
                changed = True
        if not changed:
            return data[1:-1, 1:-1].astype(bool)


def _local_segment(candidate, plan, xyz):
    from segmentation.phase17_geometry.policy import select_center
    from segmentation.phase17_geometry.boundary import project_boundary_to_path

    uv = _array(candidate['inward_path_xy'], 2)
    xyz = _array(xyz, 3)
    if len(uv) != len(xyz) or np.any(xyz[:, 2] <= 0):
        raise ValueError('metric_path')
    index = plan['boundary_path_index']
    if type(index) is not int or not 0 <= index < len(uv)-1:
        raise ValueError('boundary_index')
    boundary = np.asarray(plan['boundary_xy'], float)
    if boundary.shape != (2,) or not np.allclose(boundary, uv[index], atol=1e-6):
        raise ValueError('boundary_mismatch')
    clicked = np.asarray(plan['boundary_input_xy'], float)
    if (clicked.shape != (2,) or not np.isfinite(clicked).all()
            or np.linalg.norm(clicked-uv[index]) > 12
            or int(np.linalg.norm(uv-clicked, axis=1).argmin()) != index):
        raise ValueError('boundary_mismatch')
    projection = plan.get('boundary_projection') or {}
    if 'manual_boundary_camera_m' in projection:
        tail, anchor = project_boundary_to_path(xyz, projection['manual_boundary_camera_m'])
        i, fraction = anchor['centerline_segment'], anchor['fraction']
        raw = np.vstack([np.asarray(anchor['anchor_camera_m']), xyz[i+1:]])
        keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-10]
        tail_uv = np.vstack([uv[i]*(1-fraction)+uv[i+1]*fraction, uv[i+1:]])[keep]
    else:
        tail, tail_uv = xyz[index:], uv[index:]
    selected = select_center(tail, anchor_kind='original_boundary', direction='toward_insulation',
                             scale_evidence='MODEL_DERIVED')
    if not np.allclose(selected['center_m'], plan['grasp_camera_m'], atol=1e-8):
        raise ValueError('center_mismatch')
    i, fraction = selected['segment_index'], selected['segment_fraction']
    segment = np.vstack([tail_uv[:i+1], tail_uv[i]*(1-fraction)+tail_uv[i+1]*fraction])
    segment_xyz = np.vstack([tail[:i+1], np.asarray(selected['center_m'])])
    return segment, selected, segment_xyz


def validate_path_evidence(candidate, plan, evidence):
    """从原始局部掩码/像素及米制路径复算，不信任上游passed或path_ambiguous布尔值。"""
    result = {'schema_version': 1, 'passed': False, 'rejection_reasons': [], 'checks': {}}
    reasons = result['rejection_reasons']
    try:
        if not isinstance(evidence, dict) or evidence.get('schema_version') != 1:
            raise ValueError('schema')
        if (not candidate.get('instance_id') or candidate.get('instance_id') != evidence.get('instance_id')
                or plan.get('instance_id') != candidate.get('instance_id')
                or candidate.get('id') != evidence.get('candidate_id')
                or plan.get('candidate_id') != candidate.get('id')):
            reasons.append(PREFIX+'instance_mismatch')
        if candidate.get('path_ambiguous') or evidence.get('instance_ambiguity_reasons'):
            reasons.append(PREFIX+'instance_ambiguous')
        if float(plan.get('offset_mm', 0)) != 14:
            reasons.append(PREFIX+'offset_not_14mm')
        segment, selected, segment_xyz = _local_segment(candidate, plan, evidence['path_camera_m'])
        if not np.allclose(selected['tangent'], plan['wire_tangent_camera'], atol=1e-8):
            reasons.append(PREFIX+'metric_tangent_mismatch')
        owner, others = _mask(evidence['owner_mask_rows']), _mask(evidence['other_mask_rows'])
        if owner.shape != others.shape:
            raise ValueError('mask_shape')
        origin = np.asarray(evidence['patch_origin_xy'])
        if origin.shape != (2,) or origin.dtype.kind not in 'iu' or np.any(origin < 0):
            raise ValueError('patch_origin')
        local = segment-origin
        if np.any(local < 0) or np.any(local[:, 0] > owner.shape[1]-1) or np.any(local[:, 1] > owner.shape[0]-1):
            raise ValueError('path_outside_patch')
        steps = np.linalg.norm(np.diff(local, axis=0), axis=1)
        if np.any(steps > np.sqrt(2)+1e-5):
            reasons.append(PREFIX+'pixel_path_disconnected')
        pixels = np.rint(local).astype(int)
        distinct_steps = pixels[np.r_[True, np.any(np.diff(pixels, axis=0), axis=1)]]
        if len(distinct_steps) < 2 or len(np.unique(distinct_steps, axis=0)) != len(distinct_steps):
            reasons.append(PREFIX+'path_revisits_pixels')
        route = np.zeros_like(owner, np.uint8)
        cv2.polylines(route, [pixels.reshape(-1, 1, 2)], False, 1, 1)
        if not owner[pixels[:, 1], pixels[:, 0]].all() or np.any((route > 0) & ~owner):
            reasons.append(PREFIX+'outside_instance_mask')
        count, components = cv2.connectedComponents(owner.astype(np.uint8), 8)
        ids = components[pixels[:, 1], pixels[:, 0]]
        if np.any(ids == 0) or len(np.unique(ids)) != 1:
            reasons.append(PREFIX+'instance_disconnected')
        corridor = cv2.dilate(route, np.ones((5, 5), np.uint8)) > 0
        if np.any(corridor & others):
            reasons.append(PREFIX+'competing_instance')
        from segmentation.wire import _skeleton_graph
        skeleton = _thin_mask(owner)
        nodes, graph = _skeleton_graph(skeleton)
        branch = [node for node, neighbors in zip(nodes, graph) if len(neighbors) > 2
                  and corridor[tuple(node)]]
        if branch:
            reasons.append(PREFIX+'branch_detected')
        width = 2*cv2.distanceTransform(owner.astype(np.uint8), cv2.DIST_L2, 5)[pixels[:, 1], pixels[:, 0]]
        median = float(np.median(width))
        low_width = float(np.percentile(width, 10))
        if low_width <= 0 or np.max(width) > max(1.8*low_width, low_width+4):
            reasons.append(PREFIX+'width_abnormal')
        step = min(4, (len(local)-1)//2)
        if step < 1:
            reasons.append(PREFIX+'direction_unavailable')
            turn = None
        else:
            a, b = local[step:-step]-local[:-2*step], local[2*step:]-local[step:-step]
            norm = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
            good = norm > 1e-9
            turn = float(np.degrees(np.arccos(np.clip((a[good]*b[good]).sum(1)/norm[good], -1, 1))).max()) if good.any() else None
            if turn is None or turn > 75:
                reasons.append(PREFIX+'direction_discontinuity')
            a3, b3 = segment_xyz[step:-step]-segment_xyz[:-2*step], segment_xyz[2*step:]-segment_xyz[step:-step]
            norms3 = np.linalg.norm(a3, axis=1)*np.linalg.norm(b3, axis=1)
            good3 = norms3 > 1e-15
            if not good3.any() or np.any(np.degrees(np.arccos(np.clip(
                    (a3[good3]*b3[good3]).sum(1)/norms3[good3], -1, 1))) > 75):
                reasons.append(PREFIX+'metric_direction_discontinuity')
        result['checks'] = {'local_path_samples': len(segment), 'offset_m': selected['offset_m'],
                            'median_width_px': median, 'max_width_px': float(width.max()),
                            'max_turn_deg': turn, 'branch_nodes': len(branch)}
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        reasons.append(PREFIX+'evidence_invalid')
        result['detail'] = str(exc)
    result['rejection_reasons'] = list(dict.fromkeys(reasons))
    result['passed'] = not result['rejection_reasons']
    return result


def build_path_validation(result, candidate, plan, path_camera_m):
    """保留可重复验收的局部掩码证据；缺实例/尺度时仅保留拒绝记录。"""
    output = {'schema_version': 1, 'passed': False, 'rejection_reasons': [], 'evidence': None}
    try:
        owners = [i for i in result.instances if i.instance_id == candidate.get('instance_id')]
        if len(owners) != 1:
            raise ValueError('instance_missing')
        if path_camera_m is None or plan.get('grasp_camera_m') is None:
            raise ValueError('metric_path_missing')
        segment, _, _ = _local_segment(candidate, plan, path_camera_m)
        h, w = result.mask.shape
        low, high = np.floor(segment.min(0)).astype(int)-8, np.ceil(segment.max(0)).astype(int)+9
        x1, y1 = np.maximum(low, 0)
        x2, y2 = np.minimum(high, [w, h])
        owner = owners[0]
        other = np.zeros((y2-y1, x2-x1), bool)
        for instance in result.instances:
            if instance.instance_id != owner.instance_id:
                other |= instance.mask[y1:y2, x1:x2]
        evidence = {'schema_version': 1, 'candidate_id': candidate['id'],
                    'instance_id': owner.instance_id,
                    'instance_ambiguity_reasons': list(owner.ambiguity_reasons),
                    'image_rgb_sha256': hashlib.sha256(np.ascontiguousarray(result.image_bgr).tobytes()).hexdigest(),
                    'patch_origin_xy': [int(x1), int(y1)],
                    'owner_mask_rows': _rows(owner.mask[y1:y2, x1:x2]),
                    'other_mask_rows': _rows(other), 'path_camera_m': np.asarray(path_camera_m).tolist()}
        output.update(validate_path_evidence(candidate, plan, evidence), evidence=evidence)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        code = str(exc)
        output['rejection_reasons'] = [PREFIX+code if code in {
            'instance_missing', 'metric_path_missing', 'center_mismatch', 'boundary_mismatch',
            'boundary_index', 'metric_path', 'path_array'} else PREFIX+'evidence_invalid']
        output['detail'] = code
    return output


def _execution_path_review(recognition, plan):
    """最终适配层重新检查候选、实例、坐标绑定与局部证据，旧JSON默认只可分析。"""
    validation = plan.get('visual_path_validation')
    if not isinstance(validation, dict) or not isinstance(validation.get('evidence'), dict):
        return {'passed': False, 'rejection_reasons': [PREFIX+'validation_missing']}
    candidates = [c for c in recognition.get('candidates', []) if c.get('id') == plan.get('candidate_id')]
    if len(candidates) != 1:
        return {'passed': False, 'rejection_reasons': [PREFIX+'candidate_missing_or_duplicate']}
    candidate = candidates[0]
    check = validate_path_evidence(candidate, plan, validation['evidence'])
    reasons = check['rejection_reasons']
    try:
        if not _check_instance_patch(recognition, candidate, validation['evidence']):
            reasons.append(PREFIX+'instance_mask_evidence_mismatch')
    except (KeyError, ValueError, TypeError, IndexError):
        reasons.append(PREFIX+'instance_mask_evidence_missing_or_invalid')
    if (validation.get('schema_version') != 1 or validation.get('passed') is not True
            or validation.get('rejection_reasons') != []):
        reasons.append(PREFIX+'upstream_validation_not_passed')
    instances = [i for i in recognition.get('instances', []) if i.get('instance_id') == candidate.get('instance_id')]
    if len(instances) != 1 or recognition.get('wire_instances_schema_version') != 1:
        reasons.append(PREFIX+'instance_evidence_missing')
    elif instances[0].get('instance_ambiguous') is not False or instances[0].get('ambiguity_reasons'):
        reasons.append(PREFIX+'instance_ambiguous')
    boundary = plan.get('boundary_verification') or {}
    if (boundary.get('source') != 'manual' or boundary.get('human_confirmed') is not True
            or boundary.get('point_available') is not True):
        reasons.append(PREFIX+'boundary_not_manually_verified')
    expected_source = candidate.get('boundary_confirmation_source') or (
        'model_prediction' if candidate.get('kind') == 'terminal_model_guided_boundary_path' else 'manual')
    if boundary.get('source') != expected_source:
        reasons.append(PREFIX+'boundary_source_mismatch')
    policy = recognition.get('grasp_policy') or {}
    automatic = recognition.get('automation') or {}
    if (automatic.get('method') == 'manual_prompts' or policy.get('mask_origin') == 'manual_prompts'
            or automatic.get('manual_prompts') is True):
        record = automatic.get('manual_mask_review') or {}
        review = policy.get('manual_mask_review') or {}
        try:
            package = policy['instance_mask_evidence']
            shape = package['shape_hw']
            if (package.get('schema_version') != 1 or not isinstance(shape, list) or len(shape) != 2
                    or any(type(v) is not int or v <= 0 or v > 4096 for v in shape)):
                raise ValueError('manual_mask_shape')
            owner = next(item for item in package['masks'] if item['instance_id'] == candidate['instance_id'])
            mask = np.zeros(shape[0]*shape[1], np.uint8)
            previous = 0
            for start, length in owner['runs']:
                if (type(start) is not int or type(length) is not int or start < previous or length <= 0
                        or start+length > mask.size):
                    raise ValueError('manual_mask_runs')
                mask[start:start+length] = 1
                previous = start+length
            if (automatic.get('method') != 'manual_prompts' or policy.get('mask_origin') != 'manual_prompts'
                    or len(recognition['instances']) != 1 or review.get('passed') is not True
                    or review.get('required') is not True or review.get('rejection_reasons') != []
                    or record.get('schema_version') != 1
                    or record.get('human_reviewed') is not True or record.get('instance_id') != candidate['instance_id']
                    or record.get('operator_contract') != 'target_body_mask_and_visible_no_touch_crossing_reviewed'
                    or record.get('image_rgb_sha256') != policy.get('image_rgb_sha256')
                    or record.get('mask_sha256') != instance_mask_digest(mask.reshape(shape))):
                raise ValueError('manual_mask_review')
        except (KeyError, ValueError, TypeError, IndexError, StopIteration, AttributeError):
            reasons.append(PREFIX+'manual_mask_review_required')
    if policy.get('local_path_validation_schema_version') != 1:
        reasons.append(PREFIX+'validation_schema_missing')
    digest = policy.get('image_rgb_sha256')
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or digest != validation['evidence'].get('image_rgb_sha256')):
        reasons.append(PREFIX+'image_binding_missing_or_mismatch')
    stages = plan.get('verification_stages') or {}
    if (stages.get('instance_path_verified') is not True or stages.get('coordinate_calculation_succeeded') is not True
            or stages.get('human_confirmed') is not True or stages.get('boundary_source') != 'manual'):
        reasons.append(PREFIX+'verification_stages_incomplete')
    try:
        from arm.wire_geometry import grasp_transforms, pixels_on_plane
        coordinates = plan['coordinate_verification']
        expected_path = pixels_on_plane(candidate['inward_path_xy'], np.asarray(coordinates['K'], float),
                                        np.asarray(coordinates['D'], float),
                                        policy['support_plane'], policy['center_height_over_board_m'])
        if not np.allclose(expected_path, validation['evidence']['path_camera_m'], atol=1e-9):
            raise ValueError('metric_path_projection_mismatch')
        computed = grasp_transforms(plan, coordinates['T_base_rgb'], coordinates['T_flange_grasp'])
        for name in ('T_base_grasp', 'T_base_flange_grasp', 'T_base_flange_pregrasp'):
            if not np.allclose(computed[name], plan[name], atol=1e-8):
                raise ValueError('coordinate_mismatch')
    except (KeyError, ValueError, TypeError, IndexError, cv2.error):
        reasons.append(PREFIX+'coordinate_evidence_invalid_or_missing')
    check['rejection_reasons'] = list(dict.fromkeys(reasons))
    check['passed'] = not check['rejection_reasons']
    return check


def execution_path_review(recognition, plan):
    """安全边界：格式损坏或缺字段只产生阻断，不能退化成默认允许执行。"""
    try:
        return _execution_path_review(recognition, plan)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, cv2.error):
        return {'passed': False, 'rejection_reasons': [PREFIX+'evidence_invalid']}
