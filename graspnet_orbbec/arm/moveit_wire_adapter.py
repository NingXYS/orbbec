"""把线材识别结果验证并转换为官方 MoveIt 位姿目标；本模块不访问 ROS。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from arm.grasp_angle import closing_angle_review

from segmentation.path_validation import execution_path_review


TARGET_STATUS = 'base_frame_preview_ready'
NOMINAL_TOOL_BLOCKER = 'nominal_finger_tip_reference_not_verified_grasp_center'


def _finite_vector(value, length, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f'{label}必须是有限{length}维数值')
    return vector


def rigid_transform(value, label='T_base_flange'):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)):
        raise ValueError(f'{label}必须是有限米制4x4刚体矩阵')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(rotation) - 1.0) > 1e-6):
        raise ValueError(f'{label}旋转部分不是右手正交矩阵')
    return transform.copy()


def matrix_to_quaternion(rotation):
    """返回单位四元数xyzw；w<0时整体取反，保证轨迹报告符号稳定。"""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('旋转矩阵必须是有限3x3')
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ])
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = np.array([
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ])
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = np.array([
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def transform_to_pose(value, label='位姿'):
    transform = rigid_transform(value, label)
    position = transform[:3, 3]
    quaternion = matrix_to_quaternion(transform[:3, :3])
    return {
        'position': {'x': float(position[0]), 'y': float(position[1]), 'z': float(position[2])},
        'orientation': {
            'x': float(quaternion[0]), 'y': float(quaternion[1]),
            'z': float(quaternion[2]), 'w': float(quaternion[3]),
        },
    }


def _rotation_angle(left, right):
    relative = np.asarray(left, dtype=float).T @ np.asarray(right, dtype=float)
    return math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def load_recognition(path):
    source = Path(path).resolve(strict=True)
    recognition = json.loads(source.read_text())
    if not isinstance(recognition, dict) or not isinstance(recognition.get('grasp_plans'), list):
        raise ValueError('recognition.json缺少grasp_plans列表')
    return source, recognition


def select_grasp_plan(recognition, candidate_id=None):
    plans = recognition.get('grasp_plans', [])
    ready = [plan for plan in plans if plan.get('status') == TARGET_STATUS]
    if candidate_id is None:
        if len(ready) == 1:
            return ready[0]
        if len(ready) > 1:
            ids = [plan.get('candidate_id') for plan in ready]
            raise ValueError(f'需要明确--candidate-id；当前合格目标数为{len(ready)}，ID={ids}')
        if len(plans) == 1:
            return plans[0]
        raise ValueError('需要明确--candidate-id；当前没有合格三维目标')
    matches = [plan for plan in plans if plan.get('candidate_id') == candidate_id]
    if len(matches) != 1:
        raise ValueError(f'候选ID {candidate_id!r}不存在或不唯一')
    plan = matches[0]
    if plan.get('status') != TARGET_STATUS:
        raise ValueError(
            f'候选 {candidate_id!r}状态为{plan.get("status")}，'
            '需要已确认交界、当前平面和完整三维路径'
        )
    return plan


def build_motion_spec(recognition, candidate_id=None, *, frame_id='base_link',
                      tip_link='tcp_link', expected_parent_frame='link6',
                      allow_nominal_urdf_tool=False, binding_file=None):
    """输出预抓取、夹持和抬升三个MoveIt位姿，并保留执行阻断原因。"""
    if frame_id != 'base_link' or expected_parent_frame != 'link6':
        raise ValueError('视觉目标固定为 base_link 下的 link6 法兰位姿')
    plan = select_grasp_plan(recognition, candidate_id)
    if plan.get('status') != TARGET_STATUS:
        raise ValueError(
            f'候选 {plan.get("candidate_id")!r}状态为{plan.get("status")}，'
            '需要已确认交界、当前平面和完整三维路径'
        )
    policy = recognition.get('grasp_policy')
    if not isinstance(policy, dict):
        raise ValueError('recognition.json缺少grasp_policy')
    if plan.get('boundary_confirmed') is not True:
        raise ValueError('抓取目标缺少人工确认的金属/绝缘交界')
    if plan.get('surface_projection_only') is not False:
        raise ValueError('抓取目标仍是支撑面投影，不能用于抓取规划')
    if plan.get('tcp_compensation_applied') is not True:
        raise ValueError('抓取目标没有应用唯一一次工具偏置补偿')
    if int(plan.get('offset_mm', 0)) <= 0:
        raise ValueError('抓取中心后退距离必须为正数毫米')
    if policy.get('support_plane') is None:
        raise ValueError('抓取目标缺少当前画面的米制支撑平面')
    center_height = policy.get('center_height_over_board_m', plan.get('center_height_over_board_m'))
    if not isinstance(center_height, (int, float)) or not math.isfinite(center_height):
        raise ValueError('抓取目标缺少有限线材中心高度')

    grasp = rigid_transform(plan.get('T_base_flange_grasp'), 'T_base_flange_grasp')
    pregrasp = rigid_transform(plan.get('T_base_flange_pregrasp'), 'T_base_flange_pregrasp')
    base_grasp = rigid_transform(plan.get('T_base_grasp'), 'T_base_grasp')
    if np.linalg.norm(base_grasp[:3, 3] - np.asarray(
            _finite_vector(plan.get('grasp_base_m'), 3, 'grasp_base_m'))) > 1e-4:
        raise ValueError('grasp_base_m与T_base_grasp平移不一致')
    if _rotation_angle(grasp[:3, :3], pregrasp[:3, :3]) > 1e-4:
        raise ValueError('预抓取与夹持姿态不一致，不能形成直线接近段')
    approach = grasp[:3, 2]
    approach /= np.linalg.norm(approach)
    separation = grasp[:3, 3] - pregrasp[:3, 3]
    along = float(separation @ approach)
    lateral = float(np.linalg.norm(separation - approach * along))
    if along <= 0.005 or along > 0.2 or lateral > 1e-4:
        raise ValueError('预抓取必须沿夹持接近轴位于夹持点后方5-200mm')

    blockers = list(plan.get('readiness_blockers') or [])
    blockers.extend(policy.get('readiness_blockers') or [])
    blockers = list(dict.fromkeys(str(item) for item in blockers))
    # 名义工具默认只规划；用户显式授权后才可按URDF模型参数做真机闭合。
    # 新视觉证据缺失不阻止历史数据只规划分析，但必须阻断真实执行。
    hard_plan_blockers = [item for item in blockers if item != NOMINAL_TOOL_BLOCKER
                          and not item.startswith('visual_path_')]
    if hard_plan_blockers:
        raise ValueError('视觉目标仍有规划阻断：' + ', '.join(hard_plan_blockers))
    nominal_tool_authorized = NOMINAL_TOOL_BLOCKER in blockers and allow_nominal_urdf_tool
    execution_blockers = [
        item for item in blockers
        if item != NOMINAL_TOOL_BLOCKER or not nominal_tool_authorized
    ]
    visual_review = execution_path_review(recognition, plan)
    from arm.source_validation import execution_source_review
    source_review = execution_source_review(recognition, plan, binding_file,
                                            allow_nominal_urdf_tool=allow_nominal_urdf_tool)
    angle_review = closing_angle_review(plan)
    execution_blockers = list(dict.fromkeys(execution_blockers+visual_review['rejection_reasons']
                                           +source_review['rejection_reasons']))
    if source_review['passed'] and not angle_review['passed']:
        execution_blockers.append('grasp_closing_angle_outside_85_90_or_unavailable')
    if source_review.get('nominal_tool'):
        nominal_tool_authorized = allow_nominal_urdf_tool
        if not allow_nominal_urdf_tool and NOMINAL_TOOL_BLOCKER not in execution_blockers:
            execution_blockers.append(NOMINAL_TOOL_BLOCKER)

    handeye = policy.get('handeye') or {}
    warnings = []
    if NOMINAL_TOOL_BLOCKER in blockers or source_review.get('nominal_tool'):
        warnings.append(NOMINAL_TOOL_BLOCKER)
    if nominal_tool_authorized:
        warnings.append('nominal_urdf_tool_execution_authorized_by_user')
    if handeye.get('absolute_accuracy_verified') is not True:
        warnings.append('handeye_absolute_accuracy_not_independently_verified')
    camera_serial = handeye.get('camera_serial')
    if not camera_serial:
        raise ValueError('抓取目标缺少手眼相机序列号')

    transform_map = {
        'pregrasp': pregrasp,
        'grasp': grasp,
        'lift': pregrasp,
    }
    return {
        'schema_version': 1,
        'status': 'moveit_ready' if not execution_blockers else 'plan_only',
        'candidate_id': plan['candidate_id'],
        'capture_snapshot': policy.get('capture_snapshot'),
        'camera_serial': camera_serial,
        'frame_id': frame_id,
        'tip_link': tip_link,
        'expected_parent_frame': expected_parent_frame,
        'parent_to_tip_max_position_error_m': 0.0005,
        'parent_to_tip_max_rotation_error_rad': math.radians(0.1),
        'stages': [
            {
                'name': name,
                'pose': transform_to_pose(transform, f'{name}位姿'),
                'T_base_flange': transform.tolist(),
            }
            for name, transform in transform_map.items()
        ],
        'approach_direction_base': approach.tolist(),
        'pregrasp_clearance_m': along,
        'grasp_center_base_m': base_grasp[:3, 3].tolist(),
        'grasp_closing_direction_base': base_grasp[:3, 0].tolist(),
        'gripper': {
            'joint_name': 'gripper',
            'open_width_m': 0.1,
            'close_width_m': 0.0,
            'force_n': 1.0,
        },
        'planning_allowed': True,
        'execution_allowed': not execution_blockers,
        'execution_blockers': execution_blockers,
        'visual_path_review': visual_review,
        'source_review': source_review,
        'closing_angle_review': angle_review,
        'warnings': warnings,
        'nominal_urdf_tool_execution_authorized': nominal_tool_authorized,
        'readiness_note': (
            'MoveIt可用此位姿规划；仅允许名义预览'
            if execution_blockers else (
                '用户已授权按URDF名义工具参数执行；该参数不是实测TCP'
                if nominal_tool_authorized else
                '位姿、工具和执行前置条件已通过数据校验'
            )
        ),
    }


def write_json_exclusive(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
    return target
