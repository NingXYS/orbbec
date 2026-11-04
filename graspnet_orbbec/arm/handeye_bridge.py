"""将已配对快照中的线材候选转换到基座；始终输出预览，不发送运动。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from arm.wire_geometry import (board_plane, flange_transform, grasp_transforms,
                               pixels_on_plane, rigid_transform)
from common.paths import ROOT
from segmentation.grasp_policy import apply_grasp_policy


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def save_fixed_base_plane(bundle, snapshot, out, *, base_z_offset_m=0.0):
    """用一次性棋盘快照保存base_link修正系下的支撑平面。"""
    snapshot = Path(snapshot).resolve(strict=True)
    if not np.isfinite(base_z_offset_m) or abs(base_z_offset_m) > .05:
        raise ValueError('底座Z修正只接受±50mm内的有限米数')
    bundle.check_camera(read_json(snapshot/'capture/camera.json'))
    image = cv2.imread(str(snapshot/'capture/rgb.png'))
    if image is None or image.shape[1::-1] != bundle.size:
        raise ValueError('平面快照缺少匹配标定分辨率的RGB图像')
    pair = read_json(snapshot/'pair.json')
    if (pair.get('status') != 'paired_stationary'
            or pair.get('calibration_sha256') != bundle.settings['sha256']
            or pair.get('image_sha256') != sha256(snapshot/'capture/rgb.png')):
        raise ValueError('平面快照配对或标定指纹不匹配')
    t_base_flange = rigid_transform(pair['T_base_flange'])
    correction = np.eye(4)
    correction[2, 3] = float(base_z_offset_m)
    t_base_rgb = correction @ t_base_flange @ bundle.t_flange_rgb
    plane_camera = board_plane(image, bundle.k, bundle.d, bundle.session['board'])
    normal_camera = np.asarray(plane_camera['normal_camera'], float)
    origin_camera = np.asarray(plane_camera['origin_camera_m'], float)
    normal_base = t_base_rgb[:3, :3] @ normal_camera
    normal_base /= np.linalg.norm(normal_base)
    origin_base = t_base_rgb[:3, :3] @ origin_camera + t_base_rgb[:3, 3]
    payload = {
        'schema_version': 1,
        'frame_id': 'base_link',
        'base_z_offset_m': float(base_z_offset_m),
        'camera_serial': bundle.camera['serial'],
        'handeye_sha256': bundle.settings['sha256'],
        'normal_base': normal_base.tolist(),
        'offset_base_m': float(normal_base @ origin_base),
        'source_snapshot': str(snapshot),
        'source_image_sha256': sha256(snapshot/'capture/rgb.png'),
        'source_pair_base_flange': t_base_flange.tolist(),
        'camera_plane': plane_camera,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'operator_contract': 'table_and_robot_base_must_not_move',
    }
    target = Path(out)
    if target.exists():
        raise FileExistsError(f'固定平面文件已存在：{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def fixed_plane_in_camera(path, bundle, t_base_flange, *, base_z_offset_m):
    settings = read_json(path)
    correction = np.eye(4)
    correction[2, 3] = float(base_z_offset_m)
    if (settings.get('schema_version') != 1
            or settings.get('frame_id') != 'base_link'
            or settings.get('camera_serial') != bundle.camera['serial']
            or settings.get('handeye_sha256') != bundle.settings['sha256']
            or not np.isclose(float(settings.get('base_z_offset_m')), base_z_offset_m, atol=1e-9)):
        raise ValueError('固定平面需要同一相机、手眼标定和底座Z修正')
    normal_base = np.asarray(settings['normal_base'], float)
    offset_base = float(settings['offset_base_m'])
    if (normal_base.shape != (3,) or not np.isfinite(normal_base).all()
            or abs(np.linalg.norm(normal_base) - 1) > 1e-6 or not np.isfinite(offset_base)):
        raise ValueError('固定平面参数无效')
    t_base_rgb = correction @ t_base_flange @ bundle.t_flange_rgb
    t_rgb_base = np.linalg.inv(t_base_rgb)
    normal_camera = t_rgb_base[:3, :3] @ normal_base
    normal_camera /= np.linalg.norm(normal_camera)
    point_base = normal_base * offset_base
    origin_camera = t_rgb_base[:3, :3] @ point_base + t_rgb_base[:3, 3]
    return {
        'normal_camera': normal_camera.tolist(),
        'origin_camera_m': origin_camera.tolist(),
        'source': 'fixed_base_plane',
        'source_snapshot': settings.get('source_snapshot'),
        'base_z_offset_m': base_z_offset_m,
    }


class HandeyeBundle:
    def __init__(self, config_path):
        self.path = Path(config_path).resolve()
        self.settings = read_json(self.path)
        root = Path(self.settings['handeye_root'])
        self.root = (root if root.is_absolute() else ROOT / root).resolve(strict=True)
        self.report = self.root / self.settings['calibration_report']
        files = {'result': self.report/'result.json', 'camera': self.report/'camera_used.json',
                 'session': self.report/'config_used.json'}
        for name, path in files.items():
            if sha256(path) != self.settings['sha256'][name]:
                raise ValueError(f'冻结标定文件改变：{name}，请重新审计绑定')
        self.session, self.camera = read_json(files['session']), read_json(files['camera'])
        self.result = read_json(files['result'])
        robot = self.session['robot']
        expected = {'translation_unit': 'm', 'angle_unit': 'rad', 'representation': 'euler',
                    'reference': 'base_flange', 'euler_convention': 'extrinsic_xyz_RzRyRx'}
        if any(robot.get(key) != value for key, value in expected.items()):
            raise ValueError('不支持的机器人反馈约定，不能猜测法兰/TCP或角度单位')
        if (self.session.get('calibration_mode') != 'eye_in_hand'
                or self.camera.get('geometry_verified') is not True
                or self.camera.get('geometry_scope') != 'rgb_only'
                or self.camera['serial'] != self.settings['camera_serial']
                or self.session['camera']['serial'] != self.camera['serial']):
            raise ValueError('需要同一设备的已核实原始RGB眼在手上标定')
        self.method = self.settings['method']
        self.solution = self.result['methods'][self.method]
        self.t_flange_rgb = rigid_transform(self.solution['T_gripper_rgb'])
        rgb = self.camera['rgb']
        self.k, self.d = np.asarray(rgb['K'], float), np.asarray(rgb['D'], float)
        self.size = (rgb['width'], rgb['height'])
        if rgb['image_mode'] != 'raw_distorted' or rgb['model'] != 'brown_conrady':
            raise ValueError('仅支持原始Brown畸变RGB，不接受D2C内参替换')

    def check_camera(self, actual):
        if actual.get('serial') != self.camera['serial']:
            raise ValueError('实际快照相机与手眼标定序列号不一致')
        rgb = actual['rgb']
        if (rgb['width'], rgb['height']) != self.size or rgb['image_mode'] != 'raw_distorted':
            raise ValueError('实际图像分辨率或模式改变')
        if (np.max(np.abs(np.asarray(rgb['K'])-self.k)) > 1e-6
                or np.max(np.abs(np.asarray(rgb['D'])-self.d)) > 1e-6):
            raise ValueError('实际RGB内参与冻结标定不一致')
        processing = actual.get('capture_pixel_processing', {})
        if (actual.get('pipeline_parameter_image_is_mirrored') != 0
                or actual.get('pipeline_rgb_intrinsics_match_stream') != 1
                or processing.get('software_crop_resize_mirror_flip_undistort') != 'none'):
            raise ValueError('RGB裁剪/镜像/去畸变合同未经核实')

    def info(self):
        val = self.solution['validation']
        return {'camera_serial': self.camera['serial'], 'method': self.method,
                'T_flange_rgb': self.t_flange_rgb.tolist(),
                'validation_rms_mm': val['translation_rms_m']*1000,
                'validation_max_mm': val['translation_max_m']*1000,
                'absolute_accuracy_verified': False, 'depth_positioning_enabled': False,
                'motion_commands_enabled': False, 'sha256': self.settings['sha256']}


class WireHandeyeBridge:
    def __init__(self, bundle, snapshot, *, center_height_m=None,
                 support_relation_confirmed=False, installation_confirmed=False, tool_file=None,
                 base_z_offset_m=0.0, fixed_plane_file=None):
        self.bundle, self.snapshot = bundle, Path(snapshot).resolve(strict=True)
        self.image = cv2.imread(str(self.snapshot/'capture/rgb.png'))
        if self.image is None or self.image.shape[1::-1] != bundle.size:
            raise ValueError('快照缺少匹配标定分辨率的RGB图像')
        bundle.check_camera(read_json(self.snapshot/'capture/camera.json'))
        self.pair = read_json(self.snapshot/'pair.json')
        if (self.pair.get('status') != 'paired_stationary'
                or self.pair.get('calibration_sha256') != bundle.settings['sha256']
                or self.pair.get('image_sha256') != sha256(self.snapshot/'capture/rgb.png')):
            raise ValueError('快照配对或标定指纹不匹配，不能跨帧沿用位姿')
        # 重验原始反馈与图像覆盖，不把手写T_base_flange当作实时反馈证据。
        from arm.stationary_capture import verify_snapshot
        verified = verify_snapshot(bundle, self.snapshot)
        self.t_base_flange = flange_transform(verified['robot_pose'])
        if not np.allclose(self.t_base_flange, self.pair['T_base_flange'], atol=1e-10):
            raise ValueError('快照位姿与原始反馈不一致')
        if center_height_m is not None and (not np.isfinite(center_height_m) or abs(center_height_m) > .05):
            raise ValueError('平面测试仅接受相对棋盘±50mm内的已测线材中心高度')
        if not np.isfinite(base_z_offset_m) or abs(base_z_offset_m) > .05:
            raise ValueError('底座Z修正只接受±50mm内的有限米数')
        self.height = center_height_m
        self.base_z_offset_m = float(base_z_offset_m)
        self.support_confirmed = support_relation_confirmed
        self.installation_confirmed = installation_confirmed
        self.tool = read_json(tool_file) if tool_file else None
        self.plane_error = None
        if fixed_plane_file is not None:
            self.plane = fixed_plane_in_camera(
                fixed_plane_file, bundle, self.t_base_flange,
                base_z_offset_m=self.base_z_offset_m,
            )
        else:
            try:
                self.plane = board_plane(self.image, bundle.k, bundle.d, bundle.session['board'])
            except ValueError as exc:
                self.plane, self.plane_error = None, str(exc)

    def apply(self, result, cfg):
        if image_digest(result.image_bgr) != image_digest(self.image):
            raise ValueError('识别原图已改变，禁止跨快照使用手眼位姿')
        boundaries = {p['candidate_id']: p['boundary_input_xy'] for p in result.grasp_plans
                      if p.get('boundary_confirmed')}
        metrics = {}
        for candidate in result.candidates:
            path = candidate.get('inward_path_xy', [])
            if (self.plane is not None and len(path) >= 2 and candidate['id'] in boundaries
                    and not candidate.get('path_ambiguous')):
                metrics[candidate['id']] = {
                    'xyz_m': pixels_on_plane(path, self.bundle.k, self.bundle.d, self.plane,
                                              self.height if self.height is not None else 0),
                    'plane_normal': self.plane['normal_camera']}
        apply_grasp_policy(result, cfg, boundaries, metrics)
        blockers = []
        if self.plane is None:
            blockers.append('current_board_plane_unavailable')
        if not self.installation_confirmed:
            blockers.append('same_robot_and_camera_mount_unconfirmed')
        if not self.support_confirmed or self.height is None:
            blockers.append('measured_wire_center_height_and_support_relation_required')
        tool_transform = None
        if self.tool is None:
            blockers.append('measured_grasp_center_and_tool_axes_required')
        else:
            nominal = self.tool.get('verification') == 'official_model_nominal_preview_only'
            if ((self.tool.get('verified') is not True and not nominal) or not self.tool.get('evidence')
                    or self.tool.get('frame_definition') != 'x_closing_y_cross_z_approach'
                    or self.tool.get('camera_serial') != self.bundle.camera['serial']):
                raise ValueError('工具文件需要实测依据、设备绑定及夹持坐标轴定义')
            if nominal:
                blockers.append('nominal_finger_tip_reference_not_verified_grasp_center')
            tool_transform = rigid_transform(self.tool['T_flange_grasp'])
        base_z_offset_m = getattr(self, 'base_z_offset_m', 0.0)
        for plan in result.grasp_plans:
            if plan['status'] != 'camera_frame_preview_ready':
                continue
            base_correction = np.eye(4)
            base_correction[2, 3] = base_z_offset_m
            transformed = grasp_transforms(plan, base_correction @ self.t_base_flange @ self.bundle.t_flange_rgb,
                                            tool_transform)
            plan.update(transformed, status='base_frame_preview_ready',
                        metric_model='wire_center_parallel_to_current_board_plane',
                        center_height_over_board_m=self.height,
                        base_z_offset_m=base_z_offset_m,
                        surface_projection_only=self.height is None or not self.support_confirmed,
                        readiness_blockers=list(blockers), robot_pose_valid=False,
                        executable=False)
        result.grasp_policy.update(
            metric_source='current_rgb_board_plane_with_measured_height',
            robot_pose_valid=False, motion_commands_enabled=False,
            handeye=self.bundle.info(), support_plane=self.plane,
            support_plane_error=self.plane_error,
            base_z_offset_m=base_z_offset_m,
            capture_snapshot=str(self.snapshot), readiness_blockers=blockers,
            interpretation='冻结快照的基座坐标预览；未检查碰撞/可达性，不能直接下发机器人')
        return result
