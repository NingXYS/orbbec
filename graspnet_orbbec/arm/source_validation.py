"""执行边界重读原始快照及冻结标定；不访问相机、CAN或ROS。"""

from pathlib import Path

import cv2
import numpy as np

from common.paths import ROOT
from arm.handeye_bridge import HandeyeBundle, WireHandeyeBridge, image_digest, read_json, sha256
from arm.wire_geometry import grasp_transforms


SNAPSHOT_FILES = ('capture/rgb.png', 'capture/depth.png', 'capture/camera.json',
                  'capture/frames.json', 'capture/COMPLETE', 'robot.json', 'pair.json')


def snapshot_provenance(bridge):
    """只为真实构造的Bridge记录来源；旧的无来源记录继续仅供分析。"""
    if not hasattr(bridge, 'source_files'):
        return None
    return {'schema_version': 1, 'snapshot': str(bridge.snapshot),
            'files': dict(bridge.source_files), 'tool': bridge.source_tool,
            'fixed_plane': bridge.source_plane,
            'installation_confirmed': bridge.installation_confirmed,
            'support_relation_confirmed': bridge.support_confirmed,
            'center_height_m': bridge.height}


def execution_source_review(recognition, plan, binding_file=None, *, allow_nominal_urdf_tool=False):
    """不接受JSON自证的图片hash/坐标；绑定路径由调用入口指定，不从结果中选择。"""
    policy = recognition.get('grasp_policy') or {}
    source = policy.get('source_provenance')
    if not isinstance(source, dict) or source.get('schema_version') != 1:
        return {'passed': False, 'rejection_reasons': ['execution_source_missing']}
    nominal = False
    try:
        bundle = HandeyeBundle(binding_file or ROOT/'configs/wire_handeye_037K.json')
        snapshot = Path(source['snapshot']).resolve(strict=True)
        if Path(policy['capture_snapshot']).resolve(strict=True) != snapshot:
            raise ValueError('快照来源不一致')
        if set(source['files']) != set(SNAPSHOT_FILES):
            raise ValueError('快照来源文件清单不完整')
        for name in SNAPSHOT_FILES:
            if sha256(snapshot/name) != source['files'][name]:
                raise ValueError('快照文件改变：'+name)
        pair = read_json(snapshot/'pair.json')
        if pair.get('data_origin') != 'physical_camera' or policy.get('data_origin', 'physical_camera') != 'physical_camera':
            raise ValueError('缺少真实采集来源；模拟或旧版未声明来源仅可分析')
        if (pair.get('capture_schema_version') != 2
                or pair.get('capture_kind') not in ('single_stationary_snapshot', 'stationary_sequence_frame')):
            raise ValueError('旧版快照缺少采集类型合同，需重新采集并确认')
        # 子快照不能脱离父序列失败状态使用，也不能借单帧兼容绕过全程静止检查。
        parent = snapshot.parent.parent/'manifest.json'
        if pair['capture_kind'] == 'stationary_sequence_frame' or pair.get('sequence_manifest') or (snapshot.parent.name == 'frames' and parent.exists()):
            from arm.stationary_sequence import verify_stationary_sequence
            if pair.get('sequence_manifest') != '../../manifest.json':
                raise ValueError('序列子快照缺少明确父manifest绑定')
            verify_stationary_sequence(bundle, parent.parent)
            entries = read_json(parent)['frames']
            if sum((parent.parent/e['snapshot']).resolve() == snapshot for e in entries) != 1:
                raise ValueError('快照未唯一归属有效序列')
        if (source.get('installation_confirmed') is not True or source.get('support_relation_confirmed') is not True
                or source.get('center_height_m') != policy.get('center_height_over_board_m')):
            raise ValueError('安装、实测高度或支撑关系未确认')

        def file_contract(value, required=False):
            if value is None and not required:
                return None
            path = Path(value['path']).resolve(strict=True)
            if sha256(path) != value['sha256']:
                raise ValueError('工具或固定平面文件改变')
            return path

        tool = file_contract(source['tool'], required=True)
        plane = file_contract(source['fixed_plane'])
        bridge = WireHandeyeBridge(bundle, snapshot, center_height_m=source['center_height_m'],
                                   support_relation_confirmed=True, installation_confirmed=True,
                                   tool_file=tool, fixed_plane_file=plane,
                                   base_z_offset_m=policy.get('base_z_offset_m', 0.))
        pixels = image_digest(bridge.image)
        if (pixels != policy.get('image_rgb_sha256')
                or pixels != plan['visual_path_validation']['evidence']['image_rgb_sha256']):
            raise ValueError('像素指纹未匹配实际原图')
        handeye = policy['handeye']
        if (handeye['sha256'] != bundle.settings['sha256'] or handeye['camera_serial'] != bundle.camera['serial']
                or handeye['method'] != bundle.method
                or not np.allclose(handeye['T_flange_rgb'], bundle.t_flange_rgb, atol=1e-9, rtol=0)
                or handeye.get('absolute_accuracy_verified') != bundle.info()['absolute_accuracy_verified']):
            raise ValueError('标定绑定与当前冻结合同不一致')
        if bridge.plane is None:
            raise ValueError('实际快照缺少定位平面')
        for key in ('normal_camera', 'origin_camera_m'):
            if not np.allclose(bridge.plane[key], policy['support_plane'][key], atol=1e-9, rtol=0):
                raise ValueError('支撑平面来源不一致')
        tool_data = bridge.tool
        nominal = tool_data.get('verification') == 'official_model_nominal_preview_only'
        if nominal and not allow_nominal_urdf_tool:
            raise ValueError('名义工具未获得本次入口显式授权')
        if (not tool_data.get('evidence') or tool_data.get('camera_serial') != bundle.camera['serial']
                or tool_data.get('frame_definition') != 'x_closing_y_cross_z_approach'
                or not (tool_data.get('verified') is True
                        or tool_data.get('verification') == 'official_model_nominal_preview_only')):
            raise ValueError('工具来源无有效实测或显式名义合同')
        correction = np.eye(4)
        correction[2, 3] = bridge.base_z_offset_m
        transform = correction @ bridge.t_base_flange @ bundle.t_flange_rgb
        coordinates = plan['coordinate_verification']
        for key, actual in [('K', bundle.k), ('D', bundle.d), ('T_base_rgb', transform),
                            ('T_flange_grasp', tool_data['T_flange_grasp'])]:
            if not np.allclose(coordinates[key], actual, atol=1e-9, rtol=0):
                raise ValueError('真实来源与坐标合同不一致：'+key)
        expected = grasp_transforms(plan, transform, tool_data['T_flange_grasp'])
        for key in ('T_base_grasp', 'T_base_flange_grasp', 'T_base_flange_pregrasp'):
            if not np.allclose(expected[key], plan[key], atol=1e-8, rtol=0):
                raise ValueError('目标未由本快照真实变换生成')
        return {'passed': True, 'rejection_reasons': [], 'snapshot': str(snapshot),
                'image_rgb_sha256': pixels, 'binding': str(bundle.path), 'nominal_tool': nominal}
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, cv2.error) as exc:
        return {'passed': False, 'rejection_reasons': ['execution_source_invalid'], 'detail': str(exc),
                'nominal_tool': nominal}
