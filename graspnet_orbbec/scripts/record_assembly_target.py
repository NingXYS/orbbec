#!/usr/bin/env python3
"""被动读取两次CAN反馈并重新记录固定装配目标；不使能、不发送控制帧。"""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.read_wire_robot_state import load_binding, receive_state
from arm.assembly_target import JOINT_NAMES, load_assembly_target
from arm.moveit_wire_adapter import matrix_to_quaternion, rigid_transform
from arm.wire_geometry import flange_transform


def build_record(template, samples, *, root, report_dir):
    """目标更新必须成套重算；不保留旧目标的规划或退出示教结论。"""
    matrices, joints = [], []
    for sample in samples:
        q = np.asarray(sample['joint_angles_rad'], float)
        if (sample['application_frames_sent'] != 0 or sample['complete_pose_count'] < 10
                or q.shape != (6,) or not np.isfinite(q).all()):
            raise ValueError('缺少可靠的被动六轴反馈')
        matrices.append(flange_transform(sample['flange_pose_m_rad']))
        joints.append(q)
    if len(samples) != 2:
        raise ValueError('需要两次独立采样')
    angle = lambda a, b: float(np.arccos(np.clip((np.trace(a[:3,:3] @ b[:3,:3].T)-1)/2,-1,1)))
    deltas = {'position_m': float(np.linalg.norm(matrices[0][:3,3]-matrices[1][:3,3])),
              'orientation_rad': angle(*matrices),
              'max_joint_rad': float(np.max(np.abs(joints[0]-joints[1])))}
    if deltas['position_m'] > .0005 or deltas['orientation_rad'] > np.deg2rad(.1) or deltas['max_joint_rad'] > np.deg2rad(.1):
        raise ValueError(f'两次采样位姿不一致，请停稳后重新记录：{deltas}')
    current = samples[-1]
    if template['joint_names'] != JOINT_NAMES or template['reference_frame'] != 'base_link':
        raise ValueError('模板不是官方六轴base_link合同')
    model = ET.parse(template['planning_contract']['limits_source']).getroot()
    limits = {j.attrib['name']: j.find('limit') for j in model.findall('joint')}
    for name, value in zip(JOINT_NAMES, joints[-1]):
        limit = limits.get(name)
        if limit is None or not float(limit.attrib['lower']) <= value <= float(limit.attrib['upper']):
            raise ValueError(f'{name}超出官方模型限位，目标未更新')
    tool_path = root/template['tool']['file']
    tool_raw = tool_path.read_bytes()
    if hashlib.sha256(tool_raw).hexdigest() != template['tool']['sha256']:
        raise ValueError('工具配置已变化，不能沿用旧TCP合同')
    tool = rigid_transform(json.loads(tool_raw)['T_flange_grasp'])
    if not np.allclose(tool, template['tool']['T_flange_grasp'], atol=1e-9, rtol=0):
        raise ValueError('模板工具变换不一致')
    def pose(matrix):
        return {'position_m': matrix[:3,3].tolist(), 'position_mm': (matrix[:3,3]*1000).tolist(),
                'quaternion_xyzw': matrix_to_quaternion(matrix[:3,:3]).tolist(),
                'rpy_rad': current['flange_pose_m_rad'][3:],
                'rpy_deg': np.degrees(current['flange_pose_m_rad'][3:]).tolist(),
                'T_base_frame': matrix.tolist()}
    result = copy.deepcopy(template)
    stamp = datetime.fromtimestamp(current['observed_wall_time_s'], ZoneInfo('Asia/Taipei'))
    result.update(target_id='wire_assembly_fixed_'+stamp.strftime('%Y%m%d_%H%M%S_%f'),
                  recorded_at=stamp.isoformat(), status='integrated_pending_physical_transfer_test',
                  joint_positions_rad=joints[-1].tolist(), joint_positions_deg=np.degrees(joints[-1]).tolist(),
                  flange_goal=pose(matrices[-1]), grasp_tcp_goal=pose(matrices[-1] @ tool))
    for key in ('latest_control_state_check', 'latest_planning_validation', 'latest_pose_comparison', 'integration_validation'):
        result.pop(key, None)
    result['integration_validation'] = {'status': 'pending_for_updated_target'}
    result['observation'] = {'source': 'passive_socketcan_official_sdk_decoder',
                             'application_frames_sent': 0, 'sdk_session_connected': False,
                             'initial_report': str(report_dir/'state_01.json'),
                             'verification_report': str(report_dir/'state_02.json'),
                             'initial_sha256': hashlib.sha256((report_dir/'state_01.json').read_bytes()).hexdigest(),
                             'verification_sha256': hashlib.sha256((report_dir/'state_02.json').read_bytes()).hexdigest(),
                             'sample_deltas': deltas, 'continuous_stability_validated': False,
                             'ctrl_mode': current['arm_status']['ctrl_mode'],
                             'teach_status': current['arm_status']['teach_status']}
    contract = result['planning_contract']
    contract['execution_allowed'] = False
    contract['joint_limit_violations'] = []
    contract['blockers'] = ['postgrasp_loaded_transfer_not_validated', 'physical_obstacles_and_camera_mount_not_in_planning_scene']
    if current['arm_status']['ctrl_mode'] not in (0,1) or current['arm_status']['teach_status'] == 1:
        contract['blockers'].insert(0, 'operator_must_exit_teaching_mode')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=ROOT/'configs/wire_assembly_fixed_20261106.json')
    parser.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    original = target.read_bytes()
    template = json.loads(original)
    load_assembly_target(target, root=ROOT, frame_id='base_link', joints=JOINT_NAMES)
    stamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S_%f')
    report_dir = ROOT/'output/wire_grasp'/('assembly_record_'+stamp)
    report_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    binding = load_binding(args.binding)
    print('保持机械臂静止，正在被动采样两次（约6秒），不发送运动。', flush=True)
    for index in (1,2):
        sample = receive_state(binding)
        with (report_dir/f'state_{index:02d}.json').open('x') as out:
            json.dump(sample, out, ensure_ascii=False, indent=2)
        samples.append(sample)
    updated = build_record(template, samples, root=ROOT, report_dir=report_dir)
    backup = target.with_name(target.stem+'_backup_'+stamp+'.json')
    updated['supersedes'] = str(backup)
    # 保存前复查并原子替换，避免采样期间别人修改配置后被静默覆盖。
    if target.read_bytes() != original:
        raise ValueError('采样期间目标文件已改变，拒绝覆盖')
    with backup.open('xb') as out:
        out.write(original)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, suffix='.json', delete=False) as out:
        pending = Path(out.name)
        json.dump(updated, out, ensure_ascii=False, indent=2)
        out.write('\n')
    try:
        load_assembly_target(pending, root=ROOT, frame_id='base_link', joints=JOINT_NAMES)
        if target.read_bytes() != original:
            raise ValueError('保存前目标文件已改变，拒绝覆盖')
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)
    with (report_dir/'target_record.json').open('x') as out:
        json.dump(updated, out, ensure_ascii=False, indent=2)
    print(f'已更新：{target}\n旧目标备份：{backup}\n采样报告：{report_dir}')
    print('夹持TCP XYZ(mm)：', updated['grasp_tcp_goal']['position_mm'])
    print('RPY(deg)：', updated['flange_goal']['rpy_deg'])
    print('执行前请退出示教，并使用本次新识别结果先只规划。')


if __name__ == '__main__':
    main()
