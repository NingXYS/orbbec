"""复用标定工程的原始RGB/被动CAN采样，独立保存抓取快照。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from arm.handeye_bridge import read_json, sha256
from arm.wire_geometry import flange_transform


def helpers(bundle):
    # 复用已验收的20ms覆盖和停稳门槛，不在抓取链路复制一套较松的采样规则。
    calibration_dir = str(bundle.root/'eyeinhand/eyeinhand')
    if calibration_dir not in sys.path:
        sys.path.insert(0, calibration_dir)
    spec = importlib.util.spec_from_file_location('_wire_stationary_helpers',
                                                 bundle.root/'tools/collect_stationary_sample.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pair_rgb_window(module, robot, frames, convention):
    """完整覆盖不变，仅用RGB接收时刻附近窗口判断图像位姿是否稳定。"""
    samples = robot.get('samples', [])
    if robot.get('status') != 'passed' or len(samples) < 10:
        raise ValueError('缺少有效完整机器人反馈')
    stamps = np.array([s['host_kernel_timestamp_s'] for s in samples], dtype=float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError('机器人接收时间必须有限且严格递增')
    streams = {}
    for name in ('rgb', 'depth'):
        stamp = float(frames[name]['host_receive_timestamp_us']) * 1e-6
        before, after = np.flatnonzero(stamps <= stamp), np.flatnonzero(stamps >= stamp)
        if not len(before) or not len(after):
            raise ValueError('图像接收时刻未被完整位姿前后覆盖')
        i, j = int(before[-1]), int(after[0])
        gaps = [(stamp - stamps[i]) * 1000, (stamps[j] - stamp) * 1000]
        if max(gaps) > 20:
            raise ValueError('图像前后位姿接收间隔超过20ms')
        streams[name] = {
            'host_receive_s': stamp, 'before_index': i, 'after_index': j,
            'before_gap_ms': gaps[0], 'after_gap_ms': gaps[1],
            'nearest_index': int(np.argmin(np.abs(stamps - stamp))),
        }
    image_times = [item['host_receive_s'] for item in streams.values()]
    if stamps[0] > min(image_times) - 0.5 or stamps[-1] < max(image_times) + 0.5:
        raise ValueError('未覆盖图像接收前后各0.5秒')
    if np.max(np.diff(stamps)) > 0.020:
        raise ValueError('监听期间完整位姿存在超过20ms的缺口')
    rgb_time = streams['rgb']['host_receive_s']
    local_indices = np.flatnonzero(np.abs(stamps - rgb_time) <= 0.5)
    transforms = [module.pose_to_base_gripper(samples[i]['pose_m_rad'], convention)
                  for i in local_indices]
    if len(transforms) < 10:
        raise ValueError('RGB接收窗口附近反馈不足')
    reference = transforms[len(transforms) // 2]
    translation = max(float(np.linalg.norm(t[:3, 3] - reference[:3, 3])) for t in transforms)
    rotation = max(module.rotation_error_deg(reference[:3, :3], t[:3, :3]) for t in transforms)
    if translation > 0.0005 or rotation > 0.1:
        raise ValueError(f'RGB接收窗口反馈未稳定：最大变化{translation*1000:.3f}mm / {rotation:.3f}deg')
    all_transforms = [module.pose_to_base_gripper(s['pose_m_rad'], convention) for s in samples]
    whole_translation = max(float(np.linalg.norm(t[:3, 3] - all_transforms[0][:3, 3])) for t in all_transforms)
    whole_rotation = max(module.rotation_error_deg(all_transforms[0][:3, :3], t[:3, :3]) for t in all_transforms)
    chosen = streams['rgb']['nearest_index']
    return {
        'streams': streams, 'robot_pose': list(samples[chosen]['pose_m_rad']),
        'translation_max_change_m': translation, 'rotation_max_change_deg': rotation,
        'stationary_threshold_m': 0.0005, 'stationary_threshold_deg': 0.1,
        'complete_pose_count': len(samples), 'hardware_sync_verified': False,
        'exposure_pose_verified': False,
        'stability_scope': 'rgb_receive_plus_minus_0.5s',
        'whole_window_translation_max_change_m': whole_translation,
        'whole_window_rotation_max_change_deg': whole_rotation,
        'interpretation': '完整反馈覆盖；RGB接收附近稳定；全程抖动只记录，不挪作曝光位姿。',
    }


def verify_snapshot(bundle, output):
    output = Path(output)
    if not (output/'capture/COMPLETE').is_file():
        raise ValueError('相机采集不完整')
    actual = read_json(output/'capture/camera.json')
    bundle.check_camera(actual)
    robot = read_json(output/'robot.json')
    expected = bundle.session['robot']
    if (robot.get('firmware_version') != expected['firmware_version']
            or robot.get('source_sha') != expected['audited_source_sha']
            or robot.get('application_frames_sent_by_tool') != 0
            or robot.get('sdk_session_connected') is not False
            or robot.get('channel') != bundle.settings['channel']):
        raise ValueError('机器人反馈与冻结的固件/SDK/只读合同不一致')
    module = helpers(bundle)
    frames = read_json(output/'capture/frames.json')
    try:
        return module.verify_pair(robot, frames, expected)
    except ValueError as exc:
        if not str(exc).startswith('反馈未稳定：'):
            raise
        return verify_pair_rgb_window(module, robot, frames, expected)


def capture_snapshot(bundle, output):
    module = helpers(bundle)
    channel = bundle.settings['channel']
    module.require_can_interface(channel)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receiver = None
    try:
        with (output/'robot.log').open('w') as robot_log, (output/'camera.log').open('w') as camera_log:
            receiver = subprocess.Popen([
                str(bundle.root/'scripts/robot-python'), str(bundle.root/'tools/check_flange_feedback.py'),
                '--channel', channel, '--firmware-report', str(bundle.root/bundle.settings['firmware_report']),
                '--seconds', '4', '--output', str(output/'robot.json')],
                stdout=robot_log, stderr=subprocess.STDOUT, cwd=output)
            subprocess.run(module.capture_command(bundle.root/'build/handeye_capture', output/'capture',
                                                   bundle.session['camera']),
                           stdout=camera_log, stderr=subprocess.STDOUT, check=True, timeout=25, cwd=output)
            if receiver.wait(timeout=12) != 0:
                raise ValueError('被动反馈检查失败，查看robot.log')
        verified = verify_snapshot(bundle, output)
        pair = {'status': 'paired_stationary', 'calibration_sha256': bundle.settings['sha256'],
                'image_sha256': sha256(output/'capture/rgb.png'),
                'T_base_flange': flange_transform(verified['robot_pose']).tolist(),
                'verification': verified, 'robot_motion_commands_sent': 0,
                'hardware_sync_verified': False, 'exposure_pose_verified': False}
        (output/'pair.json').write_text(json.dumps(pair, ensure_ascii=False, indent=2)+'\n')
        return output
    except BaseException as error:
        (output/'FAILED.json').write_text(json.dumps({'error': str(error),
                                                     'robot_motion_commands_sent': 0}, ensure_ascii=False))
        raise
    finally:
        if receiver is not None and receiver.poll() is None:
            receiver.terminate()
            try:
                receiver.wait(timeout=3)
            except subprocess.TimeoutExpired:
                receiver.kill()
                receiver.wait()
