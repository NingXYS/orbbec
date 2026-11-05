"""独立RGB多帧与连续被动反馈验收；不融合检测、不平均位姿、不发送运动。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

import cv2
import numpy as np

from arm.handeye_bridge import read_json, sha256
from arm.stationary_capture import helpers, verify_snapshot
from arm.wire_geometry import flange_transform


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def merge_feedback_reports(reports):
    if not reports:
        raise ValueError('没有连续反馈报告')
    samples = {}
    first = reports[0]
    keys = ('channel', 'firmware_version', 'source_sha', 'application_frames_sent_by_tool', 'sdk_session_connected')
    for report in reports:
        if report.get('status') != 'passed' or any(report.get(k) != first.get(k) for k in keys):
            raise ValueError('反馈分段状态/来源合同不一致')
        monotonic = np.array([s['arrival_monotonic_s'] for s in report.get('samples', [])], float)
        if not np.isfinite(monotonic).all() or np.any(np.diff(monotonic) <= 0):
            raise ValueError('反馈分段单调时间异常')
        previous = -np.inf
        for sample in report.get('samples', []):
            flange_transform(sample['pose_m_rad'])
            stamp = float(sample['host_kernel_timestamp_s'])
            if not np.isfinite(stamp) or stamp <= previous:
                raise ValueError('反馈分段时间戳不递增')
            previous = stamp
            if stamp in samples:
                if not np.allclose(samples[stamp]['pose_m_rad'], sample['pose_m_rad'], atol=1e-12, rtol=0):
                    raise ValueError('重叠监听窗口的同一时间戳位姿不一致')
            else:
                samples[stamp] = sample
    ordered = [samples[t] for t in sorted(samples)]
    if len(ordered) < 10:
        raise ValueError('完整反馈不足')
    mono = np.array([s['arrival_monotonic_s'] for s in ordered], float)
    if not np.isfinite(mono).all() or np.any(np.diff(mono) <= 0):
        raise ValueError('跨段反馈存在主机时钟跳变或单调时间异常')
    return {**first, 'samples': ordered, 'complete_pose_count': len(ordered),
            'duration_seconds': float(mono[-1]-mono[0]),
            'diagnostics_scope': 'other counters inherited from first segment; consult raw segment reports',
            'feedback_segments': len(reports), 'pose_aggregation': 'none; duplicate kernel timestamps deduplicated'}


class _FeedbackSegments:
    """重叠复用已审计的10秒只读工具，避免新增CAN解码或发送路径。"""
    def __init__(self, bundle, output):
        self.bundle, self.output = bundle, output
        self.stop = threading.Event()
        self.jobs, self.error = [], None
        self.thread = None

    def _launch(self):
        base = self.output/'feedback'/f'segment_{len(self.jobs):03d}'
        base.parent.mkdir(exist_ok=True)
        report = base.with_suffix('.json')
        with base.with_suffix('.log').open('w') as log:
            process = subprocess.Popen([
                str(self.bundle.root/'scripts/robot-python'),
                str(self.bundle.root/'tools/check_flange_feedback.py'),
                '--channel', self.bundle.settings['channel'], '--firmware-report',
                str(self.bundle.root/self.bundle.settings['firmware_report']),
                '--seconds', '10', '--output', str(report)],
                stdout=log, stderr=subprocess.STDOUT, cwd=self.output)
        self.jobs.append((process, report))

    def start(self):
        self._launch()

        def run():
            try:
                while not self.stop.wait(8):
                    self._launch()
            except Exception as exc:
                self.error = exc

        self.thread = threading.Thread(target=run, name='stationary-feedback-segments', daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        if self.error:
            raise self.error
        reports, segments = [], []
        for process, path in self.jobs:
            if process.wait(timeout=12) != 0:
                raise ValueError('连续被动反馈失败，查看feedback分段日志')
            reports.append(read_json(path))
            segments.append({'path': str(path.relative_to(self.output)), 'sha256': sha256(path)})
        return merge_feedback_reports(reports), segments

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
        for process, _ in self.jobs:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _safe_path(root, value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError('manifest路径必须相对序列目录')
    path = (root/value).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest路径越出序列目录')
    return path


def _scene_change(reference, current, roi):
    x1, y1, x2, y2 = roi
    a = cv2.cvtColor(reference[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(a, maxCorners=500, qualityLevel=.03, minDistance=5)
    if corners is None or len(corners) < 8:
        raise ValueError('静止图像验证证据不足：ROI可跟踪特征少于8个')
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, corners, None)
    if moved is None:
        raise ValueError('静止图像特征跟踪失败')
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None)
    if back is None:
        raise ValueError('静止图像反向跟踪失败')
    valid = ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
    valid &= np.linalg.norm(back-corners, axis=2).ravel() <= .5
    if valid.sum() < 8 or valid.mean() < .8:
        raise ValueError('静止图像特征丢失或遮挡过多')
    motion = np.linalg.norm(moved-corners, axis=2).ravel()[valid]
    difference = np.abs(a.astype(float)-b)
    mean, changed = float(difference.mean()), float((difference > 20).mean())
    if min(a.shape) < 16:
        raise ValueError('静止检查区域过小，局部证据不足')
    # 每个可见小区域都接受独立检查，不能让背景角点和全图均值掩盖线材移动。
    local_mean = cv2.boxFilter(difference.astype(np.float32), -1, (16, 16))[8:-7, 8:-7]
    local_fraction = cv2.boxFilter((difference > 20).astype(np.float32), -1, (16, 16))[8:-7, 8:-7]
    if np.any((local_fraction > .12) & (local_mean > 3)):
        raise ValueError('相机或可见局部场景未静止：局部像素变化超限')
    if not np.isfinite(motion).all() or motion.max() > .75 or mean > 3 or changed > .02:
        raise ValueError('相机或可见场景未静止：特征位移/像素变化超限')
    return {'tracked_features': int(valid.sum()), 'feature_retention': float(valid.mean()),
            'max_displacement_px': float(motion.max()), 'mean_gray_change': mean,
            'changed_pixel_fraction': changed, 'local_change_max_fraction': float(local_fraction.max()),
            'local_mean_gray_change_max': float(local_mean.max())}


def _verify_stationary_sequence(bundle, output, *, allow_simulated=False, _pending=False):
    """只读重新验证manifest、每帧原始快照和整个序列；不访问设备。"""
    root = Path(output).resolve(strict=True)
    manifest = read_json(root/'manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != 'stationary_rgb_sequence'
            or manifest.get('status') not in (('pending', 'validated') if _pending else ('validated',))):
        raise ValueError('序列manifest版本或完成状态无效')
    if (manifest.get('data_origin') not in ('physical_camera', 'simulated')
            or manifest['data_origin'] == 'simulated' and not allow_simulated):
        raise ValueError('模拟序列不能当作真实相机多帧数据')
    if (manifest.get('scene_stationary_confirmed') is not True
            or manifest.get('calibration_sha256') != bundle.settings['sha256']
            or manifest.get('camera_serial') != bundle.settings['camera_serial']
            or manifest.get('robot_motion_commands_sent') != 0 or manifest.get('pose_aggregation') != 'none'):
        raise ValueError('序列静止确认/设备/标定/只读合同不符')
    entries = manifest['frames']
    if not 2 <= len(entries) <= 10 or len(entries) != manifest.get('requested_frames'):
        raise ValueError('序列帧数不完整')
    if not _pending:
        path = _safe_path(root, manifest['verification'])
        if sha256(path) != manifest['verification_sha256']:
            raise ValueError('序列验证报告缺失或指纹不符')
    robot_path = _safe_path(root, manifest['robot_report'])
    if sha256(robot_path) != manifest['robot_report_sha256']:
        raise ValueError('连续机器人反馈文件指纹不符')
    robot = read_json(robot_path)
    merge_feedback_reports([robot])
    segments = manifest.get('feedback_segments', [])
    if manifest['data_origin'] == 'physical_camera' and not segments:
        raise ValueError('真实序列缺少原始连续监听分段')
    if segments:
        raw = []
        for segment in segments:
            path = _safe_path(root, segment['path'])
            if sha256(path) != segment['sha256']:
                raise ValueError('原始反馈分段指纹不符')
            raw.append(read_json(path))
        if merge_feedback_reports(raw)['samples'] != robot['samples']:
            raise ValueError('合并反馈与原始分段不一致')
    module = helpers(bundle)
    frames, seen_hashes, seen_ids, stamps, device_stamps = [], set(), set(), [], set()
    reference = previous = None
    roi = manifest['scene_roi_xyxy']
    for entry in entries:
        directory = _safe_path(root, entry['snapshot'])
        expected_files = {'capture/rgb.png', 'capture/depth.png', 'capture/camera.json', 'capture/frames.json',
                          'capture/COMPLETE', 'pair.json', 'robot.json'}
        if (set(entry['sha256']) != expected_files or entry.get('camera_serial') != manifest['camera_serial']
                or entry.get('calibration_sha256') != bundle.settings['sha256']
                or _safe_path(root, entry['rgb']) != directory/'capture/rgb.png'):
            raise ValueError('单帧manifest文件清单/设备/标定信息不完整')
        for relative, digest in entry['sha256'].items():
            path = _safe_path(directory, relative)
            if sha256(path) != digest:
                raise ValueError('多帧文件完整性校验失败：'+relative)
        if sha256(directory/'robot.json') != manifest['robot_report_sha256']:
            raise ValueError('单帧与连续机器人报告不一致')
        paired = verify_snapshot(bundle, directory)
        metadata = read_json(directory/'capture/frames.json')
        # 不使用单帧RGB窗口回退：多帧必须全程反馈稳定、连续覆盖，包括帧间间隙。
        whole = module.verify_pair(robot, metadata, bundle.session['robot'])
        pair = read_json(directory/'pair.json')
        if (pair.get('status') != 'paired_stationary' or pair.get('robot_motion_commands_sent') != 0
                or pair.get('data_origin') != manifest['data_origin']
                or pair.get('image_sha256') != sha256(directory/'capture/rgb.png')
                or pair.get('calibration_sha256') != bundle.settings['sha256']
                or not np.allclose(pair['T_base_flange'], flange_transform(paired['robot_pose']), atol=1e-10, rtol=0)):
            raise ValueError('单帧配对文件与实际图像/反馈不一致')
        if entry.get('sdk_frame_index') != metadata['rgb'].get('index'):
            raise ValueError('SDK原始帧编号与manifest不符')
        stamp = int(metadata['rgb']['host_receive_timestamp_us'])
        device = int(metadata['rgb']['device_timestamp_us'])
        if (stamp <= 0 or device <= 0 or str(stamp) != entry['host_receive_timestamp_us']
                or str(device) != entry['device_timestamp_us'] or device in device_stamps):
            raise ValueError('RGB设备/主机时间戳缺失、重复或不一致')
        if stamps and stamp <= stamps[-1]:
            raise ValueError('跨帧RGB主机时间戳不严格递增')
        stamps.append(stamp)
        device_stamps.add(device)
        if entry['frame_id'] in seen_ids:
            raise ValueError('帧ID重复')
        seen_ids.add(entry['frame_id'])
        image = cv2.imread(str(directory/'capture/rgb.png'))
        if image is None or image.shape[1::-1] != tuple(bundle.size):
            raise ValueError('RGB文件不可读取或尺寸不匹配')
        depth = cv2.imread(str(directory/'capture/depth.png'), cv2.IMREAD_UNCHANGED)
        camera = read_json(directory/'capture/camera.json')
        if (depth is None or depth.dtype != np.uint16 or depth.shape !=
                (camera['depth']['height'], camera['depth']['width'])):
            raise ValueError('原始深度文件不完整；本模式不验证深度定位精度')
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen_hashes:
            raise ValueError('重复RGB像素内容：拒绝复制图片或冻结帧伪装成多帧')
        if digest != entry['rgb_pixels_sha256']:
            raise ValueError('RGB像素指纹不符')
        seen_hashes.add(digest)
        if (not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi)
                or not 0 <= roi[0] < roi[2] <= image.shape[1] or not 0 <= roi[1] < roi[3] <= image.shape[0]):
            raise ValueError('静止检查ROI无效')
        if reference is None:
            reference = image
        checks = None if previous is None else {'first_frame': _scene_change(reference, image, roi),
                                                'previous_frame': _scene_change(previous, image, roi)}
        previous = image
        frames.append({'frame_id': entry['frame_id'], 'robot_pose': paired['robot_pose'],
                       'pose_sample_index': whole['streams']['rgb']['nearest_index'], 'image_stability': checks})
    if not _pending and not (root/'COMPLETE').is_file():
        raise ValueError('序列缺少COMPLETE标记')
    return {'status': 'validated', 'data_origin': manifest['data_origin'], 'frames': frames,
            'robot_stationarity': 'continuous_full_window', 'pose_aggregation': 'none',
            'hardware_sync_verified': False, 'exposure_pose_verified': False,
            'interpretation': '原始独立采集+逐帧接收配对+全程反馈稳定+可见图像静止检查；不证明曝光同步或所有遮挡线材静止'}


def verify_stationary_sequence(bundle, output, *, allow_simulated=False, _pending=False):
    """输入格式损坏统一拒绝，便于离线分析保存人工复核报告；不吞设备/模型故障。"""
    try:
        return _verify_stationary_sequence(bundle, output, allow_simulated=allow_simulated, _pending=_pending)
    except (AttributeError, TypeError, IndexError, KeyError, OverflowError, cv2.error) as exc:
        raise ValueError('序列格式或必要字段无效：'+str(exc)) from exc


def capture_stationary_sequence(bundle, output, *, frames=5, scene_stationary_confirmed=False,
                                scene_roi=None, _capture_frame=None, _feedback=None):
    """真实模式逐次调用既有SDK采图程序；注入测试驱动时强制标记simulated。"""
    if type(frames) is not int or not 2 <= frames <= 10 or scene_stationary_confirmed is not True:
        raise ValueError('多帧需要2至10帧及明确的scene_stationary_confirmed')
    roi = [0, 0, *bundle.size] if scene_roi is None else scene_roi
    if (not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi)
            or not 0 <= roi[0] < roi[2] <= bundle.size[0] or not 0 <= roi[1] < roi[3] <= bundle.size[1]):
        raise ValueError('静止检查ROI无效')
    module = helpers(bundle)
    simulated = _capture_frame is not None or _feedback is not None
    if simulated and (_capture_frame is None or _feedback is None):
        raise ValueError('模拟模式必须同时提供相机和反馈测试驱动')
    if not simulated:
        module.require_can_interface(bundle.settings['channel'])
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    receiver = _feedback if simulated else _FeedbackSegments(bundle, root)
    manifest = {'schema_version': 1, 'kind': 'stationary_rgb_sequence', 'status': 'capturing',
                'requested_frames': frames, 'data_origin': 'simulated' if simulated else 'physical_camera',
                'camera_serial': bundle.settings['camera_serial'], 'calibration_sha256': bundle.settings['sha256'],
                'scene_stationary_confirmed': True, 'scene_roi_xyxy': roi,
                'robot_motion_commands_sent': 0, 'pose_aggregation': 'none', 'frames': [],
                'capture_method': 'independent_sdk_rgbd_captures_with_overlapping_passive_feedback',
                'hardware_sync_verified': False, 'exposure_pose_verified': False}
    _write(root/'manifest.json', manifest)
    try:
        receiver.start()
        directories = []
        for index in range(frames):
            directory = root/'frames'/f'frame_{index:03d}'
            directory.mkdir(parents=True)
            if simulated:
                _capture_frame(directory/'capture', index)
            else:
                with (directory/'camera.log').open('w') as log:
                    subprocess.run(module.capture_command(bundle.root/'build/handeye_capture', directory/'capture',
                                                          bundle.session['camera']), stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=25, cwd=directory)
            directories.append(directory)
        robot, segments = receiver.finish()
        _write(root/'robot.json', robot)
        manifest.update(robot_report='robot.json', robot_report_sha256=sha256(root/'robot.json'),
                        feedback_segments=segments)
        for directory in directories:
            os.link(root/'robot.json', directory/'robot.json')
            verified = verify_snapshot(bundle, directory)
            metadata = read_json(directory/'capture/frames.json')
            stamp, device = int(metadata['rgb']['host_receive_timestamp_us']), int(metadata['rgb']['device_timestamp_us'])
            image = cv2.imread(str(directory/'capture/rgb.png'))
            if image is None:
                raise ValueError('RGB无法读取')
            pair = {'status': 'paired_stationary', 'calibration_sha256': bundle.settings['sha256'],
                    'data_origin': manifest['data_origin'],
                    'capture_schema_version': 2, 'capture_kind': 'stationary_sequence_frame',
                    'sequence_manifest': '../../manifest.json',
                    'image_sha256': sha256(directory/'capture/rgb.png'),
                    'T_base_flange': flange_transform(verified['robot_pose']).tolist(), 'verification': verified,
                    'robot_motion_commands_sent': 0, 'hardware_sync_verified': False, 'exposure_pose_verified': False}
            _write(directory/'pair.json', pair)
            files = ['capture/rgb.png', 'capture/depth.png', 'capture/camera.json', 'capture/frames.json',
                     'capture/COMPLETE', 'pair.json', 'robot.json']
            manifest['frames'].append({
                'frame_id': f'rgb_{stamp}_{uuid.uuid4().hex[:8]}', 'snapshot': str(directory.relative_to(root)),
                'rgb': str((directory/'capture/rgb.png').relative_to(root)),
                'host_receive_timestamp_us': str(stamp), 'device_timestamp_us': str(device),
                'sdk_frame_index': metadata['rgb']['index'], 'capture_session': directory.name,
                'camera_serial': bundle.settings['camera_serial'], 'calibration_sha256': bundle.settings['sha256'],
                'rgb_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'sha256': {name: sha256(directory/name) for name in files}})
        manifest['status'] = 'pending'
        _write(root/'manifest.json', manifest)
        verification = verify_stationary_sequence(bundle, root, allow_simulated=simulated, _pending=True)
        _write(root/'verification.json', verification)
        manifest.update(status='validated', verification='verification.json', verification_sha256=sha256(root/'verification.json'))
        _write(root/'manifest.json', manifest)
        (root/'COMPLETE').write_text('Independent RGB frames and stationary pairing validated\n')
        return root
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        _write(root/'manifest.json', manifest)
        _write(root/'FAILED.json', {'error': str(exc), 'robot_motion_commands_sent': 0,
                                  'data_origin': manifest['data_origin']})
        raise
    finally:
        receiver.close()
