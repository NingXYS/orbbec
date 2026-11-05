"""仅合成相机与反馈；绝不打开相机、CAN或运动接口。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import stationary_capture as single
from arm import stationary_sequence as sequence
from arm.handeye_bridge import read_json, sha256
from arm.wire_geometry import flange_transform
from scripts.run_wire_grasp import parser
from scripts import run_wire_grasp as cli


def write(path, value):
    path.write_text(json.dumps(value))


class PairHelper:
    pose_to_base_gripper = staticmethod(lambda pose, convention: flange_transform(pose))

    @staticmethod
    def rotation_error_deg(left, right):
        return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(left @ right.T)[0])))

    @classmethod
    def verify_pair(cls, robot, frames, convention):
        report = single.verify_pair_rgb_window(cls, robot, frames, convention)
        if (report['whole_window_translation_max_change_m'] > .0005
                or report['whole_window_rotation_max_change_deg'] > .1):
            raise ValueError('反馈未稳定：完整监听窗口发生运动')
        return report


@pytest.fixture
def rig(monkeypatch):
    camera = {'serial': 'SIMULATED_CAMERA', 'rgb': {'width': 160, 'height': 120},
              'depth': {'width': 80, 'height': 60}}

    def check_camera(actual):
        if actual != camera:
            raise ValueError('相机标定或设备不匹配')

    bundle = SimpleNamespace(size=(160, 120), settings={
        'camera_serial': camera['serial'], 'sha256': {'calibration': 'simulation-only'},
        'channel': 'simulated_can', 'firmware_report': 'not-used.json'},
        session={'robot': {'firmware_version': 'SIM', 'audited_source_sha': 'SIM'}, 'camera': {}},
        check_camera=check_camera)
    samples = [{'host_kernel_timestamp_s': 999+i*.01, 'arrival_monotonic_s': 99+i*.01,
                'pose_m_rad': [i*1e-7, 0, .2, 0, 0, 0]} for i in range(701)]
    robot = {'status': 'passed', 'channel': 'simulated_can', 'firmware_version': 'SIM',
             'source_sha': 'SIM', 'application_frames_sent_by_tool': 0,
             'sdk_session_connected': False, 'samples': samples}

    class Feedback:
        started = closed = False

        def start(self):
            self.started = True

        def finish(self):
            return robot, []

        def close(self):
            self.closed = True

    feedback = Feedback()
    base = np.random.default_rng(8).integers(40, 210, (120, 160, 3), dtype=np.uint8)

    def capture(path, index):
        path.mkdir()
        # 每帧独立生成轻微传感器噪声，明确为合成测试，不能充当真实多帧。
        noise = np.random.default_rng(index+20).integers(-1, 2, base.shape)
        image = (base.astype(np.int16)+noise).astype(np.uint8)
        assert cv2.imwrite(str(path/'rgb.png'), image)
        assert cv2.imwrite(str(path/'depth.png'), np.full((60, 80), 400, np.uint16))
        write(path/'camera.json', camera)
        stamp = 1000000000+index*400000
        write(path/'frames.json', {name: {'index': str(30+index),
              'host_receive_timestamp_us': str(stamp), 'device_timestamp_us': str(2000000+index*400000)}
              for name in ('rgb', 'depth')})
        (path/'COMPLETE').write_text('synthetic test fixture\n')

    monkeypatch.setattr(sequence, 'helpers', lambda b: PairHelper)
    monkeypatch.setattr(single, 'helpers', lambda b: PairHelper)
    return SimpleNamespace(bundle=bundle, robot=robot, feedback=feedback, capture=capture, base=base)


def collect(rig, path, count=5, capture=None, **kwargs):
    return sequence.capture_stationary_sequence(
        rig.bundle, path, frames=count, scene_stationary_confirmed=True,
        _capture_frame=capture or rig.capture, _feedback=rig.feedback, **kwargs)


@pytest.mark.parametrize('count', [2, 5, 10])
def test_independent_frames_manifest_and_single_compatibility(rig, tmp_path, count):
    root = collect(rig, tmp_path/'sequence', count)
    manifest = read_json(root/'manifest.json')
    assert rig.feedback.started and rig.feedback.closed
    assert manifest['data_origin'] == 'simulated'
    assert not (root/'capture/rgb.png').exists()
    assert len({e['frame_id'] for e in manifest['frames']}) == count
    assert len({e['rgb_pixels_sha256'] for e in manifest['frames']}) == count
    assert len({e['host_receive_timestamp_us'] for e in manifest['frames']}) == count
    assert len({e['device_timestamp_us'] for e in manifest['frames']}) == count
    with pytest.raises(ValueError, match='模拟序列'):
        sequence.verify_stationary_sequence(rig.bundle, root)
    report = sequence.verify_stationary_sequence(rig.bundle, root, allow_simulated=True)
    poses = []
    for entry, verified in zip(manifest['frames'], report['frames']):
        directory = root/entry['snapshot']
        old = single.verify_snapshot(rig.bundle, directory)
        pair = read_json(directory/'pair.json')
        assert old['robot_pose'] == verified['robot_pose']
        assert verified['robot_pose'] == rig.robot['samples'][verified['pose_sample_index']]['pose_m_rad']
        assert np.allclose(pair['T_base_flange'], flange_transform(old['robot_pose']))
        poses.append(old['robot_pose'][0])
    assert len(set(poses)) == count  # 逐帧实际配对，不能用平均位姿代替。
    assert report['pose_aggregation'] == 'none'
    assert report['hardware_sync_verified'] is False


@pytest.mark.parametrize('failure', ['duplicate_pixels', 'duplicate_host', 'backward_host',
                                  'duplicate_device', 'missing_timestamp', 'missing_complete',
                                  'wrong_camera', 'bad_depth', 'moving_camera', 'no_features'])
def test_bad_camera_data_rejected_and_cleanup(rig, tmp_path, failure):
    root = tmp_path/'bad'

    def capture(path, index):
        rig.capture(path, index)
        if failure == 'no_features':
            image = np.full((120, 160, 3), 120+index, np.uint8)
            cv2.imwrite(str(path/'rgb.png'), image)
        if index != 1:
            return
        metadata = read_json(path/'frames.json')
        if failure == 'duplicate_pixels':
            first = cv2.imread(str(root/'frames/frame_000/capture/rgb.png'))
            cv2.imwrite(str(path/'rgb.png'), first)
        elif failure in ('duplicate_host', 'backward_host'):
            metadata['rgb']['host_receive_timestamp_us'] = str(1000000000-(100000 if failure == 'backward_host' else 0))
        elif failure == 'duplicate_device':
            metadata['rgb']['device_timestamp_us'] = '2000000'
        elif failure == 'missing_timestamp':
            del metadata['rgb']['device_timestamp_us']
        elif failure == 'missing_complete':
            (path/'COMPLETE').unlink()
        elif failure == 'wrong_camera':
            camera = read_json(path/'camera.json')
            camera['serial'] = 'WRONG'
            write(path/'camera.json', camera)
        elif failure == 'bad_depth':
            cv2.imwrite(str(path/'depth.png'), np.zeros((30, 30), np.uint8))
        elif failure == 'moving_camera':
            shifted = cv2.warpAffine(rig.base, np.float32([[1, 0, 3], [0, 1, 0]]), (160, 120))
            cv2.imwrite(str(path/'rgb.png'), shifted)
        write(path/'frames.json', metadata)

    with pytest.raises((ValueError, KeyError)):
        collect(rig, root, capture=capture)
    assert rig.feedback.closed
    assert read_json(root/'manifest.json')['status'] == 'failed'
    assert (root/'FAILED.json').exists() and not (root/'COMPLETE').exists()


@pytest.mark.parametrize('fault', ['gap', 'motion', 'bad_source', 'sends', 'sdk_session', 'depth_pair'])
def test_pairing_and_full_window_faults_rejected(rig, tmp_path, fault):
    if fault == 'gap':
        del rig.robot['samples'][120:130]
    elif fault == 'motion':
        rig.robot['samples'][-1]['pose_m_rad'][0] = .01  # 最后一张RGB窗口以外也必须拒绝。
    elif fault == 'bad_source':
        rig.robot['source_sha'] = 'WRONG'
    elif fault == 'sends':
        rig.robot['application_frames_sent_by_tool'] = 1
    elif fault == 'sdk_session':
        rig.robot['sdk_session_connected'] = True

    def capture(path, index):
        rig.capture(path, index)
        if fault == 'depth_pair':
            value = read_json(path/'frames.json')
            value['depth']['host_receive_timestamp_us'] = '2000000000'
            write(path/'frames.json', value)

    with pytest.raises(ValueError):
        collect(rig, tmp_path/'bad', capture=capture)
    assert rig.feedback.closed


@pytest.mark.parametrize('fault', ['pixels', 'pair_pose', 'missing_file', 'manifest_time',
                                  'missing_hash', 'path_escape', 'no_segments', 'verification'])
def test_revalidation_detects_integrity_and_manifest_errors(rig, tmp_path, fault):
    root = collect(rig, tmp_path/'sequence')
    manifest = read_json(root/'manifest.json')
    entry = manifest['frames'][0]
    directory = root/entry['snapshot']
    if fault == 'pixels':
        cv2.imwrite(str(directory/'capture/rgb.png'), np.zeros((120, 160, 3), np.uint8))
    elif fault == 'pair_pose':
        pair = read_json(directory/'pair.json')
        pair['T_base_flange'][0][3] += .01
        write(directory/'pair.json', pair)
        entry['sha256']['pair.json'] = sha256(directory/'pair.json')
    elif fault == 'missing_file':
        (directory/'capture/depth.png').unlink()
    elif fault == 'manifest_time':
        entry['host_receive_timestamp_us'] = '1'
    elif fault == 'missing_hash':
        del entry['sha256']['capture/frames.json']
    elif fault == 'path_escape':
        entry['rgb'] = '../outside.png'
        (tmp_path/'outside.png').write_bytes(b'outside')
    elif fault == 'no_segments':
        manifest['data_origin'] = 'physical_camera'
    elif fault == 'verification':
        (root/'verification.json').write_text('{}')
    write(root/'manifest.json', manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        sequence.verify_stationary_sequence(rig.bundle, root, allow_simulated=True)


@pytest.mark.parametrize('roi', [[], [0, 0, 0, 50], [-1, 0, 10, 10], [0, 0, 161, 120], [0., 0, 80, 80]])
def test_invalid_scene_roi(rig, tmp_path, roi):
    with pytest.raises(ValueError, match='ROI'):
        collect(rig, tmp_path/'bad', scene_roi=roi)
    assert not rig.feedback.started


def test_valid_scene_roi(rig, tmp_path):
    root = collect(rig, tmp_path/'sequence', scene_roi=[10, 10, 150, 110])
    assert read_json(root/'manifest.json')['scene_roi_xyxy'] == [10, 10, 150, 110]


def test_confirmation_required_before_capture(rig, tmp_path):
    with pytest.raises(ValueError, match='scene_stationary_confirmed'):
        sequence.capture_stationary_sequence(rig.bundle, tmp_path/'bad', _capture_frame=rig.capture,
                                              _feedback=rig.feedback)
    assert not rig.feedback.started


def test_merge_feedback_overlaps_without_averaging(rig):
    a, b = deepcopy(rig.robot), deepcopy(rig.robot)
    a['samples'], b['samples'] = a['samples'][:400], b['samples'][300:]
    merged = sequence.merge_feedback_reports([a, b])
    assert merged['samples'] == rig.robot['samples']
    assert merged['feedback_segments'] == 2
    b['samples'][0]['pose_m_rad'][0] += .001
    with pytest.raises(ValueError, match='同一时间戳位姿不一致'):
        sequence.merge_feedback_reports([a, b])


@pytest.mark.parametrize('fault', ['reverse', 'nan', 'monotonic', 'source', 'status'])
def test_merge_rejects_bad_feedback(rig, fault):
    report = deepcopy(rig.robot)
    if fault == 'reverse':
        report['samples'].reverse()
    elif fault == 'nan':
        report['samples'][0]['host_kernel_timestamp_s'] = float('nan')
    elif fault == 'monotonic':
        report['samples'][0]['arrival_monotonic_s'] = 1000
    elif fault == 'source':
        report['source_sha'] = 'WRONG'
    else:
        report['status'] = 'failed'
    with pytest.raises(ValueError):
        sequence.merge_feedback_reports([rig.robot, report])


def test_cli_default_single_and_opt_in_multi():
    assert parser().parse_args(['capture']).frames == 1
    args = parser().parse_args(['capture', '--frames', '5', '--scene-stationary-confirmed'])
    assert args.frames == 5 and args.scene_stationary_confirmed
    with pytest.raises(SystemExit):
        parser().parse_args(['capture', '--frames', '11'])


def test_cli_single_dispatch_unchanged(rig, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'HandeyeBundle', lambda path: rig.bundle)
    monkeypatch.setattr(single, 'capture_snapshot', lambda bundle, out: calls.append(out) or out)
    monkeypatch.setattr(sys, 'argv', ['run_wire_grasp.py', 'capture', '--out', str(tmp_path/'single')])
    assert cli.main() == 0
    assert calls == [tmp_path/'single']
    assert not rig.feedback.started


def test_feedback_segments_only_launch_readonly_tool_and_merge(rig, tmp_path, monkeypatch):
    rig.bundle.root = tmp_path/'calibration'
    commands = []

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        report = deepcopy(rig.robot)
        report['samples'] = report['samples'][:400] if len(commands) == 1 else report['samples'][300:]
        write(Path(command[command.index('--output')+1]), report)
        return Process()

    monkeypatch.setattr(sequence.subprocess, 'Popen', popen)
    recorder = sequence._FeedbackSegments(rig.bundle, tmp_path)
    recorder.start()
    recorder._launch()  # 无等待地模拟下一重叠窗口。
    robot, segments = recorder.finish()
    recorder.close()
    assert robot['samples'] == rig.robot['samples']
    assert len(segments) == 2
    for command in commands:
        assert Path(command[1]).name == 'check_flange_feedback.py'
        assert command[command.index('--seconds')+1] == '10'
        assert '--execute' not in command
    for segment in segments:
        assert sha256(tmp_path/segment['path']) == segment['sha256']


def test_feedback_failure_terminates_only_own_process(rig, tmp_path):
    recorder = sequence._FeedbackSegments(rig.bundle, tmp_path)
    calls = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            calls.append('terminate')

        def kill(self):
            calls.append('kill')

        def wait(self, timeout=None):
            if timeout == 3:
                raise subprocess.TimeoutExpired('simulated-child', 3)
            return 0

    recorder.jobs.append((Process(), tmp_path/'unused'))
    recorder.close()
    assert calls == ['terminate', 'kill']


def test_camera_error_preserves_failure_and_closes_feedback(rig, tmp_path):
    def fail(path, index):
        raise OSError('simulated camera disconnect')

    with pytest.raises(OSError, match='disconnect'):
        collect(rig, tmp_path/'failed', capture=fail)
    assert rig.feedback.closed
    assert read_json(tmp_path/'failed/FAILED.json')['data_origin'] == 'simulated'
