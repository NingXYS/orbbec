"""复用标定工程的原始RGB/被动CAN采样，独立保存抓取快照。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    return helpers(bundle).verify_pair(robot, read_json(output/'capture/frames.json'), expected)


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
