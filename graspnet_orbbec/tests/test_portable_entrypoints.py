"""在改名且含空格的目录验证入口；ROS执行端替换为记录器，不连接硬件。"""

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def portable_launch(tmp_path):
    relocated = tmp_path / 'different user' / 'orbbec'
    ws = relocated / 'agilex_official_ws'
    (ws / 'python').mkdir(parents=True)
    (ws / 'install_humble').mkdir()
    (ws / 'install_humble/setup.bash').write_text(':\n')
    ros = tmp_path / 'fake ros'
    (ros / 'bin').mkdir(parents=True)
    (ros / 'setup.bash').write_text(':\n')
    recorder = ros / 'bin/ros2'
    recorder.write_text(
        '#!/usr/bin/python3\nimport os,sys,json\n'
        'print(json.dumps({"args":sys.argv[1:],'
        '"pythonpath":os.environ["PYTHONPATH"],'
        '"home":os.environ["ROS_HOME"]}))\n'
    )
    recorder.chmod(0o755)
    target = relocated / 'bringup/piper_x/start_real_control.sh'
    target.parent.mkdir(parents=True)
    lock = tmp_path / 'test.lock'
    text = (REPO / 'bringup/piper_x/start_real_control.sh').read_text()
    text = text.replace('/opt/ros/humble/bin/ros2', shlex.quote(str(recorder)))
    text = text.replace('source /opt/ros/humble/setup.bash', 'source ' + shlex.quote(str(ros / 'setup.bash')))
    text = text.replace('/tmp/piper_x_official_can0.lock', shlex.quote(str(lock)))
    flags = tmp_path / 'can_flags'
    flags.write_text('0x1\n')
    text = text.replace('/sys/class/net/can0/flags', shlex.quote(str(flags)))
    target.write_text(text)
    return target, ws, lock


def test_launch_relocated_with_spaces_preserves_arguments(portable_launch, tmp_path):
    target, ws, _ = portable_launch
    result = subprocess.run(
        ['/bin/bash', str(target), 'test:=value with spaces'], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data['pythonpath'] == str(ws / 'python')
    assert data['home'] == str(target.parent / 'runtime/official_ros')
    assert data['args'][-1] == 'test:=value with spaces'
    assert 'auto_enable:=false' in data['args']
    assert 'speed_percent:=5' in data['args']


def test_relocated_launch_rejects_existing_driver(portable_launch, tmp_path):
    target, _, _ = portable_launch
    fake_bin = tmp_path / 'fake bin'
    fake_bin.mkdir()
    pgrep = fake_bin / 'pgrep'
    pgrep.write_text('#!/bin/bash\nexit 0\n')
    pgrep.chmod(0o755)
    result = subprocess.run(
        ['/bin/bash', str(target)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'PATH': str(fake_bin) + ':' + os.environ['PATH']},
    )
    assert result.returncode == 2
    assert '遗留' in result.stderr


def test_relocated_launch_keeps_single_instance_lock(portable_launch, tmp_path):
    import fcntl

    target, _, lock = portable_launch
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ['/bin/bash', str(target)], cwd=tmp_path,
            capture_output=True, text=True,
        )
    assert result.returncode == 2
    assert '会话已在运行' in result.stderr


def test_launch_rejects_can_down_before_ros(portable_launch, tmp_path):
    target, _, _ = portable_launch
    (tmp_path / 'can_flags').write_text('0x40080\n')
    result = subprocess.run(
        ['/bin/bash', str(target)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert 'can0尚未启用' in result.stderr
    assert not (target.parent / 'runtime').exists()
