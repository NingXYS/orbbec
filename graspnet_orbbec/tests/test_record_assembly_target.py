"""人工示教记录命令离线验证：不打开CAN，不覆盖当前项目目标。"""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('record_assembly_test', ROOT/'scripts/record_assembly_target.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def setup_record(tmp_path):
    template = json.loads((ROOT/'configs/wire_assembly_fixed_20261106.json').read_text())
    tool = tmp_path/template['tool']['file']
    tool.parent.mkdir(parents=True)
    tool.write_bytes((ROOT/template['tool']['file']).read_bytes())
    sample = {'joint_angles_rad':template['joint_positions_rad'],
              'flange_pose_m_rad':template['flange_goal']['position_m']+template['flange_goal']['rpy_rad'],
              'application_frames_sent':0, 'complete_pose_count':600,
              'observed_wall_time_s':1793935879.993318,
              'arm_status':{'ctrl_mode':2,'teach_status':2}}
    return template, sample


def test_main_records_consistent_target_backs_up_and_expires_old_checks(tmp_path, monkeypatch):
    template, sample = setup_record(tmp_path)
    p=tmp_path/'configs/target.json';p.write_text(json.dumps(template))
    original=p.read_bytes()
    monkeypatch.setattr(recorder, 'ROOT', tmp_path)
    monkeypatch.setattr(recorder, 'load_binding', lambda path: None)
    monkeypatch.setattr(recorder, 'receive_state', lambda binding: copy.deepcopy(sample))
    monkeypatch.setattr(sys, 'argv', ['record', '--target', str(p)])
    recorder.main()
    result=json.loads(p.read_text())
    assert Path(result['supersedes']).read_bytes()==original
    assert result['final_approach']==template['final_approach']
    assert result['integration_validation']['status']=='pending_for_updated_target'
    assert 'latest_pose_comparison' not in result
    assert result['planning_contract']['execution_allowed'] is False
    assert 'operator_must_exit_teaching_mode' in result['planning_contract']['blockers']
    assert result['joint_positions_rad']==sample['joint_angles_rad']
    assert np.allclose(result['grasp_tcp_goal']['T_base_frame'],template['grasp_tcp_goal']['T_base_frame'])


def test_moving_arm_does_not_change_target(tmp_path, monkeypatch):
    template, sample = setup_record(tmp_path)
    p=tmp_path/'configs/target.json';p.write_text(json.dumps(template));original=p.read_bytes()
    changed=copy.deepcopy(sample);changed['flange_pose_m_rad'][0]+=.01
    iterator=iter([sample, changed])
    monkeypatch.setattr(recorder, 'ROOT', tmp_path)
    monkeypatch.setattr(recorder, 'load_binding', lambda path: None)
    monkeypatch.setattr(recorder, 'receive_state', lambda binding: next(iterator))
    monkeypatch.setattr(sys, 'argv', ['record', '--target', str(p)])
    with pytest.raises(ValueError, match='停稳'):
        recorder.main()
    assert p.read_bytes()==original
    assert not list(p.parent.glob('*backup*'))


def test_changed_target_during_sampling_is_not_overwritten(tmp_path, monkeypatch):
    template, sample=setup_record(tmp_path)
    p=tmp_path/'configs/target.json';p.write_text(json.dumps(template))
    monkeypatch.setattr(recorder, 'ROOT', tmp_path)
    monkeypatch.setattr(recorder, 'load_binding', lambda path: None)
    def receive(binding):
        p.write_text('changed by operator')
        return copy.deepcopy(sample)
    monkeypatch.setattr(recorder, 'receive_state', receive)
    monkeypatch.setattr(sys, 'argv', ['record', '--target', str(p)])
    with pytest.raises(ValueError, match='目标文件已改变'):
        recorder.main()
    assert p.read_text()=='changed by operator'
