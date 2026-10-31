#!/usr/bin/env python3
"""只查询固件运动限值：发送12条0x472读取请求，不改变参数或控制状态。"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from read_wire_robot_state import load_binding


def main():
    import can
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=ROOT/'configs/wire_handeye_037K.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    binding = load_binding(args.binding)
    spec = importlib.util.spec_from_file_location('_limits_decoder', binding.root/'tools/check_flange_feedback.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    robot, profile = m.make_decoder(binding.session['robot']['firmware_version'], binding.settings['channel'])
    report = {'status': 'querying', 'firmware_profile': profile, 'motion_commands_sent': 0,
              'parameter_writes_sent': 0, 'sdk_session_connected': False, 'queries': []}
    with args.out.open('x') as out:
        try:
            with can.Bus(interface='socketcan', channel=binding.settings['channel'], ignore_config=True,
                         receive_own_messages=False, local_loopback=False) as bus:
                for joint in range(1, 7):
                    for kind, reply_id in ((1, 0x473), (2, 0x47C)):
                        message = robot._MSG_SearchMotorMaxAngleSpdAccLimit(joint_index=joint, search_content=kind)
                        frame = robot._parser.pack(message)
                        if frame.arbitration_id != 0x472 or bytes(frame.data) != bytes([joint, kind, 0, 0, 0, 0, 0, 0]):
                            raise ValueError('拒绝非0x472参数读取请求')
                        record = {'joint': joint, 'kind': kind, 'request_hex': bytes(frame.data).hex(), 'response': None}
                        report['queries'].append(record)
                        sent_at = time.time()
                        bus.send(frame, timeout=.1)
                        record['query_sent_wall_time_s'] = sent_at
                        deadline = time.monotonic()+.5
                        while time.monotonic() < deadline:
                            response = bus.recv(timeout=.01)
                            if (response is None or response.is_extended_id or response.is_remote_frame
                                    or response.is_error_frame or response.dlc != 8
                                    or response.arbitration_id != reply_id or response.data[0] != joint
                                    or response.timestamp < sent_at-.001 or time.time()-response.timestamp > .020):
                                continue
                            robot._parser.parse_packet(response)
                            field = 'motor_angle_limit_max_spd' if kind == 1 else 'motor_max_acc_limit'
                            record['response'] = getattr(robot._parser, field).msg.joints[joint-1].to_dict()
                            record['response_raw_hex'] = bytes(response.data).hex()
                            record['response_kernel_time_s'] = response.timestamp
                            break
            report['status'] = 'read_only_limits_observed' if all(q['response'] for q in report['queries']) else 'incomplete'
        finally:
            report['parameter_read_requests_attempted'] = len(report['queries'])
            json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
