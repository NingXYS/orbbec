#!/bin/bash
set -euo pipefail
bringup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
official_ws="$(cd "$bringup_root/../../agilex_official_ws" && pwd)"
# RViz 退出后后台可能仍存活；禁止同一真机再启动第二套 Action/驱动。
exec 9>/tmp/piper_x_official_can0.lock
if ! flock -n 9; then
  echo '官方 Piper X 会话已在运行，请使用原窗口；不要重复启动。' >&2
  exit 2
fi
if pgrep -f '^/usr/bin/python3 .*/agx_arm_ctrl/agx_arm_ctrl_single .*__ns:=/piper_x( |$)' >/dev/null; then
  echo '检测到遗留的官方 Piper X 驱动。请先退出原启动终端，再启动本入口。' >&2
  exit 2
fi
# 接口未启用时SDK可能仍启动ROS节点，提前失败以免把RViz可用误认为真机已连接。
can_flags_file=/sys/class/net/can0/flags
if [[ ! -r "$can_flags_file" ]]; then
  echo '未找到can0。请检查USB-CAN连接；参见真机启动手册第0节。' >&2
  exit 2
fi
can_flags="$(cat "$can_flags_file")"
if (( (can_flags & 1) == 0 )); then
  echo 'can0尚未启用，无法连接真机。请先按真机启动手册第0节配置1 Mbps CAN，再启动。' >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$official_ws/python"
set +u
source /opt/ros/humble/setup.bash
source "$official_ws/install_humble/setup.bash"
set -u
export ROS_DOMAIN_ID=93 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_HOME="$bringup_root/runtime/official_ros" ROS_LOG_DIR="$bringup_root/runtime/official_ros/log"
export XDG_CACHE_HOME="$bringup_root/runtime/official_cache" XDG_CONFIG_HOME="$bringup_root/runtime/official_config"
export LIBGL_ALWAYS_SOFTWARE=1 LC_NUMERIC=C
mkdir -p "$ROS_LOG_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
# 用户已批准接通真机；官方门控仅在执行轨迹时开放控制。
exec /opt/ros/humble/bin/ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 arm_type:=piper_x effector_type:=agx_gripper namespace:=piper_x \
  auto_enable:=false fast_mode:=false fw_version:=v191 speed_percent:=5 follow:=true \
  auto_control_gate:=true allow_trajectory_execution:=true \
  control_topic:=control/joint_states "$@"
