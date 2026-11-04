#!/usr/bin/env bash
set -euo pipefail

wire_grasp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_ws="$(cd "$wire_grasp_root/../agilex_official_ws" && pwd)"

# 视觉环境是仓库内Conda，官方MoveIt环境是系统Humble；这里必须显式清理后再加载。
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
unset LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$official_ws/python"
set +u
source /opt/ros/humble/setup.bash
source "$official_ws/install_humble/setup.bash"
set -u
export PYTHONPATH="${wire_grasp_root}:${PYTHONPATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-93}"
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LC_NUMERIC=C

exec /usr/bin/python3 "${wire_grasp_root}/scripts/run_moveit_wire_grasp.py" "$@"
