#!/usr/bin/env bash
set -euo pipefail
YOLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${YOLO_ROOT}/.yolo-deps:${YOLO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export YOLO_CONFIG_DIR="${YOLO_ROOT}/output/yolo_config"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
export LD_LIBRARY_PATH="${YOLO_ROOT}/.conda/lib/python3.10/site-packages/torch/lib:${YOLO_ROOT}/third_party/pyorbbecsdk-v1/install/lib:${YOLO_ROOT}/third_party/pyorbbecsdk-v1/sdk/lib/linux_x64:${LD_LIBRARY_PATH:-}"
exec "${YOLO_ROOT}/.conda/bin/python" "${YOLO_ROOT}/scripts/run_yolo.py" "$@"
