#!/usr/bin/env bash
set -euo pipefail
wire_grasp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${wire_grasp_root}"
export LD_LIBRARY_PATH="${wire_grasp_root}/.conda/lib/python3.10/site-packages/torch/lib:${wire_grasp_root}/third_party/pyorbbecsdk-v1/install/lib:${wire_grasp_root}/third_party/pyorbbecsdk-v1/sdk/lib/linux_x64:${LD_LIBRARY_PATH:-}"
exec "${wire_grasp_root}/.conda/bin/python" "${wire_grasp_root}/scripts/run_wire_grasp.py" "$@"
