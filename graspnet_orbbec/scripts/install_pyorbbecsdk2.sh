#!/usr/bin/env bash
# 把 PyOrbbecSDK v2 装到 third_party/pyorbbecsdk2，不覆盖 Astra 用的 v1。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ROOT}/.conda"
export PYTHONNOUSERSITE=1
TARGET="${ROOT}/third_party/pyorbbecsdk2"
mkdir -p "${TARGET}"
python -m pip install 'pyorbbecsdk2==2.1.2' --target "${TARGET}" --no-deps
echo "已安装到 ${TARGET}（--no-deps，避免拉入 OpenCV 5 / 另一份 Open3D）"
PYTHONPATH="${TARGET}" python -c "import pyorbbecsdk; print('pyorbbecsdk2', getattr(pyorbbecsdk,'__version__','?'), pyorbbecsdk.__file__)"
