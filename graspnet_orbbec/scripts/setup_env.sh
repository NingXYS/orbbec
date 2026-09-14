#!/usr/bin/env bash
# 在项目目录内创建 conda 环境 .conda/ 并编译 GraspNet CUDA 扩展。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT}/.conda"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1

if ! command -v conda >/dev/null 2>&1; then
  echo "需要已安装 conda / miniconda"
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo ">>> 在 ${ENV_PREFIX} 创建 Python 3.10 环境"
  conda create --prefix "${ENV_PREFIX}" python=3.10 -y
fi

echo ">>> 安装 PyTorch cu126（与 RTX 40 系驱动兼容）"
"${PYTHON_BIN}" -m pip install --upgrade pip
if ! "${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  "${PYTHON_BIN}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    || "${PYTHON_BIN}" -m pip install torch torchvision -i https://mirrors.aliyun.com/pytorch-wheels/cu126
fi

echo ">>> 安装 Python 依赖"
"${PYTHON_BIN}" -m pip install -r "${ROOT}/requirements.txt" -i https://mirrors.aliyun.com/pypi/simple
"${PYTHON_BIN}" -m pip install -e "${ROOT}/third_party/graspnetAPI" ninja pybind11 \
  -i https://mirrors.aliyun.com/pypi/simple

echo ">>> 从源码编译 PyOrbbecSDK v1（PyPI 轮子是损坏的 macOS 包）"
bash "${ROOT}/scripts/build_pyorbbecsdk.sh"

echo ">>> 安装 CUDA 12.6 nvcc"
bash "${ROOT}/scripts/setup_cuda126.sh"

echo ">>> 编译 PointNet2 / knn"
bash "${ROOT}/scripts/compile_extensions.sh"

mkdir -p "${ROOT}/weights" "${ROOT}/output"
echo
echo "环境已就绪："
echo "  source ${ROOT}/scripts/activate.sh"
echo "  sudo bash ${ROOT}/scripts/install_udev.sh   # 相机 USB 权限"
echo "  bash ${ROOT}/scripts/download_weights.sh"
echo "  python ${ROOT}/infer/run_demo.py"
