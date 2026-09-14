#!/usr/bin/env bash
# 编译 GraspNet PointNet2 / knn。需要 CUDA 12.6 nvcc（与 PyTorch cu126 对齐）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ROOT}/.conda"
CUDA_HOME_DIR="${ENV}/cuda-12.6"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME_DIR}"
export PATH="${CUDA_HOME_DIR}/bin:${ENV}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export LD_LIBRARY_PATH="${CUDA_HOME_DIR}/lib64:${ENV}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -x "${CUDA_HOME_DIR}/bin/nvcc" ]]; then
  echo "缺少 ${CUDA_HOME_DIR}/bin/nvcc"
  echo "请先执行: bash scripts/setup_cuda126.sh"
  exit 1
fi

"${ENV}/bin/python" "${ROOT}/scripts/patch_cuda_ext.py"
(
  cd "${ROOT}/third_party/graspnet-baseline/pointnet2"
  rm -rf build
  "${ENV}/bin/python" setup.py install
)
(
  cd "${ROOT}/third_party/graspnet-baseline/knn"
  rm -rf build
  "${ENV}/bin/python" setup.py install
)
"${ENV}/bin/python" -c "import pointnet2._ext as e; from knn_pytorch import knn_pytorch; print('CUDA 扩展 OK', e, knn_pytorch)"
