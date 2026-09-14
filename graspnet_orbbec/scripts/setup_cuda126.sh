#!/usr/bin/env bash
# 安装与 PyTorch cu126 匹配的 CUDA 12.6 nvcc + cudart + CCCL（NVIDIA redistributable）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ROOT}/.conda"
DEST="${ENV}/cuda-12.6"
VER=12.6.77
BASE=https://developer.download.nvidia.com/compute/cuda/redist
TMP="${TMPDIR:-/tmp}/cuda126_redist"
mkdir -p "${TMP}" "${DEST}"

fetch() {
  local pkg="$1"
  local f="${pkg}-linux-x86_64-${VER}-archive.tar.xz"
  local url="${BASE}/${pkg}/linux-x86_64/${f}"
  echo ">>> $url"
  wget -q -O "${TMP}/${f}" "${url}"
  tar -C "${TMP}" -xf "${TMP}/${f}"
}

fetch cuda_nvcc
fetch cuda_cudart
fetch cuda_cccl

rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -a "${TMP}/cuda_nvcc-linux-x86_64-${VER}-archive/." "${DEST}/"
mkdir -p "${DEST}/include" "${DEST}/lib64"
cp -a "${TMP}/cuda_cudart-linux-x86_64-${VER}-archive/include/." "${DEST}/include/"
cp -a "${TMP}/cuda_cccl-linux-x86_64-${VER}-archive/include/." "${DEST}/include/"
if [[ -d "${TMP}/cuda_cudart-linux-x86_64-${VER}-archive/lib" ]]; then
  cp -a "${TMP}/cuda_cudart-linux-x86_64-${VER}-archive/lib/." "${DEST}/lib64/"
fi
ln -sfn lib64 "${DEST}/lib"

# 补 cusparse/cublas 头（来自 pip NVIDIA 包）
SP="${ENV}/lib/python3.10/site-packages"
for lib in cusparse cublas curand cufft cusolver; do
  if [[ -d "${SP}/nvidia/${lib}/include" ]]; then
    cp -a "${SP}/nvidia/${lib}/include/." "${DEST}/include/"
  fi
done

"${DEST}/bin/nvcc" --version
echo "CUDA_HOME=${DEST}"
