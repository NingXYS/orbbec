#!/usr/bin/env bash
# 下载 GraspNet 官方 RealSense 预训练权重 checkpoint-rs.tar（推荐迁移）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/weights/checkpoint-rs.tar"
mkdir -p "${ROOT}/weights"
if [[ -f "${OUT}" ]]; then
  echo "已存在 ${OUT}"
  exit 0
fi
PY="${ROOT}/.conda/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python)"
fi
echo ">>> 下载 checkpoint-rs.tar（官方 RealSense 预训练权重）"
"${PY}" -m pip install -q gdown
set +e
"${PY}" -m gdown "https://drive.google.com/uc?id=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk" -O "${OUT}"
rc=$?
set -e
if [[ $rc -ne 0 || ! -s "${OUT}" ]]; then
  echo
  echo "自动下载失败（Google Drive 经常触发流量配额）。请手动下载后放到："
  echo "  ${OUT}"
  echo "官方链接："
  echo "  Google Drive: https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"
  echo "  百度网盘:     https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A"
  exit 1
fi
ls -lh "${OUT}"
