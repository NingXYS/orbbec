#!/usr/bin/env bash
set -euo pipefail
sam_task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sam_source_sha="2b90b9f5ceec907a1c18123530e92e794ad901a4"
cd "$sam_task_root"
if [[ ! -d third_party/sam2 ]]; then
  git clone https://github.com/facebookresearch/sam2.git third_party/sam2
  git -C third_party/sam2 checkout --detach "$sam_source_sha"
fi
if [[ "$(git -C third_party/sam2 rev-parse HEAD)" != "$sam_source_sha" ]]; then
  printf '%s\n' 'SAM2源码版本与固定版本不同，已停止；请保留现有目录并检查，安装器不会重置它。' >&2
  exit 2
fi
# 依赖放在项目私有目录；不升级现有PyTorch/CUDA，也不构建额外CUDA扩展。
.conda/bin/python -m pip install --target third_party/sam2_runtime --no-deps \
  hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 iopath==0.1.10 portalocker==2.8.2
mkdir -p weights/sam2
if [[ ! -f weights/sam2/sam2.1_hiera_small.pt ]]; then
  curl --fail --location --retry 2 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
    -o weights/sam2/sam2.1_hiera_small.pt.part
  mv weights/sam2/sam2.1_hiera_small.pt.part weights/sam2/sam2.1_hiera_small.pt
fi
printf '%s\n' '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38  weights/sam2/sam2.1_hiera_small.pt' | sha256sum --check
printf '%s\n' 'SAM2.1 small 已就绪；运行 python scripts/run_live_gui.py --wire'
