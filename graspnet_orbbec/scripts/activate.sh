# 用法: source scripts/activate.sh
# 激活项目内 .conda 环境（不要用 conda activate graspnet_orbbec 那个全局名）。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ROOT}/.conda"
export CUDA_HOME="${ROOT}/.conda/cuda-12.6"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# 禁止混入 ~/.local 的 numpy 2.x 等包
export PYTHONNOUSERSITE=1
# 不要把 .conda/lib 放最前，会盖掉系统 libtinfo 导致 bash 报警
# v2 由 capture/sdk_v2.py 在打开 Gemini 时按需加载，勿把 v2 和 v1 同时放进 PYTHONPATH
export LD_LIBRARY_PATH="${ROOT}/.conda/cuda-12.6/lib64:${ROOT}/.conda/lib/python3.10/site-packages/torch/lib:${ROOT}/third_party/pyorbbecsdk-v1/install/lib:${ROOT}/third_party/pyorbbecsdk-v1/sdk/lib/linux_x64:${LD_LIBRARY_PATH:-}"
# 阶段一默认清掉 ROS PYTHONPATH，避免 pytest 加载 launch_testing
if [[ "${KEEP_ROS_PYTHONPATH:-0}" != "1" ]]; then
  export PYTHONPATH="${ROOT}/third_party/pyorbbecsdk-v1/install/lib"
fi
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
echo "已激活 ${ROOT}/.conda  ($(python -c 'import sys; print(sys.executable)'))"
