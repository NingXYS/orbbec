# 在仓库根目录 ~/orbbec 下也可以 source。实际环境与代码在 graspnet_orbbec/。
# 用法: cd ~/orbbec && source scripts/activate.sh
_ORBBEC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${_ORBBEC_ROOT}/graspnet_orbbec/scripts/activate.sh"
cd "${_ORBBEC_ROOT}/graspnet_orbbec"
echo "工作目录: ${PWD}"
