#!/usr/bin/env bash
# 从 third_party/pyorbbecsdk-v1 源码编译 Linux 绑定。
# PyPI 上的 pyorbbecsdk==1.3.2 manylinux 轮子实际是 macOS dylib，不能用。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT}/.conda"
SRC="${ROOT}/third_party/pyorbbecsdk-v1"
PY="${ENV_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1

"${PY}" -m pip uninstall -y pyorbbecsdk >/dev/null 2>&1 || true
rm -f "${ENV_PREFIX}/lib/python3.10/site-packages"/pyorbbecsdk*.so \
      "${ENV_PREFIX}/lib/python3.10/site-packages"/libOrbbecSDK* \
      "${ENV_PREFIX}/lib/python3.10/site-packages"/pyorbbecsdk.cpython-* 2>/dev/null || true
rm -rf "${ENV_PREFIX}/lib/python3.10/site-packages"/pyorbbecsdk-*.dist-info

"${PY}" -m pip install 'pybind11>=2.11' -i https://mirrors.aliyun.com/pypi/simple
PYBIND_DIR="$("${PY}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

rm -rf "${SRC}/build"
mkdir -p "${SRC}/build"
cd "${SRC}/build"
cmake -DPython3_ROOT_DIR="${ENV_PREFIX}" \
      -DPython3_EXECUTABLE="${PY}" \
      -Dpybind11_DIR="${PYBIND_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      ..
cmake --build . -j"$(nproc)"
cmake --install .

cd "${SRC}"
"${PY}" setup.py install --prefix="${ENV_PREFIX}"

# 运行时能找到 libOrbbecSDK.so
mkdir -p "${ENV_PREFIX}/lib"
cp -a "${SRC}/sdk/lib/linux_x64/"*.so* "${ENV_PREFIX}/lib/" || true
cp -a "${SRC}/install/lib/"*.so* "${ENV_PREFIX}/lib/" 2>/dev/null || true

echo ">>> 验证 import pyorbbecsdk"
LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${SRC}/sdk/lib/linux_x64:${LD_LIBRARY_PATH:-}" \
  "${PY}" -c "import pyorbbecsdk as s; print('pyorbbecsdk OK', s.get_version() if hasattr(s,'get_version') else s)"
