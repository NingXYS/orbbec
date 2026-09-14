#!/usr/bin/env python3
"""把 GraspNet 的 PointNet2 / knn CUDA 扩展改成兼容 PyTorch 2.x + RTX 40 系。

官方代码残留：
  - tensor.type().is_cuda()（PyTorch 2 中 type() 返回字符串）
  - tensor.data<T>()（应改为 data_ptr<T>()）
  - knn 依赖已删除的 THC.h
  - setup.py 写死 sm_80 / sm_90
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PN2 = ROOT / "third_party" / "graspnet-baseline" / "pointnet2"
KNN = ROOT / "third_party" / "graspnet-baseline" / "knn"


def replace_in_file(path: Path, mapping: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    orig = text
    for a, b in mapping:
        text = text.replace(a, b)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")


def patch_pointnet2() -> None:
    mapping = [
        ("x.type().is_cuda()", "x.is_cuda()"),
        (".type().is_cuda()", ".is_cuda()"),
        (".data<float>()", ".data_ptr<float>()"),
        (".data<int>()", ".data_ptr<int>()"),
        (".data<long>()", ".data_ptr<long>()"),
    ]
    for p in (PN2 / "_ext_src").rglob("*"):
        if p.suffix in {".h", ".hpp", ".cpp", ".cu", ".cuh"}:
            replace_in_file(p, mapping)

    setup = PN2 / "setup.py"
    text = setup.read_text(encoding="utf-8")
    # 去掉写死的 -arch=compute_80，交给 TORCH_CUDA_ARCH_LIST
    text = text.replace('"-arch=compute_80",', "")
    text = text.replace('"-arch=compute_80"', "")
    setup.write_text(text, encoding="utf-8")
    print(f"patched {setup}")


KNN_H = r'''#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = idx.data_ptr<long>();

  if (ref.is_cuda()) {
#ifdef WITH_CUDA
    float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(
        ref_nb * query_nb * sizeof(float));

    for (int b = 0; b < batch; b++)
    {
      knn_device(ref_dev + b * dim * ref_nb, ref_nb,
                 query_dev + b * dim * query_nb, query_nb, dim, k,
                 dist_dev, idx_dev + b * k * query_nb,
                 c10::cuda::getCurrentCUDAStream());
    }
    c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        AT_ERROR("aborting");
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb,
            query_dev + b * dim * query_nb, query_nb, dim, k,
            dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;
}
'''

KNN_CUDA_VISION = r'''#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, cudaStream_t stream);
'''

KNN_SETUP = r'''#!/usr/bin/env python
import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu
    extra_compile_args = {"cxx": []}
    define_macros = []
    extension = CppExtension
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    sources = [os.path.join(extensions_dir, s) if not os.path.isabs(s) else s for s in sources]
    return [
        extension(
            "knn_pytorch.knn_pytorch",
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="knn_pytorch",
    version="0.1",
    description="KNN for GraspNet (PyTorch 2.x)",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
'''


def patch_knn() -> None:
    (KNN / "src" / "knn.h").write_text(KNN_H, encoding="utf-8")
    (KNN / "src" / "cuda" / "vision.h").write_text(KNN_CUDA_VISION, encoding="utf-8")
    (KNN / "setup.py").write_text(KNN_SETUP, encoding="utf-8")
    print("patched knn headers/setup")


def main() -> int:
    patch_pointnet2()
    patch_knn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
