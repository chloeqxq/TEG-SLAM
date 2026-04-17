from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))


def _cuda_compile_args(opt_level: str):
    # Let PyTorch pick the right arch flags from the current machine or
    # TORCH_CUDA_ARCH_LIST instead of hardcoding older SM targets only.
    return {"cxx": [opt_level], "nvcc": [opt_level]}

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
            sources=[
                'src/lib/droid.cpp',
                'src/lib/droid_kernels.cu',
                'src/lib/correlation_kernels.cu',
                'src/lib/altcorr_kernel.cu',
            ],
            extra_compile_args=_cuda_compile_args('-O3')),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
