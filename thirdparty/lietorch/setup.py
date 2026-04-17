from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp


ROOT = osp.dirname(osp.abspath(__file__))


def _cuda_compile_args(opt_level: str):
    # Avoid pinning the build to legacy architectures only. PyTorch will
    # derive the right targets from the visible GPU or TORCH_CUDA_ARCH_LIST.
    return {"cxx": [opt_level], "nvcc": [opt_level]}

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    author='teedrz',
    packages=['lietorch'],
    ext_modules=[
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'lietorch/include'),
                osp.join(ROOT, 'eigen')],
            sources=[
                'lietorch/src/lietorch.cpp',
                'lietorch/src/lietorch_gpu.cu',
                'lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args=_cuda_compile_args('-O2')),

        CUDAExtension('lietorch_extras',
            sources=[
                'lietorch/extras/altcorr_kernel.cu',
                'lietorch/extras/corr_index_kernel.cu',
                'lietorch/extras/se3_builder.cu',
                'lietorch/extras/se3_inplace_builder.cu',
                'lietorch/extras/se3_solver.cu',
                'lietorch/extras/extras.cpp',
            ],
            extra_compile_args=_cuda_compile_args('-O2')),
    ],
    cmdclass={ 'build_ext': BuildExtension }
)
