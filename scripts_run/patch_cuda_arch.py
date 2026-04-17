from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]


TARGET_FILES = {
    REPO_ROOT / "setup.py": dedent(
        """\
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
        """
    ),
    REPO_ROOT / "thirdparty" / "lietorch" / "setup.py": dedent(
        """\
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
        """
    ),
}


def main() -> None:
    for path, desired in TARGET_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        current = path.read_text()
        if current == desired:
            print(f"already patched: {path.relative_to(REPO_ROOT)}")
            continue

        path.write_text(desired)
        print(f"patched: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
