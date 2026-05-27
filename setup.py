from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_attention',
    ext_modules=[
        CUDAExtension(
            name='custom_attention',
            sources=['kernels/src/binding.cpp',           # Updated path adjustment
            'kernels/src/cuda_kernel_h100.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': [
                    '-arch=sm_90a',        # sm_90a = H200 native (includes H100 too)
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',           # free — enables Nsight profiling
                    '--expt-relaxed-constexpr',
                ]
            }
        )   # <-- FIX #1: closing paren for CUDAExtension() was missing
    ],      # <-- FIX #2: was a ] closing extra_compile_args, not ext_modules
    cmdclass={'build_ext': BuildExtension}
)