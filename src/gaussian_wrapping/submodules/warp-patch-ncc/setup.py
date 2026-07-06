from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="warp_patch_ncc",
    packages=['warp_patch_ncc'],
    ext_modules=[
        CUDAExtension(
            name="warp_patch_ncc._C",
            sources=[
            "cuda_warp_patch_ncc/warp_patch_ncc_impl.cu",
            "warp_patch_ncc.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-O3",
                "--use_fast_math"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)