# setup.py

from setuptools import Extension, find_packages, setup
import numpy as np
import glob
import subprocess
import os
from pathlib import Path

# compile .cu code before calling pip install:
# nvcc -Xcompiler -fPIC -I rfnetwork/core/inc rfnetwork/core/src/solver.cu -c -o rfnetwork/core/solver_cu.o
# TODO: compile this automatically in the Extension module.

def find_cuda() -> dict:
    """
    Return a dict with keys: home, include, lib64, nvcc.
    Raises RuntimeError if CUDA cannot be located.
    """
    # 1. Honour explicit env override
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    # 2. Ask nvcc where it lives
    if cuda_home is None:
        try:
            nvcc_path = subprocess.check_output(
                ["which", "nvcc"], stderr=subprocess.DEVNULL
            ).decode().strip()
            # nvcc lives at <cuda_home>/bin/nvcc
            cuda_home = str(Path(nvcc_path).parent.parent)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "CUDA toolkit not found. "
                "Set the CUDA_HOME environment variable to your CUDA install directory, "
            )

    cuda_home = Path(cuda_home)
    nvcc      = cuda_home / "bin" / "nvcc"

    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found at {nvcc}")

    return {
        "home"   : str(cuda_home),
        "include": str(cuda_home / "include"),
        "lib64"  : str(cuda_home / "lib64"),
        "nvcc"   : str(nvcc),
    }

cuda_path = find_cuda()


setup(
    ext_modules=[
        Extension(
            name="rfnetwork.core.core_func",
            sources=glob.glob("rfnetwork/core/src/*.cpp"),
            include_dirs=["rfnetwork/core/inc", "rfnetwork/core/lib/eigen", np.get_include()],
            optional=False,
            extra_objects=["rfnetwork/core/solver_cu.o"],
            libraries=["cudart"],
            library_dirs=[cuda_path["lib64"]],
            runtime_library_dirs=[cuda_path["lib64"]],
        )
    ]
)
