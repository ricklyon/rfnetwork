# setup.py

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build

import numpy as np
import glob
import subprocess
import os
from pathlib import Path
import setuptools


def find_cuda() -> dict:
    """
    Return the path to the CUDA installation. 
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
    
    return cuda_home

cuda_path = find_cuda()


class build(_build):
    sub_commands = _build.sub_commands + [('build_cuda', None)]


class build_cuda(setuptools.Command):
    
    # build .cu files before compiling .cpp files
    def initialize_options(self):
        
        result = subprocess.run(
            "nvcc -Xcompiler -fPIC -I rfnetwork/core/inc "
            "rfnetwork/core/src/solver.cu -c "
            "-o rfnetwork/core/solver_cu.o",
            shell=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    def finalize_options(self):
        pass
    
    def run(self):
        pass
        
setup(
    ext_modules=[
        Extension(
            name="rfnetwork.core.core_func",
            sources=glob.glob("rfnetwork/core/src/*.cpp"),
            include_dirs=["rfnetwork/core/inc", "rfnetwork/core/lib/eigen", np.get_include()],
            optional=False,
            extra_objects=["rfnetwork/core/solver_cu.o"], # add cuda objects to the linker
            libraries=["cudart"],  # include cuda runtime in linker
            library_dirs=[str(cuda_path / "lib64")],
            runtime_library_dirs=[str(cuda_path / "lib64")],
        )
    ],
    # https://stackoverflow.com/questions/20194565/running-custom-setuptools-build-during-install
    cmdclass={
        'build': build,
        'build_cuda': build_cuda,
    },
)
