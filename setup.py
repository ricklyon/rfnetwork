# setup.py

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build

import numpy as np
import glob
import subprocess
import os
from pathlib import Path
import setuptools
import warnings


def find_cuda_lib() -> Path:
    """
    Return the path to the CUDA installation. 
    Returns None if a CUDA installation cannot be found. Ensure the CUDA_HOME environment variable is 
    set to your CUDA install directory.
    """
    # explicit env override
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    # ask nvcc where it lives
    if cuda_home is None:
        try:
            nvcc_path = subprocess.check_output(
                ["which", "nvcc"], stderr=subprocess.DEVNULL
            ).decode().strip()
            # nvcc lives at <cuda_home>/bin/nvcc
            cuda_home = str(Path(nvcc_path).parent.parent)
        except subprocess.CalledProcessError:
            return None
        
    cuda_home = Path(cuda_home)
    nvcc      = cuda_home / "bin" / "nvcc"

    if not nvcc.exists():
        return None
    
    return cuda_home / "lib64"

cuda_path = find_cuda_lib()


class build(_build):
    sub_commands = _build.sub_commands + [('build_cuda', None)]


class build_cuda(setuptools.Command):
    
    # build .cu files before compiling .cpp files
    def initialize_options(self):

        # skip if CUDA is not installed
        if cuda_path is None:
            warnings.warn("CUDA installation not found. Unable to build GPU solver from source.")
            return
        
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

core_cpp = [
    "rfnetwork/core/src/postprocess.cpp",
    "rfnetwork/core/src/core_binding.cpp",
    "rfnetwork/core/src/connect.cpp",
    "rfnetwork/core/src/solver.cpp"
]

cuda_cpp = [
    "rfnetwork/core/src/cuda_binding.cpp",
    "rfnetwork/core/src/solver.cpp"
 ]

core_ext = Extension(
    name="rfnetwork.core.core_func",
    sources=core_cpp,
    include_dirs=["rfnetwork/core/inc", "rfnetwork/core/lib/eigen", np.get_include()],
    optional=False,
)

cuda_ext = Extension(
    name="rfnetwork.core.cuda_func",
    sources=cuda_cpp,
    include_dirs=["rfnetwork/core/inc", "rfnetwork/core/lib/eigen", np.get_include()],
    optional=False,
    extra_objects=["rfnetwork/core/solver_cu.o"], # add cuda objects to the linker
    libraries=["cudart"],  # include cuda runtime in linker
    library_dirs=[str(cuda_path)],
    runtime_library_dirs=[str(cuda_path)],
    define_macros=[
        ("CUDA_AVAILABLE", None),      # equivalent to -DCUDA_AVAILABLE
    ]
)

# only compile CUDA solver code if nvcc is available
ext_modules = [core_ext] if cuda_path is None else [cuda_ext, core_ext]
        
setup(
    ext_modules=ext_modules,
    # https://stackoverflow.com/questions/20194565/running-custom-setuptools-build-during-install
    cmdclass={
        'build': build,
        'build_cuda': build_cuda,
    },
)
