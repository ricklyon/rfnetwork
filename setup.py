# setup.py

from setuptools import Extension, find_packages, setup
import numpy as np
import glob

setup(
    ext_modules=[
		Extension(
			name="rfnetwork.core.core_func",
			sources=glob.glob("rfnetwork/core/src/*.cpp"),
			include_dirs=["rfnetwork/core/inc", "rfnetwork/core/lib/eigen", np.get_include()],
			optional=False
		)
	]
)
