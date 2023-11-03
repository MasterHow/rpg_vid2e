from setuptools import setup
from Cython.Build import cythonize
import numpy as np  # 添加这行导入

setup(
    ext_modules=cythonize("normalize_image.pyx"),
    include_dirs=[np.get_include()]  # 添加这行
)
