from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("Discrete_LDPC_decoding/GF2MatrixMul_c.pyx"),
    include_dirs=[numpy.get_include()]
)