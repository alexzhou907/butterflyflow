from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("inverse_op_cython.pyx", annotate=True),include_dirs=[numpy.get_include()],

    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)
