from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "AverageHermitian",
    ["AverageHermitian.pyx"],
#     extra_compile_args=['-fopenmp'],
#     extra_link_args=['-fopenmp'],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    include_dirs = [numpy.get_include()]
)
# python setup2.py build_ext --inplace
