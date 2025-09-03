# from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize

# setup(ext_modules=cythonize("Bai_toan_buong_dem_cython.pyx"))
setup(ext_modules=cythonize("Scale_image_cython.pyx"))
