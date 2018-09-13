from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

os.environ["CC"] = "gcc-7 "

sourcefiles  = ['logistic.pyx', 
                'logisticSGD.cpp',
                'dataset.cpp']

compile_opts = ['-Iinclude/', '-std=c++1z', '-fopenmp']
link_opts = ['-fopenmp']

ext=[Extension('*',
            sourcefiles,
            extra_compile_args=compile_opts,
            extra_link_args=link_opts,
            language='c++')]

setup(
  name='logistic',
  ext_modules=cythonize(ext)
)
