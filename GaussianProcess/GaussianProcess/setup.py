# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:06:46 2013

@author: wangronin
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

name = raw_input("The .pyx file name:")

extensions = [Extension(name, [name + ".pyx"])]
setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(extensions)
)
