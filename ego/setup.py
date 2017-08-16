# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:06:46 2013

@author: wangronin
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

name = 'hello'
setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension(name, [name + ".pyx"],
      include_dirs=[numpy.get_include()])]
)
