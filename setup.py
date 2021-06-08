#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import platform

install_requires = ['tqdm', 'numpy']
if platform.machine() == 'arm64':
    try:
    import tensorflow
  except ImportError:
    raise ImportError('install tensorflow manually')
else:
  install_requires.append('tensorflow>=2.0')

setup(name='affine',
      version='v0.1',
      description='parallel affine sampling in tensorflow',
      author='Justin Alsing',
      url='https://github.com/justinalsing/affine',
      packages=find_packages(),
      install_requires=install_requires)


