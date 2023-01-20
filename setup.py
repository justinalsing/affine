#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import platform

install_requires = ['tqdm', 'numpy', 'matplotlib']
try:
    import jax 
except ImportError:
    raise ImportError('Please install JAX manually! See https://jax.readthedocs.io/en/latest/installation.html for more information.')

setup(name='affine',
      version='v0.1',
      description='Parallel affine sampling in JAX.',
      author='Justin Alsing & Jed Homer',
      url='https://github.com/justinalsing/affine',
      packages=find_packages(),
      install_requires=install_requires)