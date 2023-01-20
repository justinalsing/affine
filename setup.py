#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import platform

install_requires = ['tqdm', 'numpy', 'torch']

setup(name='affine',
      version='v0.1',
      description='Parallel affine sampling in torch',
      author='Justin Alsing & Jed Homer',
      url='https://github.com/justinalsing/affine',
      packages=find_packages(),
      install_requires=install_requires)
