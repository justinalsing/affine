#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='affine',
      version='v0.1',
      description='parallel affine sampling in tensorflow',
      author='Justin Alsing',
      url='https://github.com/justinalsing/affine',
      packages=find_packages(),
      install_requires=[
	        "numpy",
          "tqdm>=4.41.1",
          "tensorflow>=2.3.0"
      ])
