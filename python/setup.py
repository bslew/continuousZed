#!/usr/bin/env python

# from distutils.core import setup, Extension
from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
os.environ["CC"] = "c++" 
os.environ["CXX"] = "c++"


reqired_packages=[
    'numpy',
    'matplotlib',
    'statsmodels',
    ]

setup(name='RT32continuousPointing',
      version='1.0',
      description='self-updating ZD corrections for RT-32',
      long_description=read('../readme.md'),
      long_description_content_type='text/markdown',
      author='Bartosz Lew',
      author_email='bartosz.lew@umk.pl',
      url='https://github.com/bslew/continuousZed',
      install_requires=reqired_packages,
      package_dir = {'': '.'},
      packages = ['RT32continuousPointing',
                  ],
      scripts=['RT32continuousPointing/continuousZed.py',
               ],
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
        ],
     )
