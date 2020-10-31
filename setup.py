#!/usr/bin/env python3

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


import os, sys


NAME = 'gadma'

VERSION = '2.0.0rc5'
SUPPORTED_PYTHON_VERSIONS = ['3.6', '3.7']


# Check python version
if sys.version[0:3] not in SUPPORTED_PYTHON_VERSIONS:
    sys.stderr.write("Python version " + sys.version[0:3] + " is not supported!\n" +
          "Supported versions are " + ", ".join(SUPPORTED_PYTHON_VERSIONS) + "\n")
    sys.stderr.flush()
    sys.exit(1)


# Create a simple version.py module; less trouble than hard-coding the version
with open(os.path.join('gadma', 'version.py'), 'w') as f:
    f.write('__version__ = %r\nversion = __version__\n' % VERSION)
    f.write('\n# This is a new line that ends the file.\n')


# Load up the description from README.rst
with open('README.md') as f:
    DESCRIPTION = f.read()

requirements = ['numpy', 'scipy', 'matplotlib',
                'Pillow', 'Cython', 'mpmath', 'nlopt', 'ruamel.yaml',
                'dadi']

setup(
    name=NAME,
    version=VERSION,
    author='Ekaterina Noskova',
    author_email='ekaterina.e.noskova@gmail.com',
    url='https://github.com/ctlab/GADMA',
    description='Genetic Algorithm for Demographic Inference',
    long_description=DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
    ],
    packages=find_packages(exclude=['examples', 'tests']),
    include_package_data=True,
    package_data={
        'gadma.cli': ['*.py',  'params_template', 'extra_params_template', 'test_settings']
    },
    data_files=[["gadma", ["gadma/test.fs"]], ("", ["LICENSE"])],
    install_requires=requirements,
    entry_points={
        'console_scripts': ['gadma = gadma.core:main',
            'gadma-run_ls_on_boot_data = gadma.run_ls_on_boot_data:main',
            'gadma-get_confidence_intervals = gadma.get_confidence_intervals:main']
    },
)
