#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup


import os, sys


NAME = 'gadma'
VERSION = '1.0.0'
SUPPORTED_PYTHON_VERSIONS = ['2.5', '2.6', '2.7']


# Check python version
if sys.version[0:3] not in SUPPORTED_PYTHON_VERSIONS:
    sys.stderr.write("Python version " + sys.version[0:3] + " is not supported!\n" +
          "Supported versions are " + ", ".join(SUPPORTED_PYTHON_VERSIONS) + "\n")
    sys.stderr.flush()
    sys.exit(1)


# Create a simple version.py module; less trouble than hard-coding the version
with open(os.path.join('gadma', 'version.py'), 'w') as f:
    f.write('__version__ = version = %r' % VERSION)


# Load up the description from README.rst
with open('README.md') as f:
    DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    author='Ekaterina Noskova',
    author_email='ekaterina.e.noskova@gmail.com',
    url='https://bitbucket.org/noscode/gadma/src/master/',
    description='Genetic Algorithm for Demographic Inference',
    long_description=DESCRIPTION,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development',
    ],
    packages=['gadma'],
    python_requires='>=2.5.*, <=2.7.*',
    include_package_data=True,
    package_data={
        'gadma': ['*.py',  'params_template', 'extra_params_template']
    },
    data_files=[('fs_examples', [os.path.join('fs_examples', 'test.fs')]), ("", ["LICENSE"])],
    install_requires=['numpy>=1.2.0', 'scipy>=0.6.0'],
    entry_points={
        'console_scripts': ['gadma = gadma.core:main',
            'gadma-run_ls_on_boot_data = gadma.run_ls_on_boot_data:main',
            'gadma-get_confidence_intervals = gadma.get_confidence_intervals:main']
    }
)
