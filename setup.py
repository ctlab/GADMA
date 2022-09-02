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

SUPPORTED_PYTHON_VERSIONS = ['3.6', '3.7', '3.8']

# Check python version
if sys.version[0:3] not in SUPPORTED_PYTHON_VERSIONS:
    sys.stderr.write("Python version " + sys.version[0:3] + " is not supported!\n" +
                     "Supported versions are " + ", ".join(SUPPORTED_PYTHON_VERSIONS) + "\n")
    sys.stderr.flush()
    sys.exit(1)

# Load up the description from README.rst
with open('README.md') as f:
    DESCRIPTION = f.read()

requirements = ['numpy', 'scipy', 'matplotlib',
                'Pillow', 'Cython', 'mpmath', 'nlopt', 'ruamel.yaml',
                'dadi', 'scikit-allel', "tensorflow", "keras", "sklearn"]

saved_ml_models = []
ml_models_path = os.path.join("gadma", "utils", "saved_ml_models")
for filename in os.listdir(ml_models_path):
    path = os.path.join(ml_models_path, filename)
    write_path = os.path.join("saved_ml_models", filename)
    if os.path.isdir(path):
        for subfilename in os.listdir(path):
            saved_ml_models.append(os.path.join(write_path, subfilename))
    else:
        saved_ml_models.append(write_path)

print(saved_ml_models)

setup(
    name=NAME,
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
        'gadma.cli': ['*.py', 'params_template', 'extra_params_template', 'test_settings'],
        'gadma.utils': ["*.py", *saved_ml_models]
    },
    data_files=[["gadma", ["gadma/test.fs"]], ("", ["LICENSE"])],
    install_requires=requirements,
    entry_points={
        'console_scripts': ['gadma = gadma.core:main',
                            'gadma-run_ls_on_boot_data = gadma.run_ls_on_boot_data:main',
                            'gadma-get_confidence_intervals = gadma.get_confidence_intervals:main',
                            'gadma-get_confidence_intervals_for_ld = gadma.get_confidence_intervals_for_ld:main',
                            'gadma-precompute_ld_data = gadma.precompute_ld_data:main']
    },
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "gadma/version.py", "local_scheme": "no-local-version"},
)
