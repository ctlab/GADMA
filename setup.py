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

# for dical
import urllib.request
import tarfile
import glob


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
                'dadi', 'jpype1']

data_files = [["gadma", ["gadma/test.fs"]], ("", ["LICENSE"])]

# Load dical2 or use the saved version
def unarchive(tar_gz_archive):
    tar = tarfile.open(tar_gz_archive, "r:gz")
    tar.extractall()
    names = tar.getnames()
    tar.close()
    return names

dical2_url = "https://sourceforge.net/projects/dical2/files/latest/download?source=files"
dical2_saved = os.path.join(".", "diCal2_2_0_5.tar.gz")
dical2_download = os.path.join(".", "diCal2_LATEST.tar.gz")
try:
    urllib.request.urlretrieve(dical2_url, filename=dical2_download)
    success = True
except Exception as e:
    success = False
    print(f"Dical2 download failed due to the following exception: {e}")

if not os.path.exists(dical2_download):
    dical2_download = dical2_saved
    print("Dical2 was taken from saved release.")
    success = False

if success:
    try:
        tar_names = unarchive(dical2_download)
    except:
        print("Dical unarchivement failed.")
        success = False

if not success:
    try:
        tar_names = unarchive(dical2_saved)
        print("Dical2 was taken from saved release.")
        success = True
    except Exception as e:
        print(f"Dical installation failed: {e}. Please try to do it manually. "
              "Path to dical2 could be set in gadma/dical2_path.py file")
        success = False
# write our information to gadma files before installation
if success:
    dical_name = None
    for name in tar_names:
        if name.startswith("diCal2_"):
            if os.path.isdir(os.path.join(".", name)):
                if os.path.exists(os.path.join(".", name, "diCal2.jar")):
                    dical_name = name
    if dical_name is not None:
        with open(os.path.join("gadma", "dical2_path.py"), 'w') as fl:
            fl.write("# Generated automatically from setup.py\n")
            fl.write(f"import os\n\n"
                     f"dical2_path = os.path.abspath(\n"
                     f"    os.path.join(os.path.dirname(__file__)),\n"
                     f"    '..',\n"
                     f"    '{dical_name}'\n"
                     f")\n")
        # add all files to data_files
#        data_files.append(("", []))
        for (dirpath, dirnames, filenames) in os.walk(dical_name):
            filenames = [os.path.join(dirpath, f_name) for f_name in filenames]
            data_files.append((dirpath, filenames))

# Begin installation
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
        'gadma.cli': ['*.py',  'params_template', 'extra_params_template', 'test_settings'],
    },
    data_files=data_files,
    install_requires=requirements,
    entry_points={
            'console_scripts': ['gadma = gadma.core:main',
            'gadma-run_ls_on_boot_data = gadma.run_ls_on_boot_data:main',
            'gadma-get_confidence_intervals = gadma.get_confidence_intervals:main']
    },
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "gadma/version.py", "local_scheme": "no-local-version"},
)
