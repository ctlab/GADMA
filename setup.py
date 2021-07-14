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
import shutil


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

# Other setup options that could be changed during next steps
requirements = ['numpy', 'scipy', 'matplotlib',
                'Pillow', 'Cython', 'mpmath', 'nlopt', 'ruamel.yaml',
                'dadi', 'jpype1']

data_files = [["gadma", ["gadma/test.fs"]], ("", ["LICENSE"])]

package_data = {
    'gadma': ["CatchSystemExit.jar"],
    'gadma.cli': ['*.py',  'params_template', 'extra_params_template', 'test_settings'],
}


# Load dical2 or use the saved version
def unarchive(tar_gz_archive):
    tar = tarfile.open(tar_gz_archive, "r:gz")
    dirname = os.path.commonprefix(tar.getnames())
    if not os.path.exists(dirname):
        tar.extractall(path="gadma")
    tar.close()
    return os.path.join("gadma", dirname)

#dical2_url = "https://sourceforge.net/projects/dical2/files/latest/download?source=files"
dical2_saved = os.path.join(".", "diCal2_2_0_5.tar.gz")
dical2_download = os.path.join(".", "diCal2_LATEST.tar.gz")
success = False
#try:
#    urllib.request.urlretrieve(dical2_url, filename=dical2_download)
#    success = True
#except Exception as e:
#    success = False
#    print(f"Dical2 download failed due to the following exception: {e}")

if not os.path.exists(dical2_download):
    dical2_download = dical2_saved
    print("Dical2 was taken from saved release.")
    success = False

# path to keep unarchieved diCal2 version
data_path = None

if success:
    try:
        data_path = unarchive(dical2_download)
    except:
        print("Dical unarchivement failed.")
        success = False

if not success:
    try:
        data_path = unarchive(dical2_saved)
        print("Dical2 was taken from saved release.")
        success = True
    except Exception as e:
        print(f"Dical installation failed: {e}. GADMA will not have diCal "
              "engine. Please try to do it manually. "
              "Please\n\t1a) Download archive with diCal2 from codeforce: "
              "https://sourceforge.net/projects/dical2/\n"
              "\t2), unarchive tar.gz file to `gadma` directory: "
              "tar -xzf rebol.tar.gz -C gadma/\n\t3) Run this install again.\n"
              "Also Path to dical2 could be set in gadma/dical2_path.py file.")
        success = False

# write our information to gadma files before installation
if success:
    assert data_path is not None
    path_to_dical_jar = os.path.join(data_path, "diCal2.jar")
    if os.path.exists(path_to_dical_jar):
        with open(os.path.join("gadma", "dical2_path.py"), 'w') as fl:
            fl.write("# Generated automatically from setup.py\n")
            fl.write(f"import os\n\n"
                     f"dical2_path = os.path.abspath(\n"
                     f"    os.path.join(\n"
                     f"        os.path.dirname(os.path.dirname(__file__)),\n"
                     f"        '{data_path}'\n"
                     f"    )\n"
                     f")\n")
        # save data_path to gadma package_data
        package_data['gadma'].append(os.path.join(data_path, "*"))
    else:
        print(f"diCal2.jar was not found in diCal2 directory ({data_path}), "
              "Please remove or rename the directory if it is not unarchieved "
              "folder.")

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
    package_data=package_data,
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
