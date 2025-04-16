#!/usr/bin/env python3
from setuptools import setup

setup(
    # Set the name so that github correctly tracks reverse dependencies.
    # https://github.com/ctlab/GADMA/network/dependents
    name="gadma",
    use_scm_version={"write_to": "gadma/version.py"},
)
