#!/usr/bin/env python

from setuptools import find_packages, setup
from pathlib import Path

LIBRARY_NAME = "mlb-predictions"
VERSION = "0.1.0"
HERE = Path(__file__).absolute().parent

INSTALL_REQUIRES = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name=LIBRARY_NAME,
    version=VERSION,
    author="Elliott Evans",
    author_email="evanselliott1@gmail.com",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
