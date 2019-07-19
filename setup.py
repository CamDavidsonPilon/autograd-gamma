#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autograd-gamma",
    version="0.4.0",
    description="Autograd compatible approximations to the gamma family of functions",
    author="Cameron Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    url="https://github.com/CamDavidsonPilon/autograd-gamma",
    keywords=["autograd", "gamma", "incomplete gamma function"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["autograd>=1.2.0", "scipy>=1.2.0"],
    packages=setuptools.find_packages(),
)
