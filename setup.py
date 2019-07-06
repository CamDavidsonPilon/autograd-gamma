#!/usr/bin/env python

from distutils.core import setup

setup(name='autograd-gamma',
      version='0.1',
      description='Autograd compatible approximations to the gamma family of functions',
      author='Cameron Davidson-Pilon',
      author_email='cam.davidson.pilon@gmail.com',
      url='https://github.com/CamDavidsonPilon/autograd-gamma',
      packages=['autograd_gamma'],
      keywords=['autograd', 'gamma', 'incomplete gamma function'],
      classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
      ],
      install_requires=[
       'autograd>=1.2.0',
       'scipy>=1.2.0',
     ]
 )