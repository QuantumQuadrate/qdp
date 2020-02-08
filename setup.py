# -*- coding: utf8 -*-
#
# This file were created by Python Boilerplate. Use Python Boilerplate to start
# simple, usable and best-practices compliant Python projects.
#
# Learn more about it at: http://github.com/fabiommendes/python-boilerplate/
#

import os

from setuptools import setup, find_packages

# Meta information
version = open('VERSION').read().strip()
dirname = os.path.dirname(__file__)

# Save version and author to __meta__.py
path = os.path.join(dirname, 'QDP', '__meta__.py')
data = '''# Automatically created. Please do not edit.
__version__ = u'%s'
__author__ = u'Matthew Ebert'
''' % version
with open(path, 'wb') as F:
    F.write(data.encode())

setup(
    # Basic info
    name='QDP',
    version=version,
    author='Matthew Ebert',
    author_email='mfe5003@gmail.com',
    url='https://github.com/mfe5003/qdp',
    description='Quantum Data Processing',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
    ],

    # Packages and depencies
    packages=find_packages('qdp'),
    install_requires=[
        'h5py==2.7.1',
        'numpy==1.13.3',
        'scipy==1.0.0',
        'six==1.11.0',
        'scikit-learn==0.19.1',
    ],
    extras_require={
        'dev': [
            'manuel',
            'pytest',
            'pytest-cov',
            'coverage',
            'mock',
        ],
    },

    # Other configurations
    platforms='any',
)
