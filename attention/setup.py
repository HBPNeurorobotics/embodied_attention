"""
Attention package
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name = "attention",
    version = "0.0.1",
    description = ("Attention package"),
    keywords = "attention, saliency, saccade",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    # Choose your license
    license='GPLv3',
    install_requires=[]
)
