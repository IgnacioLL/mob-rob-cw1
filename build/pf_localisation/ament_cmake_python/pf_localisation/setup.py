from setuptools import find_packages
from setuptools import setup

setup(
    name='pf_localisation',
    version='3.0.0',
    packages=find_packages(
        include=('pf_localisation', 'pf_localisation.*')),
)
