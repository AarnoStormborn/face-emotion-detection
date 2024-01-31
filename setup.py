# This file is for setting up the pipeline as a python package
# Not included in the container

from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath:str)->List[str]:
    with open(filepath) as f:
        requirements = [req.strip() for req in f.readlines()]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name='face-emotion-detection',
    version='0.0',
    author='Harman',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'))