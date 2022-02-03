import os
from setuptools import setup, find_packages


def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))


setup(
    name='eocrops',
    python_requires='>=3.7',
    version='1.0.0',
    description='Wrapper designed for crop monitoring using Earth Observation data.',
    author='Johann Desloires',
    author_email='johann.desloires@gmail.com',
    packages=find_packages(),
    package_data={'eocrops': ['environment.yml']}
)
