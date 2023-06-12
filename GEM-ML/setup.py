from setuptools import setup, find_packages
import os
from typing import List

def parse_requirements(filename: str) -> List[str]:
    with open(os.path.join(os.path.dirname(__file__), filename)) as req_file:
        return list(req_file)

setup(
    name = 'GEM-ML',
    version = '0.0.1',
    python_requires = ">=3.8",
    #install_requires = parse_requirements("requirements.txt"),
    packages = find_packages(),
    author = "Michael Engel",
    author_email = "m.engel@tum.de"
)