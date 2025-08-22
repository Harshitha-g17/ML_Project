
    
from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="CLV_ML_Project",
    version= "0.1",
    author= "Harshitha",
    packages= find_packages(),
    install_requires = requirements,
)