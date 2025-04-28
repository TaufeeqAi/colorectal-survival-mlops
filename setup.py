from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Colorectal-MLOps",
    version="0.1",
    author="Taufeeq Ahmad",
    packages=find_packages(),
    install_requires = requirements,
)