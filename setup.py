import os
from setuptools import setup, find_packages

scripts = ['bin/firecrown']

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "firecrown",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "requirements.txt")
with open(pth, 'r') as fp:
    rqs = [r.strip() for r in fp.readlines()]

setup(
    name='firecrown',
    version=__version__,
    description="DESC Cosmology Constraints Tool",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=rqs,
)
