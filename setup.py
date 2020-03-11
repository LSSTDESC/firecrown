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

setup(
    name='firecrown',
    version=__version__,
    description="DESC Cosmology Constraints Tool",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=[
        'pyccl', 'click', 'numpy',
        'scipy', 'pandas', 'pyyaml', 'jinja2'],
)
