import os
from setuptools import setup, find_packages


def _munge_req(r):
    for sym in ["~", "=", "<", ">", ",", "!", "!"]:
        r = r.split(sym)[0]
    return r


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
    rqs = [_munge_req(r.strip()) for r in fp.readlines()]

setup(
    name='firecrown',
    version=__version__,
    description="DESC Cosmology Constraints Tool",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=['bin/firecrown'],
    install_requires=rqs,
)
