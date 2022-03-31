import os
from setuptools import setup, find_packages


def _munge_req(r):
    for sym in ["~", "=", "<", ">", ",", "!", "!"]:
        r = r.split(sym)[0]
    return r


__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "firecrown", "version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())

pth = os.path.join(os.path.dirname(os.path.realpath(__file__)), "environment.yml")
rqs = []
with open(pth, "r") as fp:
    start = False
    for line in fp.readlines():
        if line.strip() == "dependencies:":
            start = False
        if start:
            if "- pip:" in line.strip():
                continue
            r = line.strip()[3:].strip()
            rqs.append(_munge_req(r))

setup(
    name="firecrown",
    version=__version__,
    description="DESC Cosmology Likelihood Framework",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    install_requires=rqs,
)
