from setuptools import setup, find_packages

scripts = ['bin/nightvision']

setup(
    name='nightvision',
    description="DESC Parameter Estimation for Seeing in the Dark",
    author="DESC",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=['numpy>=1.15', 'pyccl', 'click'],
)
