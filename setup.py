from setuptools import setup, find_packages

scripts = ['bin/firecrown']

setup(
    name='firecrown',
    description="DESC Cosmology Constraints Tool",
    author="DESC Team",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=['cosmosis-standalone', 'pyccl'],
)
