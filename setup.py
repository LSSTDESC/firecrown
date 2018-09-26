from setuptools import setup, find_packages

scripts = ['bin/falcon']

setup(
    name='falcon',
    description="DESC Parameter Estimation",
    author="DESC",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    install_requires=['numpy>=1.15', 'pyccl', 'click'],
)
