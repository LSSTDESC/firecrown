
Firecrown
=========

[![firecrown-ci](https://github.com/LSSTDESC/firecrown/actions/workflows/ci.yml/badge.svg)](https://github.com/LSSTDESC/firecrown/actions/workflows/ci.yml) [![firecrown-nightly](https://github.com/LSSTDESC/firecrown/actions/workflows/nightly.yml/badge.svg)](https://github.com/LSSTDESC/firecrown/actions/workflows/nightly.yml) [![Documentation Status](https://readthedocs.org/projects/firecrown/badge/?version=latest)](https://firecrown.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/LSSTDESC/firecrown/graph/badge.svg?token=LevlRR3wJj)](https://codecov.io/gh/LSSTDESC/firecrown)

Firecrown is a Python package that provides the DESC *framework* to implement likelihoods, as well as specific likelihood implementations.
Firecrown is intended to be usable *from* external statistical analysis tools.

## Documentation

- [Tutorials](https://firecrown.readthedocs.io/en/latest/_static/index.html): the introductory article and step-by-step guides to tomographic bins, two-point statistics, and systematics.
- [Cookbook](https://firecrown.readthedocs.io/en/latest/_static/cookbook/index.html): short recipes, each downloadable as a Jupyter notebook.
- [Main documentation](https://firecrown.readthedocs.io/en/latest/): installation instructions, usage with Cobaya, CosmoSIS, and NumCosmo, and the API reference.
- Examples: the command `firecrown examples` generates example analyses and lists them when run without arguments; see [Using Firecrown](https://firecrown.readthedocs.io/en/latest/basic_usage.html).

## Contributing

Contributions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for the developer workflow,
coding standards, and instructions for running the test suite.
The [CI System Architecture](CONTRIBUTING_ADVANCED.md#ci-system-architecture)
section explains how to add a new branch to the nightly scheduled CI runs.
