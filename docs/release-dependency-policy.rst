
Release and dependency policy
=============================

.. role:: bash(code)
   :language: bash

Release and versioning policy
-----------------------------

Firecrown is following the practice of `semantic versioning <https://semver.org>`_.
In brief, this means that Firecrown version numbers will be of the form `x.y.z`.
The *major version*, `x`, will be incremented whenever a change is made that is not
backwards-compatible.
The *minor version*, `y`, will be incremented when new features are added in a
backwards-compatible fashion.
The *point release version*, `z`, will be incremented for bug fixes that introduce no
new functionality (and are also backwards-compatible).
Firecrown development will hold to these policies for all tagged releases deployed to
`Conda Forge <https://anaconda.org/conda-forge/firecrown>`_.

Firecrown is currently under rapid development.
It will remain so for the forseeable future.
In order to facilitate this rapid development, it is important to be able to resolve
pull requests in a timely fashion.
However, it is also important that we be able to maintain stability in released
interfaces, in accordance with our versioning policy.

To best statisfy both requirements, the

d
To run the current version, one needs to *build* (but not yet install) Firecrown.
In this directory, run:

    python3 setup.py build

This will put modules into the subdirectory :bash:`build/lib`.
Set the environment variable :bash:`FIRECROWN_DIR` to the full path to that directory.
Set the environment variable :bash:`FIRECROWN_EXAMPLES_DIR` to be the full path to the :bash:`examples` subdirectory.

These environment variables are needed by the example :bash:`ini` files.
The directory :bash:`$FIRECROWN_DIR` should also be on :bash:`PYTHONPATH`.

