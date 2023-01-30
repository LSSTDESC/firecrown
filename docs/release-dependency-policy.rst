
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

To best statisfy both requirements, backward compatibility as denoted by the semantic
versioning is promised only for tagged releases.
The head of the `master` branch of `the repository on GitHub <https://github
.com/LSSTDESC/firecrown>`_ is not subject to the same constraint.
This is to allow us to merge pull requests into `master` as quickly as possible,
without taking the extra time necessary to be sure they are following a design
suitable for longer-term stability.

We recommend that projects which do *not* involve new development of Firecrown should
use tagged releases as distributed on Conda Forge.
We recommend that projects which do involve new development of Firecrown work on a
branch started from `master`.
For the most stability, it may be useful to start a branch from the most recent tag
on the `master` branch.
However, before making a pull request from such a branch, it is necessary to first
update the branch from which the pull request will be made to conform with the
current `HEAD` of the `master` branch.