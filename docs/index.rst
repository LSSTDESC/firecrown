Welcome to firecrown's documentation!
=====================================

Introduction
------------

Firecrown is a Python package that provides the DESC *framework* to implement
likelihoods, as well as specific likelihood implementations. Firecrown is intended to
be usable *from* external statistical analysis tools.

Currently, it supports Cobaya,  CosmoSIS, and NumCosmo, providing the necessary classes
or modules to allow users the sampling frameworks to call any Firecrown likelihood from
within those samplers.

* `Cobaya <https://github.com/CobayaSampler/cobaya>`_
* `CosmoSIS <https://github.com/joezuntz/cosmosis>`_
* `NumCosmo <https://github.com/NumCosmo/NumCosmo>`_

It can also be used as a library in other contexts, and so the installation of
Firecrown does not *require* the installation of a sampler.

.. note::
   Before installing Firecrown, we recommend visiting the `tutorial site <_static/index.html>`_.
   This site serves as a comprehensive resource for Firecrown, offering both pre-installation information and post-installation guidance. 
   It covers essential topics such as the general concepts behind Firecrown, installation instructions, and usage guidelines.

Data Format
-----------

Firecrown uses the `SACC <https://sacc.readthedocs.io/>`_ (Save All Correlations and
Covariances) format to organize and manage measurements and their associated metadata.
SACC provides a standardized way to store, read, and validate cosmological survey data
across different analysis pipelines.

The SACC format is essential for Firecrown's operation because it:

- Organizes measurements from multiple tracers (tomographic bins)
- Enforces naming conventions for unambiguous interpretation
- Stores measurement covariances and window functions

For detailed information about using SACC files with Firecrown, including naming
conventions, mixed-type measurements, and guidance for fixing convention violations,
see the **SACC Usage** guide.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getstart

   Installation Quickstart<install_quick.rst>
   Using Firecrown<basic_usage.rst>
   SACC Usage<sacc_usage.rst>
   Tutorial Site <tutorial.rst>

.. toctree::
   :maxdepth: 1
   :caption: Developing with Firecrown
   :name: devnote

   Release and dependency policy<release-dependency-policy.rst>
   Developer Notes<dev-notes.rst>
   Contributing<contrib.rst>

.. toctree::
   :maxdepth: 2
   :caption: Cookbook
   :name: cookbook

   NumCosmo <numcosmo_cookbook.rst>
 
.. toctree::
   :maxdepth: 1
   :caption: License
   :name: license

   termcond

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

