Welcome to firecrown's documentation!
=====================================

Introduction
------------

Firecrown is a Python package that provides the DESC *framework* to implement
likelihoods, as well as specific likelihood implementations. Firecrown is intended to
be usable *from* external statistical analysis tools.

Currently, it supports Cobaya,  CosmoSIS, and NumCosmo, providing the necessary
classes or modules to allow users the sampling frameworks to call any Firecrown
likelihood from within those samplers.

* `Cobaya <https://github.com/CobayaSampler/cobaya>`_
* `CosmoSIS <https://github.com/joezuntz/cosmosis>`_
* `NumCosmo <https://github.com/NumCosmo/NumCosmo>`_

It can also be used as a library in other contexts, and so the installation of
Firecrown does not *require* the installation of a sampler.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: getstart

   Installation Quickstart<install_quick.rst>
   Using Firecrown<basic_usage.rst>


.. toctree::
   :maxdepth: 1
   :caption: Developing with Firecrown
   :name: devnote

   Release and dependency policy<release-dependency-policy.rst>
   Developer Notes<dev-notes.rst>
   Contributing<contrib.rst>
   Class Diagrams<diagrams.rst>

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: apiref

   API Documentation<api.rst>

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

