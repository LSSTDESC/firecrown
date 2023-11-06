Using Firecrown
===============

.. role:: python(code)
   :language: python




The easiest way to get started using Firecrown is to look at the examples.
These are found in the `examples folder <https://github.com/LSSTDESC/firecrown/tree/master/examples>`_.
There are examples that show use of each of the supported samplers.

Next, we recommend reading the `introductory article <_static/intro_article.html>`_.

CCL Usage Details
-----------------

Firecrown makes use of `pyccl`, and provides components to be used from various Markov Chain Monte Carlo (MCMC) sampling frameworks.

Because it makes use of `pyccl` it needs to instantiate a `pyccl.Cosmology` object for each MCMC sample created by the sampler.
For each MCMC sample we extract *all* of the cosmological parameters from the sampler.
They are all passed to `pyccl` to create the appropriate `pyccl.Cosmology`, using what `pyccl` calls its `calculator mode <https://ccl.readthedocs.io/en/latest/source/notation_and_other_cosmological_conventions.html#the-calculator-mode>`_.
This means we rely upon the sampling framework to supply cosmological parameters.
The sampling framework will also supply other observables such as *distances*, rather than calculating them within `pyccl`.

Sampler Details
---------------

The following parameters must be supplied by the sampling framework.
This means the configuration you supply for the sampling framework must be such that it calculates all these quantities.
This is done differently for each of the supported frameworks.

* For Cobaya, you must supply either the `Camb`, `Classy`, or some other sublcass of `BoltmannBase` as a theory module.
* For CosmoSIS, you must use the `consistency` module (`documented here <https://cosmosis.readthedocs.io/en/latest/reference/standard_library/consistency.html>`_). This means you must also load a Boltzmann calculator.
* For NumCosmo, you must instantiate `MappingNumCosmo` with the models of your choice.

If you have not specified the things required as appropriate for your sampler Firecrown will fail.
The exact failure mode depends on which sampler you are using.

For Cobaya, if you forget to include `theory.camb` (or some other `BoltzmannBase` subclass)  key, Firecrown will ask for objects not produced by Cobaya and it will fail. This failure will happen at configuration time.

For CosmoSIS, if you forget to include the `consistency` module in your pipeline, Firecrown will fail when it looks for one of the quantities that `consistency` would have produced. This will happen during the first sample.

For NumCosmo, if you forget to put the cosmological model in your model set, NumCosmo will fail at its first step.

