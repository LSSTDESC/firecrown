=======================================
NumCosmo: Firecrown Likelihood Best Fit
=======================================

This recipe outlines the process to compute the best fit for Supernova SNIa data using
the Firecrown example with NumCosmo. Ensure that you have already installed
Firecrown and NumCosmo.

.. _generate-experiment-file-sn-srd:

Generate the Experiment File
============================

Use the Firecrown command-line tool to generate the supernova example as a
NumCosmo experiment. The command below downloads the SNIa data and writes the example
into the ``sn_srd_example`` directory. Use ``firecrown examples sn_srd --help`` for
available options.

.. code-block:: bash

   firecrown examples sn_srd sn_srd_example --target-framework numcosmo
   cd sn_srd_example

This command creates a file called ``numcosmo_sn_srd.yaml``, NumCosmo's experiment
file. It contains cosmological and likelihood parameters, modeling choices, and chosen
precision. Next to it are ``numcosmo_sn_srd.builders.yaml``, which defines the
Firecrown model used by the experiment, the SACC data file ``sn_srd.sacc``, and the
likelihood factory ``sn_srd_factory.py``.

This recipe fits the cold dark matter density ``Omegac`` and the supernova absolute
magnitude ``sn_ddf_sample_M``. Supernova distances depend only on the total matter
density, so the baryon density ``Omegab`` is kept fixed. Edit
``numcosmo_sn_srd.yaml`` so that its flag reads:

.. code-block:: yaml

   Omegab-fit: false

.. _compute-bestfit-sn-srd:

Run NumCosmo App
================

Execute the NumCosmo app with the experiment file to compute the best fit. The
following command saves the output in a new ``yaml`` file. Use the ``--help`` flag for
available options.

.. code-block:: bash

   numcosmo run fit numcosmo_sn_srd.yaml --output numcosmo_sn_srd-bestfit.yaml

The best fit is saved in ``numcosmo_sn_srd-bestfit.yaml``. Inspect the file to view the
best-fitting values. This file serves as the starting point for subsequent runs.

Restart Minimization Algorithm (Optional)
=========================================

If dealing with a high-dimensional parameter space or non-converging algorithms,
consider using the `--restart` flag. This restarts the minimization algorithm from the
best-fit found so far.

.. code-block:: bash

   numcosmo run fit numcosmo_sn_srd.yaml --output numcosmo_sn_srd-bestfit.yaml --starting-point numcosmo_sn_srd-bestfit.yaml --restart 1.0e-3 0.0

The command restarts the minimization algorithm if the absolute change in the
likelihood is less than 1.0e-3 and the relative change is less than 0.0.

Note: ``numcosmo_sn_srd-bestfit.yaml`` cannot be used as the experiment file for the
``numcosmo`` command-line tool since it lacks data and model information.

Results Summary
===============

* **Starting Point:**

  * m2lnL: 2037
  * Parameters:

    * sn_ddf_sample_M: -19.30
    * Omegac: 0.250

* **Best Fit:**

  * m2lnL: 3.24293
  * Parameters:

    * sn_ddf_sample_M: -19.460
    * Omegac: 0.260
