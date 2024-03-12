=======================================
NumCosmo: Firecrown Likelihood Best Fit
=======================================

This recipe outlines the process to compute the best fit for Supernova SNIa data using
the Firecrown example with NumCosmo. Ensure that you have already installed 
Firecrown, CosmoSIS, and NumCosmo and set the environment variable ``FIRECROWN_DIR``
to the location of the Firecrown installation.

.. _convert-configuration-file-sn-srd:

Convert Configuration File
==========================

Use the NumCosmo command-line tool to convert the ``sn_srd.ini`` file into a NumCosmo
configuration in a ``yaml`` file. Since this likelihood does not require a 
power-spectrum likelihood, you can use the ``from-cosmosis`` command without any extra
flags `--matter-ps eisenstein_hu` nor `--nonlin-matter-ps halofit`. To mute the output 
of the consistency cosmosis module, use ``--mute-cosmosis``.

.. code-block:: bash

   cd $FIRECROWN_DIR/examples/srd_sn
   numcosmo from-cosmosis sn_srd.ini --mute-cosmosis

This command creates a file called ``sn_srd.yaml``, NumCosmo's experiment file. It 
contains cosmological and likelihood parameters, modeling choices, and chosen precision.

.. _compute-bestfit-sn-srd:

Run NumCosmo App
================

Execute the NumCosmo app with the experiment file to compute the best fit. The 
following command saves the output in a new ``yaml`` file. Use the ``--help`` flag for 
available options.

.. code-block:: bash

   cd $FIRECROWN_DIR/examples/srd_sn
   numcosmo run fit sn_srd.yaml --output sn_srd-bestfit.yaml

The best fit is saved in ``sn_srd-bestfit.yaml``. Inspect the file to view the 
best-fitting values. This file serves as the starting point for subsequent runs.

Restart Minimization Algorithm (Optional)
=========================================

If dealing with a high-dimensional parameter space or non-converging algorithms,
consider using the `--restart` flag. This restarts the minimization algorithm from the
best-fit found so far.

.. code-block:: bash

   cd $FIRECROWN_DIR/examples/srd_sn
   numcosmo run fit sn_srd.yaml --output sn_srd-bestfit.yaml --starting-point sn_srd-bestfit.yaml --restart 1.0e-3 0.0

The command restarts the minimization algorithm if the absolute change in the 
likelihood is less than 1.0e-3 and the relative change is less than 0.0.

Note: ``sn_srd-bestfit.yaml`` cannot be used as the experiment file for the 
``numcosmo`` command-line tool since it lacks data and model information.

Results Summary
===============

* **Starting Point:**

  * m2lnL: 1215
  * Parameters:

    * sn_ddf_sample_M: -19.30
    * Omegac: 0.260

* **Best Fit:**

  * m2lnL: 1.70934
  * Parameters:

    * sn_ddf_sample_M: -19.426
    * Omegac: 0.265
