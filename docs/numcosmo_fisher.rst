=====================================
NumCosmo: Firecrown Likelihood Fisher
=====================================

This recipe outlines the process of computing the Fisher matrix for Supernova SNIa data
likelihood using the Firecrown example with NumCosmo. Ensure that you have already
installed Firecrown, CosmoSIS, and NumCosmo and set the environment variable
``FIRECROWN_DIR`` to the location of the Firecrown installation.

The starting point for this recipe is the NumCosmo experiment configuration file
``sn_srd.yaml``. In section :ref:`convert-configuration-file-sn-srd`, we show how to
convert this file to a Firecrown configuration file. The steps below assume that you
have already converted the configuration file.

Computing the Fisher Matrix
===========================

The next step is to compute the Fisher matrix using NumCosmo's command line tool. There
are two options for the command ``numcosmo run fisher`` that can be used to compute the
Fisher matrix:

Expected Fisher Matrix
----------------------

This option computes the Fisher matrix using the analytic formula for the expected 
Fisher matrix. This requires a Gaussian or Poisson likelihood. The command to compute
the Fisher matrix is:

.. code-block:: bash

   cd $FIRECROWN_DIR/examples/srd_sn
   numcosmo run fisher sn_srd.yaml --fisher-type expected

This command produces the output:

.. code-block:: text

    # NcmMSet parameters covariance matrix
    #                                                           -------------------------------
    # sn_ddf_sample_M[00000:00] = -19.3        +/-  0.01389     |  1           |  0.9626      |
    #          Omegac[04000:01] =  0.26        +/-  0.01315     |  0.9626      |  1           |
    #                                                           -------------------------------

When possible, the expected Fisher matrix is the preferred option as it is faster, more
accurate, and can be computed at any point in the parameter space.

Observed Fisher Matrix
----------------------

This option computes the Fisher matrix using the observed Fisher matrix. This option
computes an estimate of the Fisher matrix using the observed data. It computes the
Hessian of the likelihood and therefore should be computed at the maximum likelihood
point. The command to compute the Fisher matrix is:

.. code-block:: bash

   cd $FIRECROWN_DIR/examples/srd_sn
   numcosmo run fisher sn_srd.yaml --fisher-type observed

This command produces the output:

.. code-block:: text

    #                                                           -------------------------------
    # sn_ddf_sample_M[00000:00] = -19.3        +/-  0.008047    |  1           |  0.8837      |
    #          Omegac[04000:01] =  0.26        +/-  0.006992    |  0.8837      |  1           |
    #                                                           -------------------------------

The observed Fisher matrix is less accurate and slower to compute than the expected
Fisher matrix. The output above shows that the observed Fisher matrix does not agree
with the expected Fisher matrix. This is because the observed Fisher matrix is an
estimate that needs to be computed at the maximum likelihood point.

Running the command again but now using the previously computed maximum likelihood
point as the starting point (:ref:`compute-bestfit-sn-srd`):

.. code-block:: bash

   numcosmo run fisher sn_srd.yaml --fisher-type observed --starting-point sn_srd-bestfit.yaml

Resulting in the output:

.. code-block:: text

    #                                                           -------------------------------
    # sn_ddf_sample_M[00000:00] = -19.43       +/-  0.01396     |  1           |  0.9629      |
    #          Omegac[04000:01] =  0.2654      +/-  0.01332     |  0.9629      |  1           |
    #                                                           -------------------------------

The observed Fisher matrix now agrees with the expected Fisher matrix. Recomputing the
expected Fisher matrix at the maximum likelihood point will produce almost exactly the
same result as the observed Fisher matrix. One notable advantage of the observed Fisher
matrix is its applicability to any likelihood function, providing a versatile option 
despite its computational demands.


Output
======

The Fisher matrix commands support the ``--output`` option, allowing you to save the
computed Fisher matrix to a file. For instance:

.. code-block:: bash

   numcosmo run fisher sn_srd.yaml --fisher-type expected --starting-point sn_srd-bestfit.yaml --output sn_srd-bestfit.yaml

Executing this command saves the Fisher matrix as ``sn_srd-bestfit.yaml`` in the same
directory as the input configuration file. This feature proves useful for storing the
Fisher matrix for future use, such as initializing a Markov Chain Monte Carlo (MCMC)
run by sampling from the saved matrix.
