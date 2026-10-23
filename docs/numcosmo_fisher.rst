=====================================
NumCosmo: Firecrown Likelihood Fisher
=====================================

This recipe outlines the process of computing the Fisher matrix for Supernova SNIa data
likelihood using the Firecrown example with NumCosmo. Ensure that you have already
installed Firecrown and NumCosmo.

The starting point for this recipe is the NumCosmo experiment configuration file
``numcosmo_sn_srd.yaml``. In section :ref:`generate-experiment-file-sn-srd`, we show
how to generate this file with ``firecrown examples``. The steps below assume that you
have already generated the experiment file and that you run the commands from the
``sn_srd_example`` directory.

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

   numcosmo run fisher numcosmo_sn_srd.yaml --fisher-type expected

This command produces the output:

.. code-block:: text

    # NcmMSet parameters covariance matrix
    #                                                           -------------------------------
    #          Omegac[03000:01] =  0.25        +/-  0.01315     |  1           |  0.9626      |
    # sn_ddf_sample_M[24000:00] = -19.3        +/-  0.01389     |  0.9626      |  1           |
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

   numcosmo run fisher numcosmo_sn_srd.yaml --fisher-type observed

This command produces the output:

.. code-block:: text

    #                                                           -------------------------------
    #          Omegac[03000:01] =  0.25        +/-  0.006303    |  1           |  0.8616      |
    # sn_ddf_sample_M[24000:00] = -19.3        +/-  0.007398    |  0.8616      |  1           |
    #                                                           -------------------------------

The observed Fisher matrix is less accurate and slower to compute than the expected
Fisher matrix. The output above shows that the observed Fisher matrix does not agree
with the expected Fisher matrix. This is because the observed Fisher matrix is an
estimate that needs to be computed at the maximum likelihood point.

Running the command again but now using the previously computed maximum likelihood
point as the starting point (:ref:`compute-bestfit-sn-srd`):

.. code-block:: bash

   numcosmo run fisher numcosmo_sn_srd.yaml --fisher-type observed --starting-point numcosmo_sn_srd-bestfit.yaml

Resulting in the output:

.. code-block:: text

    #                                                           -------------------------------
    #          Omegac[03000:01] =  0.2597      +/-  0.013       |  1           |  0.9605      |
    # sn_ddf_sample_M[24000:00] = -19.46       +/-  0.01349     |  0.9605      |  1           |
    #                                                           -------------------------------

The observed Fisher matrix now agrees with the expected Fisher matrix. Recomputing the
expected Fisher matrix at the maximum likelihood point will produce a result very
close to the observed Fisher matrix. One notable advantage of the observed Fisher
matrix is its applicability to any likelihood function, providing a versatile option
despite its computational demands.


Output
======

The Fisher matrix commands support the ``--output`` option, allowing you to save the
computed Fisher matrix to a file. For instance:

.. code-block:: bash

   numcosmo run fisher numcosmo_sn_srd.yaml --fisher-type expected --starting-point numcosmo_sn_srd-bestfit.yaml --output numcosmo_sn_srd-bestfit.yaml

Executing this command saves the Fisher matrix as ``numcosmo_sn_srd-bestfit.yaml`` in
the same directory as the input configuration file. A Markov Chain Monte Carlo (MCMC)
run can then be initialized by sampling from the saved matrix.
