==========================================
NumCosmo: Firecrown Likelihood Fisher Bias
==========================================

This recipe outlines the process of computing the Fisher bias for Supernova SNIa data
likelihood using the Firecrown example with NumCosmo. Ensure you have already 
installed Firecrown, CosmoSIS, and NumCosmo and set the environment variable 
``FIRECROWN_DIR`` to the location of the Firecrown installation.

The starting point for this recipe is the NumCosmo experiment configuration file 
``sn_srd.yaml``. In section :ref:`convert-configuration-file-sn-srd`, we explain how to
convert this file to a Firecrown configuration file. The steps below assume you have 
already completed this conversion.

Computing the Theory Vector
===========================

The Fisher bias represents the difference between the current parameters and the 
parameters most compatible with a given theory vector. To compute the theory vector, we
first need to calculate it at any point in the parameter space. In this example, we
compute the theory vector at the previously determined best-fit point, obtained in 
:ref:`compute-bestfit-sn-srd`.

Run the following command to compute the theory vector:

.. code-block:: bash

    numcosmo run theory-vector sn_srd.yaml --starting-point sn_srd-bestfit.yaml --output sn_srd-bestfit.yaml

This command updates the file ``sn_srd-bestfit.yaml`` to include the theory vector.

Computing the Fisher Bias
=========================

Now, compute the Fisher bias using the following command:

.. code-block:: bash

    numcosmo run fisher-bias sn_srd.yaml --theory-vector sn_srd-bestfit.yaml

This command produces the shift vector: [-0.126, 0.005].

The shift vector represents the difference between the current parameters and the
parameters most compatible with the theory vector. The parameters in ``sn_srd.yaml``
are:

    * sn_ddf_sample_M: -19.3
    * Omegac: 0.26

While the best-fit parameters are:

    * sn_ddf_sample_M: -19.426
    * Omegac: 0.265

Note that the shift vector precisely reflects the difference between the best-fit
parameters and the parameters in the configuration file. This aligns with expectations,
as we utilized the best-fit point to compute the theory vector.
