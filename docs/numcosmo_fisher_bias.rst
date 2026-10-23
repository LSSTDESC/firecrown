==========================================
NumCosmo: Firecrown Likelihood Fisher Bias
==========================================

This recipe outlines the process of computing the Fisher bias for Supernova SNIa data
likelihood using the Firecrown example with NumCosmo. Ensure you have already
installed Firecrown and NumCosmo.

The starting point for this recipe is the NumCosmo experiment configuration file
``numcosmo_sn_srd.yaml``. In section :ref:`generate-experiment-file-sn-srd`, we explain
how to generate this file with ``firecrown examples``. The steps below assume you have
already generated the experiment file and that you run the commands from the
``sn_srd_example`` directory.

Computing the Theory Vector
===========================

The Fisher bias represents the difference between the current parameters and the
parameters most compatible with a given theory vector. To compute the theory vector, we
first need to calculate it at any point in the parameter space. In this example, we
compute the theory vector at the previously determined best-fit point, obtained in
:ref:`compute-bestfit-sn-srd`.

Run the following command to compute the theory vector:

.. code-block:: bash

    numcosmo run theory-vector numcosmo_sn_srd.yaml --starting-point numcosmo_sn_srd-bestfit.yaml --output numcosmo_sn_srd-bestfit.yaml

This command updates the file ``numcosmo_sn_srd-bestfit.yaml`` to include the theory
vector.

Computing the Fisher Bias
=========================

Now, compute the Fisher bias using the following command:

.. code-block:: bash

    numcosmo run fisher-bias numcosmo_sn_srd.yaml --theory-vector numcosmo_sn_srd-bestfit.yaml

This command produces the shift vector: [0.010, -0.160].

The shift vector represents the difference between the current parameters and the
parameters most compatible with the theory vector. The parameters in
``numcosmo_sn_srd.yaml`` are:

    * Omegac: 0.25
    * sn_ddf_sample_M: -19.3

While the best-fit parameters are:

    * Omegac: 0.260
    * sn_ddf_sample_M: -19.460

Note that the shift vector precisely reflects the difference between the best-fit
parameters and the parameters in the configuration file. This aligns with expectations,
as we utilized the best-fit point to compute the theory vector.
