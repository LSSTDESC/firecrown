:orphan:

=====================================
Using Firecrown in non-developer mode
=====================================

A Firecrown installed from conda-forge, as described in :doc:`../non_developer_installation`, generates the same examples as a developer installation.
The command below writes the DES Y1 3x2pt example into the directory ``des_y1_example``: the SACC data file, the likelihood factory ``des_y1_3x2pt_factory.py``, and the configuration files for the framework chosen with ``--target-framework`` (``cosmosis``, ``cobaya``, or ``numcosmo``).

.. code:: bash

    firecrown examples des_y1_3x2pt des_y1_example --target-framework cosmosis

Run the framework from that directory, with the command for the chosen framework:

.. code:: bash

    cd des_y1_example
    cosmosis cosmosis_des_y1_3x2pt.ini
    cobaya-run cobaya_des_y1_3x2pt.yaml
    numcosmo run test numcosmo_des_y1_3x2pt.yaml

The generated files refer to each other by relative paths and do not record where Firecrown is installed.
The CosmoSIS configuration loads the Firecrown likelihood through ``firecrown_cosmosis.py``, which imports it from the Python environment that runs CosmoSIS.
The directory can therefore be moved, or copied to another machine with Firecrown installed, and run there.
CosmoSIS also needs the environment variable ``CSL_DIR``, set as described in :doc:`../non_developer_installation`.

``firecrown examples`` lists the available examples, and ``firecrown examples des_y1_3x2pt --help`` describes the options of this one.
The likelihood factory written into the directory is a starting point for a likelihood of your own.
