
Developer Notes
===============

.. role:: bash(code)
   :language: bash

To run the development version, one needs to do an *editable installation* of firecrown.
This will create an entry in the conda environment that knows about the local code.
You must have the `:bash:firecrown_developer` conda environment activated before running this installation.

Note that we use the :bash:`--no-deps` flag to prevent the installation from accidentally taking in any new dependencies through  :bash:`pip`.
If the installation fails because of the lack of a dependency, install that dependency using  :bash:`conda` and not  :bash:`pip`.
If you find a dependency on a package not available through :bash:`conda` please file an issue on the issue tracker.

.. code:: bash
    
    python -m pip install --no-deps --editable ${PWD}

Some of the examples and tests depend on the environment variables :bash:`FIRECROWN_DIR` and :bash:`CSL_DIR`.
These environment variables are defined for you every time you activate the conda environment.
