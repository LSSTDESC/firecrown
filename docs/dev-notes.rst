
Developer Notes
===============

.. role:: bash(code)
   :language: bash

To run the current version, one needs to *build* (but not yet install) Firecrown.
In this directory, run:

    python3 setup.py build

This will put modules into the subdirectory :bash:`build/lib`.
Set the environment variable :bash:`FIRECROWN_DIR` to the full path to that directory.
Set the environment variable :bash:`FIRECROWN_EXAMPLES_DIR` to be the full path to the :bash:`examples` subdirectory.

These environment variables are needed by the example :bash:`ini` files.
The directory :bash:`$FIRECROWN_DIR` should also be on :bash:`PYTHONPATH`.

