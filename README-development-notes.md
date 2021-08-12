The CosmoSIS connector is still under development.

To run the current version, one needs to *build* (but not yet install) Firecrown.
In this directory, run:

    python3 setup.py build

This will put modules into the subdirectory `build/lib`.
Set the environment variable `FIRECROWN_DIR` to the full path to that directory.
Set the environment variabel `FIRECROWN_EXAMPLES_DIR` to be the full path to the
`examples` subdirectory.

These environment variables are needed by the example `ini` files.
The directory `$FIRECROWN_DIR` should also be on `PYTHONPATH`.
