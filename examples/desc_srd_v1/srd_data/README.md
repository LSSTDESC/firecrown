## LSST DESC SRD Data

The scripts in this directory repackage the SRD data into the SACC format. To
run them, follow the following steps.

1. Download the [SRD data](https://zenodo.org/record/1409816) to this directory
   and unpack pack it (e.g., run `tar xzvf` on the tarball).
2. Run `python repackage_srd_data.py`. You will need to have the `srd_models`
   and `sacc` packages installed. The `srd_models` package is part of this
   example and `sacc` can be found on `pypi` and `conda`.

A repackaged version of the data has been added to this repo as
`srd_v1_sacc_data.fits` so that you do not have to run the steps above.
