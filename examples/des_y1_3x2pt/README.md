# DES Y1 3x2pt Analysis

The code here will run a short MCMC chain for a Flat LCDM cosmology using
`emcee`. While in principle you can run this chain serially, you probably
want access to a compute cluster where this chain can be run in parallel.

To run the code and output the likelihood at the fiducial parameter values,
you can run the following

```bash
$ firecrown compute des_y1_3x2pt.yaml
```

Finally, to run a chain, type

```bash
$ firecrown run-emcee des_y1_3x2pt.yaml
```

You might need to install the `schwimmbad` modeule if not already there. You can do this via pip:

```bash
pip install schwimmbad
```

## Generating the `firecrown` Inputs

Note, this code below only has to be run if you want to generate the firecrown
input from the DES Y1 3x2pt data products. This file is stored at
`examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits`. You do not have to make
them yourself.

To generate the `firecrown` inputs, first download the DES Y1 3x2pt
data products [here](http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits).

Then run the script

```bash
$ python generate_des_data.py
```
