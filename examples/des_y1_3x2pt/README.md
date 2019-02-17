# DES Y1 3x2pt Analysis

The code here will run a short MCMC chain for a Flat LCDM cosmology using
`emcee`. While in principle you can run this chain serially, you probably
want access to a compute cluster where this chain can be run in parallel.

Before doing anything else, you need to unpack the data with `tar`

```bash
$ tar xzvf des_data.tar.gz
```

To run the code and output the likelihood at the fiducial parameter values,
you can run the following

```bash
$ firecrown compute des_y1_3x2pt.yaml
```

Finally, to run a chain, type

```bash
$ firecrown run-emcee des_y1_3x2pt.yaml
```

## Generating the `firecrown` Inputs

Note, this code below only has to be run if you want to generate the firecrown
inputs from the DES Y1 3x2pt data products. These files are stored in the repo
with tar  at `examples/des_y1_3x2pt/des_data.tar.gz`. You do not have to make
them yourself.

To generate the `firecrown` inputs, first download the DES Y1 3x2pt
data products [here](http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits).

Then run the script

```bash
$ python generate_des_data.py
```
