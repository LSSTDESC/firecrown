# DES Y1 3x2pt Analysis

The DES Y1 3x2pt analysis is a joint analysis of the DES Y1 cosmic shear, galaxy-galaxy
lensing, and galaxy clustering data. In this example, we show how to use the DES Y1
3x2pt data products to calculate the likelihood using `Cobaya`, `CosmoSIS`, and
`NumCosmo`.

## Likelihood factories

Firecrown likelihood can be constructed through user defined factories. In this example:

  - `factory.py`: The FireCrown factory file for the DES Y1 3x2pt analysis.
  - `factory_PT.py`: The FireCrown factory file for the DES Y1 3x2pt analysis with
    perturbation theory.

Firecrown default factories can also be used, in this case they are configure through the
experiment configuration file:

  - `experiment.yaml`: The FireCrown experiment configuration file for the DES Y1 3x2pt
    analysis.
  - `pure_ccl_experiment.yaml`: The FireCrown experiment configuration file for the DES Y1
    3x2pt analysis using the `pure_ccl` mode.
  - `mu_sigma_experiment.yaml`: The FireCrown experiment configuration file for the DES Y1
    3x2pt analysis using the `mu_sigma_isitgr` mode.


## Running cosmosis

For each of the likelihood factories, we provide a CosmoSIS pipeline configuration file:

  - `cosmosis/factory.ini`: The CosmoSIS pipeline configuration file for the DES Y1 3x2pt
    analysis.
  - `cosmosis/factory_PT.ini`: The CosmoSIS pipeline configuration file for the DES Y1 3x2pt
    analysis with perturbation theory.
  - `cosmosis/pure_ccl.ini`: The CosmoSIS pipeline configuration file for the DES Y1 3x2pt
    analysis using the `pure_ccl` mode.
  - `cosmosis/mu_sigma.ini`: The CosmoSIS pipeline configuration file for the DES Y1 3x2pt



## Running Cobaya

The Cobaya file `cobaya_evaluate.yaml` configures Cobaya to use the `des_y1_3x2pt.py` likelihood.
Run it using:

    cobaya-run cobaya_evaluate.yaml

This will produce the output files:

    cobaya_evaluate_output.input.yaml
    cobaya_evaluate_output.updated.yaml
    cobaya_evaluate_output.updated.dill_pickle
    cobaya_evaluate_output.1.txt

## Running CosmoSIS

The pipeline configuration file `des_y1_3x2pt.ini` and the related `des_y1_3x2pt_values.ini` configure CosmoSIS to use the `des_y1_3x2pt.py` likelihood.
Run this using:

    cosmosis des_y1_3x2pt.ini

This will produce the output to the screen, showing the calculated likelihood.
It also creates the directory `des_y1_3x2pt_output` which is populated with the CosmoSIS datablock contents for the generated sample.

## Generating the `firecrown` Inputs

Note, this code below only has to be run if you want to generate the firecrown input from the DES Y1 3x2pt data products.
This file is stored at `examples/des_y1_3x2pt/sacc_data.fits`.
You do not have to make it yourself.

To generate the `firecrown` inputs, first download the DES Y1 3x2pt data products [here](http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits).

Then run the script

```bash
$ python generate_des_data.py
```
