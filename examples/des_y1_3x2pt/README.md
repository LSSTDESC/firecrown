# DES Y1 3x2pt Analysis

The DES Y1 3x2pt analysis of the Dark Energy Survey (DES) Year 1 (Y1) data, focusing on
**cosmic shear**, **galaxy-galaxy lensing**, and **galaxy clustering**. This repository
demonstrates how to use the DES Y1 3x2pt data products to compute the likelihood with
the following tools:

- **Cobaya**  
- **CosmoSIS**  
- **NumCosmo**  

## Likelihood Factories

The **Firecrown** likelihoods can be customized through user-defined factories. In this
example, the following factory files are provided:

- `factory.py`: Factory for the DES Y1 3x2pt analysis.
- `factory_PT.py`: Factory for the DES Y1 3x2pt analysis using perturbation theory.

Alternatively, Firecrown’s **default factories** can be configured using experiment
configuration files:

- `experiment.yaml`: Standard configuration for the DES Y1 3x2pt analysis.
- `pure_ccl_experiment.yaml`: Configuration for the DES Y1 3x2pt analysis in
  **pure_ccl** mode.
- `mu_sigma_experiment.yaml`: Configuration for **mu_sigma_isitgr** mode.

## Running CosmoSIS

For each likelihood factory, a corresponding **CosmoSIS** pipeline configuration file is
provided:

- `cosmosis/factory.ini`: Pipeline for the standard DES Y1 3x2pt analysis.
- `cosmosis/factory_PT.ini`: Pipeline for the analysis with perturbation theory.
- `cosmosis/pure_ccl.ini`: Pipeline for the analysis in **pure CCL** mode.
- `cosmosis/mu_sigma.ini`: Pipeline for the analysis in **mu_sigma_isitgr** mode.

### Example Command

To run the CosmoSIS pipeline, use the following command:

```bash
cosmosis cosmosis/factory.ini
```

## Running Cobaya

For Cobaya, the pipeline configuration files are:

- `cobaya/evaluate.yaml`: Pipeline for the standard DES Y1 3x2pt analysis.
- `cobaya/evaluate_PT.yaml`: Pipeline for the analysis with perturbation theory.
- `cobaya/evaluate_pure_ccl.yaml`: Pipeline for the analysis in **pure CCL** mode.
- `cobaya/evaluate_mu_sigma.yaml`: Pipeline for the analysis in **mu_sigma_isitgr** mode.

### Example Command

To run the Cobaya pipeline, use the following command:

```bash
cobaya-run cobaya/factory.yaml
```

## Generating the `firecrown` Inputs

Note, this code below only has to be run if you want to generate the firecrown input from the DES Y1 3x2pt data products.
This file is stored at `examples/des_y1_3x2pt/sacc_data.fits`.
You do not have to make it yourself.

To generate the `firecrown` inputs, first download the DES Y1 3x2pt data products [here](http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits).

Then run the script

```bash
$ python generate_des_data.py
```
