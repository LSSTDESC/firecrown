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

Note that `mu_sigma_experiment.yaml` requires the
[`isitgr`](https://github.com/mishakb/ISiTGR) package to be installed. This package is
not installed by default in the Firecrown environment.

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

Here’s the improved version of your text with better clarity and conciseness:  

---

## Running NumCosmo  

Each likelihood factory has a corresponding **NumCosmo** pipeline configuration file:  

- `numcosmo/factory.yaml`: Standard DES Y1 3x2pt analysis.  
- `numcosmo/factory_PT.yaml`: Analysis with perturbation theory.  
- `numcosmo/pure_ccl.yaml`: Analysis in **pure CCL** mode.  
- `numcosmo/mu_sigma.yaml`: Analysis in **mu_sigma_isitgr** mode.  

### Example Command  

Run the NumCosmo pipeline from the `numcosmo` directory with:  

```bash
numcosmo run test factory.yaml
```  

This runs the pipeline using `factory.yaml` and computes a single-point likelihood
estimate.  

To compute the best-fit point, use:  

```bash
numcosmo run fit pure_ccl.yaml -p
```  

This runs the pipeline with `pure_ccl.yaml`, computing the best-fit point of the
likelihood. Results are saved in `pure_ccl.product.yaml`.

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

There is already a `sacc_data.fits` file in this directory that contains the DES Y1
3x2pt data products. If you want to regenerate the `firecrown` inputs, you can run
`generate_des_data.py`. To generate the `firecrown` inputs, first download the DES Y1
3x2pt data products
[here](http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits).

Then run the script

```bash
$ python generate_des_data.py
```
