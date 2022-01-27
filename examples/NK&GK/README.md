# NK and GK constraints

This is part of 5x2pt analysis. `NK.yaml` constrains cosmology based on galaxy number count x CMB lensing correlation (data stored in `NK.sacc`). `GK.yanml` constrains cosmology based on galaxy weak lensing x CMB lensing correlation (data stored in `GK.sacc`). 

Both constraints use `emcee` as the sampler for MCMC chains. 

To run this example, type

```bash
$ firecrown run-cosmosis NK.yaml
```

or for GK, type

```bash
$ firecrown run-cosmosis GK.yaml
```
