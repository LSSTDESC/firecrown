# Sampling and Analysis Examples

## Sampling with `emcee` via `cosmosis`

You first need to install the test loglike via

```bash
pip install -e .
```

Then see the file `emcee.yaml` and the comments for instructions.

To run the example type

```bash
firecrown run-cosmosis emcee.yaml
```

For MPI, you will need to use `mpirun`, `srun` or the equivalent on your system like this

```bash
mpirun -n 10 firecrown run-cosmosis emcee.yaml
```
