# Sampling and Analysis Examples

## Sampling with `emcee`

See the file `emcee.yaml` and the comments for instructions. To run the example
type

```bash
firecrown run-emcee emcee.yaml
```

For the mpi backend, you will probably need to use `mpirun`, `srun` or the
equivalent on your system like this

```bash
mpirun -n 10 firecrown run-emcee emcee.yaml
```

Remember to set the proper backend in your configuration file as well.
