#!/usr/bin/env python

import pyccl as ccl

from number_counts_rich import build_likelihood


"""
In here we define the cosmology to be used to compute the statistics.
"""
cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)

"""
Initiate the likelihood object, which will read the sacc file and then
compute the log(Likelihood). To change the data file and
the mass function type, check number_counts.py
"""
lk = build_likelihood(None)
log = lk.compute_loglike(cosmo)
print(f"The log(L) is: {log}")
