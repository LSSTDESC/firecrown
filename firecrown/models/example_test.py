#!/usr/bin/env python
"""Example test for cluster abundance model."""

from typing import Any, Dict
import itertools

import pyccl as ccl
import numpy as np

from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_redshift_spec import ClusterRedshiftSpec
from firecrown.parameters import ParamsMap

ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
hmd_200 = ccl.halos.MassDef200c()
hmf_args: Dict[str, Any] = {}
hmf_name = "Tinker08"
# proxy_params = [3.091,0.8685889638,0.0,0.33,-0.03474355855,0.0]

parameters = ParamsMap(
    {
        "mu_p0": 3.19,
        "mu_p1": 0.8685889638,
        "mu_p2": -0.30400613733,
        "sigma_p0": 0.33,
        "sigma_p1": -0.03474355855,
        "sigma_p2": 0.0,
    }
)

pivot_mass = 14.625862906
pivot_redshift = 0.6
sky_area = 439.0

proxy_bins = np.array([0.45805137, 0.81610273, 1.1741541, 1.53220547, 1.89025684])
cluster_mass_r = ClusterMassRich(pivot_mass, pivot_redshift)
mass_bin_args = cluster_mass_r.gen_bins_by_array(proxy_bins)

z_bins = np.array([0.2000146, 0.31251036, 0.42500611, 0.53750187, 0.64999763])
cluster_z = ClusterRedshiftSpec()
z_bin_args = cluster_z.gen_bins_by_array(z_bins)

cluster_abundance = ClusterAbundance(hmd_200, hmf_name, hmf_args, sky_area)

cluster_abundance.update(parameters)
cluster_mass_r.update(parameters)
cluster_z.update(parameters)

for mass_arg, z_arg in itertools.product(mass_bin_args, z_bin_args):
    print(mass_arg, z_arg)
    ca = cluster_abundance.compute(ccl_cosmo, mass_arg, z_arg)
    ca_logM = cluster_abundance.compute_unormalized_mean_logM(
        ccl_cosmo, mass_arg, z_arg
    )
    print(f"ca {ca} mean logM {ca_logM/ca}")
