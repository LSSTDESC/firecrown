#!/usr/bin/env python

import pyccl as ccl
from firecrown.models.cluster_abundance import ClusterAbundance, ClusterAbundanceInfo
from firecrown.models.cluster_abundance_binned import ClusterAbundanceBinned
from firecrown.models.cluster_redshift import ClusterRedshift
from firecrown.models.cluster_mass import ClusterMass
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_mean_mass_bin import ClusterMeanMass
import numpy as np

ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
hmd_200 = ccl.halos.MassDef200c()
# hmf_args = [True, True]
hmf_args = None
hmf_200 = ccl.halos.MassFuncTinker08
# proxy_params = [3.091,0.8685889638,0.0,0.33,-0.03474355855,0.0]
proxy_params = [
    3.19,
    0.8685889638,
    -0.30400613733,
    0.33,
    -0.03474355855,
    0.0,
]
pivot_mass = 14.625862906
pivot_redshift = 0.6
sky_area = 439

z_bins = [0.2000146, 0.31251036, 0.42500611, 0.53750187, 0.64999763]
# proxy_bins = [0.1 ,0.45805137, 0.81610273, 1.1741541 , 1.53220547 ,1.89025684]
proxy_bins = [0.45805137, 0.81610273, 1.1741541, 1.53220547, 1.89025684]
cluster_mass_r = ClusterMassRich(
    hmd_200,
    hmf_200,
    proxy_params,
    pivot_mass=pivot_mass,
    pivot_redshift=pivot_redshift,
    hmf_args=hmf_args,
)
cluster_mass = ClusterMass(hmd_200, hmf_200, hmf_args=hmf_args)
cluster_z = ClusterRedshift()
cluster_abundance_bin = ClusterAbundanceBinned(cluster_mass_r, cluster_z, sky_area)
cluster_abundance_bin_p = ClusterAbundanceBinned(cluster_mass, cluster_z, sky_area)
cluster_mean_bin = ClusterMeanMass(cluster_mass_r, cluster_z, sky_area)
test = []
for i in range(len(z_bins) - 1):
    for j in range(len(proxy_bins) - 1):
        result = cluster_abundance_bin.compute_bin_N(
            ccl_cosmo, proxy_bins[j], proxy_bins[j + 1], z_bins[i], z_bins[i + 1]
        )
        test.append(result)
print(test)
mass_bins = [13, 14, 15]
test = []
for i in range(len(z_bins) - 1):
    for j in range(len(mass_bins) - 1):
        result = cluster_abundance_bin_p.compute_bin_N(
            ccl_cosmo, mass_bins[j], mass_bins[j + 1], z_bins[i], z_bins[i + 1]
        )
        test.append(result)
print(test)


test = []
for i in range(len(z_bins) - 1):
    for j in range(len(proxy_bins) - 1):
        result = cluster_mean_bin.compute_bin_logM(
            ccl_cosmo, proxy_bins[j], proxy_bins[j + 1], z_bins[i], z_bins[i + 1]
        )
        test.append(result)
print(test)
