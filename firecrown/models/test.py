import cluster_mass
import cluster_redshift
import cluster_abundance
import pyccl as ccl
import numpy as np
import cluster_mass_rich_proxy
import cluster_abundance_binned


cl_z = cluster_redshift.ClusterRedshift()
hmd_200 = ccl.halos.MassDef200c()
hmf_200 = ccl.halos.MassFuncBocquet16
args = [False, True]
cl_m = cluster_mass.ClusterMass(hmd_200, hmf_200, args)
ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
proxy_params = [3.091, 0.8, 0, 0.3, 0.8, 0]
cl_m_r = cluster_mass_rich_proxy.ClusterMassRich(
    hmd_200, hmf_200, proxy_params, np.log10(4.2253521e14), 0.6, args
)
cl_abundance = cluster_abundance.ClusterAbundance(cl_m_r, cl_z, 489)

cl_abundance_bin = cluster_abundance_binned.ClusterAbundanceBinned(cl_m, cl_z, 489)
cl_mean_mass_bin = _cluster_mean_mass_bin_logM.ClusterMeanMass(cl_m, cl_z, 489)
print(cl_abundance_bin.compute_N(13, 15, 0.2, 1.0))
