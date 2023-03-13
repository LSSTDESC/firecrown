import cluster_mass
import cluster_redshift
import cluster_abundance
import pyccl as ccl
import cluster_mass_rich_proxy

cl_z = cluster_redshift.ClusterRedshift()
hmd_200 = ccl.halos.MassDef200c()
hmf_200 = ccl.halos.MassFuncBocquet16
args = [False, True]
cl_m = cluster_mass.ClusterMass(hmd_200, hmf_200, args)
ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
proxy_params = [3.091,0.8,0,0.3,0.8,0]
cl_m_r = cluster_mass_rich_proxy.ClusterMassRich(hmd_200, hmf_200, proxy_params, 14, 0.6, args )
cl_abundance = cluster_abundance.ClusterAbundance(cl_m_r, cl_z, 489)

print(cl_abundance.compute_N(ccl_cosmo))




