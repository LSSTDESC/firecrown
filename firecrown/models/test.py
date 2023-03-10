import cluster_mass
import cluster_redshift
import cluster_abundance
import pyccl as ccl


cl_z = cluster_redshift.ClusterRedshift()
hmd_200 = ccl.halos.MassDef200c()
hmf_200 = ccl.halos.MassFuncBocquet16
args = [False, True]
cl_m = cluster_mass.ClusterMass(hmd_200, hmf_200, args)
ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
print(cl_m.compute_mass_function(ccl_cosmo, 13, 0.2))
print(cl_z.compute_differential_comoving_volume(ccl_cosmo, 0.5))

cl_abundance = cluster_abundance.ClusterAbundance(cl_m, cl_z)





