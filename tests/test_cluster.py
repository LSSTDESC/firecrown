from firecrown.models.cluster_abundance import ClusterAbundance, ClusterAbundanceInfo
from firecrown.models.cluster_abundance_binned import ClusterAbundanceBinned
from firecrown.models.cluster_redshift import ClusterRedshift
from firecrown.models.cluster_mass import ClusterMass
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_mean_mass_bin import ClusterMeanMass
import pyccl as ccl

ccl_cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
)
hmd_200 = ccl.halos.MassDef200c()
hmf_args = [False, True]
hmf_200 = ccl.halos.MassFuncBocquet16
proxy_params = [3.091,0.8,0,0.3,0.8,0]
pivot_mass = 14.0
pivot_redshift = 0.6
sky_area = 489

def initialize_objects():
    cluster_mass = ClusterMass(hmd_200, hmf_200, hmf_args)
    cluster_z = ClusterRedshift()
    cluster_mass_proxy = ClusterMassRich(hmd_200, hmf_200, proxy_params, pivot_mass, pivot_redshift, hmf_args )
    cluster_abundance = ClusterAbundance(cluster_mass, cluster_z, sky_area)
    cluster_abundance_Mproxy = ClusterAbundance(cluster_mass_proxy, cluster_z, sky_area)
    cluster_abundance_binned = ClusterAbundanceBinned(cluster_mass, cluster_z, sky_area)
    cluster_abundance_binned_Mproxy = ClusterAbundanceBinned(cluster_mass_proxy, cluster_z, sky_area)
    return [cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy]


def test_initialize_objects():

    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy,cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()

    assert(type(cluster_mass).__name__ == 'ClusterMass')
    assert(type(cluster_z).__name__ == 'ClusterRedshift')
    assert(type(cluster_mass_proxy).__name__ == 'ClusterMassRich')
    assert(type(cluster_abundance).__name__ == 'ClusterAbundance')
    assert(type(cluster_abundance_Mproxy).__name__ == 'ClusterAbundance')

def test_cluster_mass_functions():
    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()

    logM = 13.0
    z = 1.0
    logM_obs = 2.0
    logM_obs_lower = 1.0
    logM_obs_upper = 2.5
    a = 1.0 / (1.0 + z)
    hmf_test = ccl.halos.MassFuncBocquet16(ccl_cosmo, hmd_200, hmf_args)

    assert(cluster_mass.compute_mass_function(ccl_cosmo, logM, z) == hmf_test.get_mass_function(ccl_cosmo, 10**logM, a))
    cluster_mass.set_logM_limits(12, 15)
    assert(cluster_mass.logMl == 12)
    assert(cluster_mass.logMu == 15)
    test_mass_p = cluster_mass_proxy.cluster_logM_p(logM, z, logM_obs)
    assert(test_mass_p != 0.0)
    test_mass_intp = cluster_mass_proxy.cluster_logM_intp(logM, z)
    assert(test_mass_intp != 0.0)
    test_mass_intp_bin = cluster_mass_proxy.cluster_logM_intp_bin(logM, z,logM_obs_lower, logM_obs_upper)
    assert(test_mass_intp != 0.0)

def test_cluster_redshift_functions():
    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()
    z = 1.0
    a = 1.0 / (1.0 + z)  # pylint: disable=invalid-name
    # pylint: disable-next=invalid-name
    da = ccl.background.angular_diameter_distance(ccl_cosmo, a)
    E = ccl.background.h_over_h0(ccl_cosmo, a)  # pylint: disable=invalid-name
    dV = (  # pylint: disable=invalid-name
        ((1.0 + z) ** 2)
        * (da**2)
        * ccl.physical_constants.CLIGHT_HMPC
        / ccl_cosmo["h"]
        / E
        )
    assert(cluster_z.compute_differential_comoving_volume(ccl_cosmo, z) == dV)
    cluster_z.set_redshift_limits(0.0, 10.0)


    assert(cluster_z.zl == 0.0)
    assert(cluster_z.zu == 10.0)


def test_cluster_abundance_functions():
    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()
    logM = 13.0
    z = 1.0
    logM_obs = 2.0
    logM_obs_lower = 1.0
    logM_obs_upper = 2.5
    z_obs_lower = 0.2
    z_obs_upper = 1.0
    #Counts unbinned for true mass and richness proxy
    N_true = cluster_abundance.compute_N(ccl_cosmo)
    N_logM_p = cluster_abundance_Mproxy.compute_N(ccl_cosmo)
    assert(type(N_true) == float)
    assert(type(N_logM_p) == float)
    assert(N_true != N_logM_p)
    #d2n unbinned for true mass and richness proxy
    d2n_true = cluster_abundance.compute_intp_d2n(ccl_cosmo, logM, z)
    d2n_logM_p = cluster_abundance_Mproxy.compute_intp_d2n(ccl_cosmo, logM, z)
    logM_p = cluster_mass_proxy.cluster_logM_p(logM, z, logM_obs)
    cluster_abundance_Mproxy.info = ClusterAbundanceInfo(ccl_cosmo, z=z, logM_obs=logM_obs)
    d2n_logM_p_int = cluster_abundance_Mproxy._cluster_abundance_logM_p_d2n_integrand(logM)
    assert(d2n_true != d2n_logM_p)
    assert((d2n_logM_p_int - d2n_true * logM_p) / d2n_logM_p_int < 0.05  )



def test_cluster_abundance_binned_functions():
    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()
    logM = 13.0
    z = 0.2
    logM_obs = 2.0
    logM_lower, logM_upper = 13, 15
    logM_obs_lower = 13
    logM_obs_upper = 15
    z_obs_lower = 0.0
    z_obs_upper = 1.2

    N_bin = cluster_abundance_binned.compute_bin_N(ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper)
    assert(N_bin != 0.0)

    d2n_logM_p_bin = cluster_abundance_binned_Mproxy.compute_intp_bin_d2n(ccl_cosmo, logM, z, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper)
    N_logM_p_bin = cluster_abundance_binned_Mproxy.compute_bin_N(ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper)
    assert(d2n_logM_p_bin != 0.0)
    assert(N_logM_p_bin != 0.0)


def test_cluster_mean_mass():
    cluster_mass, cluster_z, cluster_mass_proxy, cluster_abundance, cluster_abundance_Mproxy, cluster_abundance_binned, cluster_abundance_binned_Mproxy = initialize_objects()
    cluster_mean_mass = ClusterMeanMass(cluster_mass, cluster_z, sky_area)


