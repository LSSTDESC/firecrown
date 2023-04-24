"""Tests for the cluster module."""

from typing import Any, Dict, cast

import pytest
import pyccl as ccl
import numpy as np

from firecrown.models.cluster_mass import ClusterMass
from firecrown.models.cluster_redshift import ClusterRedshift
from firecrown.models.cluster_abundance import ClusterAbundance
from firecrown.models.cluster_mass_rich_proxy import ClusterMassRich
from firecrown.models.cluster_redshift_spec import ClusterRedshiftSpec
from firecrown.parameters import ParamsMap


@pytest.fixture(name="ccl_cosmo")
def fixture_ccl_cosmo():
    return ccl.Cosmology(
        Omega_c=0.22, Omega_b=0.0448, h=0.71, sigma8=0.8, n_s=0.963, Neff=3.04
    )


@pytest.fixture(name="cluster_abundance")
def fixed_cluster_redshift():
    hmd_200 = ccl.halos.MassDef200c()
    hmf_args: Dict[str, Any] = {}
    hmf_name = "Bocquet16"
    pivot_mass = 14.0
    pivot_redshift = 0.6
    sky_area = 489

    parameters = ParamsMap(
        {
            "mu_p0": 3.19,
            "mu_p1": 0.8,
            "mu_p2": 0.0,
            "sigma_p0": 0.3,
            "sigma_p1": 0.8,
            "sigma_p2": 0.0,
        }
    )

    z_bins = np.array([0.2000146, 0.31251036, 0.42500611, 0.53750187, 0.64999763])
    proxy_bins = np.array([0.45805137, 0.81610273, 1.1741541, 1.53220547, 1.89025684])

    cluster_z = ClusterRedshiftSpec()
    cluster_z.set_bins_by_array(z_bins)

    cluster_mass_r = ClusterMassRich(pivot_mass, pivot_redshift)
    cluster_mass_r.set_bins_by_array(proxy_bins)

    cluster_abundance = ClusterAbundance(
        hmd_200, hmf_name, hmf_args, cluster_mass_r, cluster_z, sky_area
    )
    cluster_abundance.update(parameters)

    return cluster_abundance


def test_initialize_objects(ccl_cosmo, cluster_abundance: ClusterAbundance):
    cluster_m = cluster_abundance.cluster_m
    cluster_z = cluster_abundance.cluster_z

    assert isinstance(ccl_cosmo, ccl.Cosmology)
    assert isinstance(cluster_m, ClusterMass)
    assert isinstance(cluster_z, ClusterRedshift)
    assert isinstance(cluster_abundance, ClusterAbundance)


def test_cluster_mass_functions(
    ccl_cosmo: ccl.Cosmology, cluster_abundance: ClusterAbundance
):
    cluster_m = cast(ClusterMassRich, cluster_abundance.cluster_m)
    cluster_z = cast(ClusterRedshiftSpec, cluster_abundance.cluster_z)
    cluster_abundance.compute_from_args(
        ccl_cosmo, cluster_m.point_arg(0.2), cluster_z.point_arg(0.8)
    )


"""
def test_cluster_redshift_functions():
    (
        cluster_mass,
        cluster_z,
        cluster_mass_proxy,
        cluster_abundance,
        cluster_abundance_Mproxy,
        cluster_abundance_binned,
        cluster_abundance_binned_Mproxy,
    ) = initialize_objects()
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
    assert cluster_z.compute_differential_comoving_volume(ccl_cosmo, z) == dV
    cluster_z.set_redshift_limits(0.0, 10.0)

    assert cluster_z.zl == 0.0
    assert cluster_z.zu == 10.0


def test_cluster_abundance_functions():
    (
        cluster_mass,
        cluster_z,
        cluster_mass_proxy,
        cluster_abundance,
        cluster_abundance_Mproxy,
        cluster_abundance_binned,
        cluster_abundance_binned_Mproxy,
    ) = initialize_objects()
    logM = 13.0
    z = 1.0
    logM_obs = 2.0
    logM_obs_lower = 1.0
    logM_obs_upper = 2.5
    z_obs_lower = 0.2
    z_obs_upper = 1.0
    # Counts unbinned for true mass and richness proxy
    N_true = cluster_abundance.compute_N(ccl_cosmo)
    N_logM_p = cluster_abundance_Mproxy.compute_N(ccl_cosmo)
    assert type(N_true) == float
    assert type(N_logM_p) == float
    assert N_true != N_logM_p
    # d2n unbinned for true mass and richness proxy
    d2n_true = cluster_abundance.compute_intp_d2n(ccl_cosmo, logM, z)
    d2n_logM_p = cluster_abundance_Mproxy.compute_intp_d2n(ccl_cosmo, logM, z)
    logM_p = cluster_mass_proxy.cluster_logM_p(logM, z, logM_obs)
    cluster_abundance_Mproxy.info = ClusterAbundanceInfo(
        ccl_cosmo, z=z, logM_obs=logM_obs
    )
    d2n_logM_p_int = cluster_abundance_Mproxy._cluster_abundance_logM_p_d2n_integrand(
        logM
    )
    assert d2n_true != d2n_logM_p
    assert (d2n_logM_p_int - d2n_true * logM_p) / d2n_logM_p_int < 0.05


def test_cluster_abundance_binned_functions():
    (
        cluster_mass,
        cluster_z,
        cluster_mass_proxy,
        cluster_abundance,
        cluster_abundance_Mproxy,
        cluster_abundance_binned,
        cluster_abundance_binned_Mproxy,
    ) = initialize_objects()
    logM = 13.0
    z = 0.2
    logM_obs = 2.0
    logM_lower, logM_upper = 13, 15
    logM_obs_lower = 13
    logM_obs_upper = 15
    z_obs_lower = 0.0
    z_obs_upper = 1.2

    N_bin = cluster_abundance_binned.compute_bin_N(
        ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    )
    assert N_bin != 0.0

    d2n_logM_p_bin = cluster_abundance_binned_Mproxy.compute_intp_bin_d2n(
        ccl_cosmo, logM, z, logM_obs_lower=logM_obs_lower, logM_obs_upper=logM_obs_upper
    )
    N_logM_p_bin = cluster_abundance_binned_Mproxy.compute_bin_N(
        ccl_cosmo, logM_obs_lower, logM_obs_upper, z_obs_lower, z_obs_upper
    )
    assert d2n_logM_p_bin != 0.0
    assert N_logM_p_bin != 0.0


def test_cluster_mean_mass():
    (
        cluster_mass,
        cluster_z,
        cluster_mass_proxy,
        cluster_abundance,
        cluster_abundance_Mproxy,
        cluster_abundance_binned,
        cluster_abundance_binned_Mproxy,
    ) = initialize_objects()
    cluster_mean_mass = ClusterMeanMass(cluster_mass, cluster_z, sky_area) """
