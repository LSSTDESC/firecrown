"""Synthetic and published data sets for the galaxy cluster examples.

Simulates cluster catalogs with NumCosmo, bins them in redshift and richness,
optionally computes stacked weak lensing profiles with CLMM, and writes the
result as SACC data. Also provides the SDSS redMaPPer counts and mean masses
of Costanzi et al. (2019), arXiv:1810.09456.
"""

import itertools
import math
import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyccl
import sacc
from numcosmo_py import Nc, Ncm
from scipy import stats

from ..analysis import CCL_COSMOLOGY_MINIMAL_SET, COSMO_DESC, CCLCosmologySpec

# Fiducial cosmology used to simulate the cluster catalogs.
FIDUCIAL_H0 = 71.0
FIDUCIAL_OMEGA_B = 0.0448
FIDUCIAL_OMEGA_C = 0.22
FIDUCIAL_N_S = 0.963
FIDUCIAL_SIGMA8 = 0.8

# Survey and mass-richness relation used to simulate the cluster catalogs.
SIM_AREA = 439.78986
SIM_LN_RICHNESS_RANGE = (0.0, 5.0)
SIM_Z_RANGE = (0.2, 0.65)
SIM_PIVOT_MASS = 3.0e14 / 0.71
SIM_PIVOT_REDSHIFT = 0.6
SIM_MIN_LN_RICHNESS = 2.0
SIM_N_Z_BINS = 4
SIM_N_RICHNESS_BINS = 5

# Lensing profile model and radial binning used for the shear data.
PROFILE_CONCENTRATION = 4.0
PROFILE_RADIUS_RANGE = (0.3, 6.0)
PROFILE_N_RADIUS_BINS = 6
PROFILE_Z_INF = 10.0
PROFILE_SOURCE_ZMAX = 5.0

SDSS_SURVEY_NAME = "SDSSCluster_redshift_richness"


def fiducial_cosmology_spec() -> CCLCosmologySpec:
    """Return the cosmology specification matching the simulated catalogs.

    Omega_c and sigma8 are free; all other parameters are fixed at the
    values used to simulate the data.
    """
    fixed = {
        "Omega_b": FIDUCIAL_OMEGA_B,
        "h": FIDUCIAL_H0 / 100.0,
        "n_s": FIDUCIAL_N_S,
    }
    free = {
        "Omega_c": (0.1552, 0.3552, FIDUCIAL_OMEGA_C),
        "sigma8": (0.7, 0.9, FIDUCIAL_SIGMA8),
    }
    parameters = []
    for name in CCL_COSMOLOGY_MINIMAL_SET + ["sigma8"]:
        param = COSMO_DESC[name]
        if name in fixed:
            param = param.model_copy(
                update={"default_value": fixed[name], "free": False}
            )
        elif name in free:
            lower, upper, default = free[name]
            param = param.model_copy(
                update={
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "default_value": default,
                    "free": True,
                }
            )
        parameters.append(param)
    return CCLCosmologySpec(parameters=parameters)


@dataclass(frozen=True)
class ClusterCatalog:
    """Observed redshift, log10 richness and true log10 mass of each cluster."""

    z: npt.NDArray[np.float64]
    log10_richness: npt.NDArray[np.float64]
    log10_mass: npt.NDArray[np.float64]


@dataclass(frozen=True)
class BinnedClusters:
    """Cluster counts and mean log10 mass binned in redshift and richness."""

    z_edges: npt.NDArray[np.float64]
    richness_edges: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    mean_log10_mass: npt.NDArray[np.float64]
    var_mean_log10_mass: npt.NDArray[np.float64]


@dataclass(frozen=True)
class StackedProfiles:
    """Stacked DeltaSigma and reduced shear profiles per redshift-richness bin.

    Arrays have shape (n_z, n_richness, n_radius).
    """

    radius_edges: npt.NDArray[np.float64]
    mean_delta_sigma: npt.NDArray[np.float64]
    var_delta_sigma: npt.NDArray[np.float64]
    mean_reduced_shear: npt.NDArray[np.float64]
    var_reduced_shear: npt.NDArray[np.float64]


def _numcosmo_cosmology() -> Nc.HICosmoDEXcdm:
    """Create the fiducial NumCosmo cosmology normalized to sigma8."""
    cosmo = Nc.HICosmoDEXcdm()
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(reion)
    cosmo.add_submodel(prim)

    psml = Nc.PowspecMLTransfer.new(
        Nc.TransferFuncEH.new()  # pylint: disable=no-value-for-parameter
    )
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    psf = Ncm.PowspecFilter.new(psml, Ncm.PowspecFilterType.TOPHAT)
    psf.set_best_lnr0()

    cosmo.props.H0 = FIDUCIAL_H0
    cosmo.props.Omegab = FIDUCIAL_OMEGA_B
    cosmo.props.Omegac = FIDUCIAL_OMEGA_C
    cosmo.omega_x2omega_k()
    cosmo.param_set_by_name("Omegak", 0.0)

    prim.props.n_SA = FIDUCIAL_N_S
    old_amplitude = math.exp(prim.props.ln10e10ASA)
    prim.props.ln10e10ASA = math.log(
        (FIDUCIAL_SIGMA8 / cosmo.sigma8(psf)) ** 2 * old_amplitude
    )
    return cosmo


def _ascaso_mass_relation() -> Nc.ClusterMassAscaso:
    """Create the Ascaso mass-richness relation with arXiv:1904.07524 values."""
    cluster_m = Nc.ClusterMassAscaso(
        M0=SIM_PIVOT_MASS,
        z0=SIM_PIVOT_REDSHIFT,
        lnRichness_min=SIM_LN_RICHNESS_RANGE[0],
        lnRichness_max=SIM_LN_RICHNESS_RANGE[1],
    )
    for name, value in [
        ("mup0", 3.19),
        ("mup1", 2.0 / np.log(10)),
        ("mup2", -0.7 / np.log(10)),
        ("sigmap0", 0.33),
        ("sigmap1", -0.08 / np.log(10)),
        ("sigmap2", 0.0),
    ]:
        cluster_m.param_set_by_name(name, value)
    return cluster_m


def _halo_mass_function(
    mass_def: Nc.MultiplicityFuncMassDef,
) -> Nc.HaloMassFunction:
    """Create the Tinker halo mass function for overdensity 200 over the survey."""
    mulf = Nc.MultiplicityFuncTinker.new()  # pylint: disable=no-value-for-parameter
    mulf.set_linear_interp(True)
    mulf.set_mdef(mass_def)
    mulf.set_Delta(200)
    hmf = Nc.HaloMassFunction.new(
        Nc.Distance.new(2.0),
        Ncm.PowspecFilter.new(
            Nc.PowspecMLTransfer.new(
                Nc.TransferFuncEH.new()  # pylint: disable=no-value-for-parameter
            ),
            Ncm.PowspecFilterType.TOPHAT,
        ),
        mulf,
    )
    hmf.set_area_sd(SIM_AREA)
    return hmf


def simulate_catalog(seed: int, mass_def: Nc.MultiplicityFuncMassDef) -> ClusterCatalog:
    """Simulate a richness-selected cluster catalog with NumCosmo.

    Uses a Tinker halo mass function with overdensity 200 with respect to
    ``mass_def`` and the Ascaso mass-richness relation with parameters from
    arXiv:1904.07524. Clusters with ln(richness) below
    :data:`SIM_MIN_LN_RICHNESS` are discarded.

    :param seed: Seed for the NumCosmo random number generator
    :param mass_def: Background density used to define halo masses
    :returns: The simulated catalog
    """
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter
    cosmo = _numcosmo_cosmology()

    cluster_z = Nc.ClusterRedshiftNodist(z_min=SIM_Z_RANGE[0], z_max=SIM_Z_RANGE[1])
    cluster_m = _ascaso_mass_relation()

    abundance = Nc.ClusterAbundance.new(_halo_mass_function(mass_def), None)
    ncount = Nc.DataClusterNCount.new(
        abundance, "NcClusterRedshiftNodist", "NcClusterMassAscaso"
    )
    abundance.prepare(cosmo, cluster_z, cluster_m)
    mset = Ncm.MSet.new_array([cosmo, cluster_z, cluster_m])
    ncount.init_from_sampling(
        mset, SIM_AREA * (np.pi / 180.0) ** 2, Ncm.RNG.seeded_new(None, seed)
    )

    z_obs = np.array(ncount.get_z_obs().dup_array())
    ln_richness = np.array(ncount.get_lnM_obs().dup_array())
    ln_mass = np.array(ncount.get_lnM_true().dup_array())
    selected = ln_richness > SIM_MIN_LN_RICHNESS

    return ClusterCatalog(
        z=z_obs[selected],
        log10_richness=ln_richness[selected] / np.log(10.0),
        log10_mass=ln_mass[selected] / np.log(10.0),
    )


def bin_catalog(catalog: ClusterCatalog) -> BinnedClusters:
    """Bin the catalog in redshift and richness.

    :param catalog: Simulated cluster catalog
    :returns: Counts, mean log10 mass and its variance in each bin
    """
    counts, z_edges, richness_edges, _ = stats.binned_statistic_2d(
        catalog.z,
        catalog.log10_richness,
        catalog.log10_mass,
        "count",
        bins=[SIM_N_Z_BINS, SIM_N_RICHNESS_BINS],
    )
    bins = [z_edges, richness_edges]
    mean = stats.binned_statistic_2d(
        catalog.z, catalog.log10_richness, catalog.log10_mass, "mean", bins=bins
    ).statistic
    std = stats.binned_statistic_2d(
        catalog.z, catalog.log10_richness, catalog.log10_mass, "std", bins=bins
    ).statistic
    n_clusters = np.asarray(counts, dtype=np.float64)
    return BinnedClusters(
        z_edges=z_edges,
        richness_edges=richness_edges,
        counts=n_clusters,
        mean_log10_mass=np.asarray(mean, dtype=np.float64),
        var_mean_log10_mass=np.asarray(std, dtype=np.float64) ** 2 / n_clusters,
    )


def _cluster_profiles(
    catalog: ClusterCatalog, cosmo: pyccl.Cosmology
) -> tuple[npt.NDArray[np.float64], np.ndarray, np.ndarray]:
    """Evaluate the NFW DeltaSigma and reduced shear profiles of every cluster.

    :returns: Radial bin edges, and DeltaSigma and reduced shear at the bin
        centers with shape (n_clusters, n_radius)
    """
    os.environ.setdefault("CLMM_MODELING_BACKEND", "ccl")
    # pylint: disable-next=import-outside-toplevel
    import clmm
    from clmm.utils.beta_lens import (  # pylint: disable=import-outside-toplevel
        compute_beta_s_mean_from_distribution,
        compute_beta_s_square_mean_from_distribution,
    )

    modeling = clmm.Modeling(
        massdef="critical", delta_mdef=200, halo_profile_model="nfw"
    )
    modeling.set_cosmo(clmm.Cosmology(be_cosmo=cosmo))
    modeling.set_concentration(PROFILE_CONCENTRATION)

    radius_edges = np.asarray(
        clmm.make_bins(
            *PROFILE_RADIUS_RANGE, nbins=PROFILE_N_RADIUS_BINS, method="evenlog10width"
        )
    )
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    delta_sigma = []
    reduced_shear = []
    for z_cl, log10_mass in zip(catalog.z, catalog.log10_mass):
        modeling.set_mass(10.0**log10_mass)
        delta_sigma.append(modeling.eval_excess_surface_density(radius_centers, z_cl))
        beta_kwargs = {
            "z_cl": z_cl,
            "z_inf": PROFILE_Z_INF,
            "cosmo": modeling.cosmo,
            "zmax": PROFILE_SOURCE_ZMAX,
        }
        beta = compute_beta_s_mean_from_distribution(**beta_kwargs)
        beta2 = compute_beta_s_square_mean_from_distribution(**beta_kwargs)
        reduced_shear.append(
            modeling.eval_reduced_tangential_shear(
                radius_centers,
                z_cl,
                (beta, beta2),
                z_src_info="beta",
                approx="order1",
            )
        )
    return radius_edges, np.array(delta_sigma), np.array(reduced_shear)


def _stack(
    catalog: ClusterCatalog, binned: BinnedClusters, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-cluster profiles in each redshift-richness bin.

    :param values: Profiles with shape (n_clusters, n_radius)
    :returns: Mean and variance of the mean, with shape (n_z, n_richness, n_radius)
    """
    bins = [binned.z_edges, binned.richness_edges]
    n_radius = values.shape[1]
    mean = np.empty(binned.counts.shape + (n_radius,))
    var = np.empty_like(mean)
    for i in range(n_radius):
        mean[:, :, i] = stats.binned_statistic_2d(
            catalog.z, catalog.log10_richness, values[:, i], "mean", bins=bins
        ).statistic
        std = stats.binned_statistic_2d(
            catalog.z, catalog.log10_richness, values[:, i], "std", bins=bins
        ).statistic
        var[:, :, i] = std**2 / binned.counts
    return mean, var


def stack_profiles(
    catalog: ClusterCatalog, binned: BinnedClusters, cosmo: pyccl.Cosmology
) -> StackedProfiles:
    """Compute NFW DeltaSigma and reduced shear profiles stacked in each bin.

    Every cluster has concentration :data:`PROFILE_CONCENTRATION` and sources
    following the CLMM default redshift distribution.

    :param catalog: Simulated cluster catalog
    :param binned: Redshift and richness binning of the catalog
    :param cosmo: Cosmology used to evaluate the profiles
    :returns: Mean profiles and the variance of the mean in each bin
    """
    radius_edges, delta_sigma, reduced_shear = _cluster_profiles(catalog, cosmo)
    mean_ds, var_ds = _stack(catalog, binned, delta_sigma)
    mean_gt, var_gt = _stack(catalog, binned, reduced_shear)
    return StackedProfiles(
        radius_edges=radius_edges,
        mean_delta_sigma=mean_ds,
        var_delta_sigma=var_ds,
        mean_reduced_shear=mean_gt,
        var_reduced_shear=var_gt,
    )


def _add_counts_and_mass(
    sacc_data: sacc.Sacc,
    survey_name: str,
    z_edges: npt.NDArray[np.float64],
    richness_edges: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    mean_log10_mass: npt.NDArray[np.float64],
) -> tuple[list[str], list[str]]:
    """Add the binning tracers, counts and mean log10 mass to a SACC object.

    :returns: Labels of the redshift and richness bin tracers
    """
    z_labels = [f"bin_z_{i}" for i in range(len(z_edges) - 1)]
    richness_labels = [f"bin_rich_{i}" for i in range(len(richness_edges) - 1)]
    for label, lower, upper in zip(z_labels, z_edges[:-1], z_edges[1:]):
        sacc_data.add_tracer("bin_z", label, lower, upper)
    for label, lower, upper in zip(
        richness_labels, richness_edges[:-1], richness_edges[1:]
    ):
        sacc_data.add_tracer("bin_richness", label, lower, upper)

    bin_pairs = list(itertools.product(z_labels, richness_labels))
    # pylint: disable=no-member
    for value, (z_label, richness_label) in zip(counts.flatten(), bin_pairs):
        sacc_data.add_data_point(
            sacc.standard_types.cluster_counts,
            (survey_name, z_label, richness_label),
            int(value),
        )
    for value, (z_label, richness_label) in zip(mean_log10_mass.flatten(), bin_pairs):
        sacc_data.add_data_point(
            sacc.standard_types.cluster_mean_log_mass,
            (survey_name, z_label, richness_label),
            float(value),
        )
    # pylint: enable=no-member
    return z_labels, richness_labels


def counts_sacc(survey_name: str, binned: BinnedClusters) -> sacc.Sacc:
    """Build a SACC object with cluster counts and mean log10 mass.

    The covariance is diagonal: Poisson variance for the counts and the
    variance of the mean for the masses.

    :param survey_name: Name of the survey tracer
    :param binned: Binned cluster data
    :returns: The SACC object
    """
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("survey", survey_name, SIM_AREA)
    _add_counts_and_mass(
        sacc_data,
        survey_name,
        binned.z_edges,
        binned.richness_edges,
        binned.counts,
        binned.mean_log10_mass,
    )
    sacc_data.add_covariance(
        np.diag(
            np.concatenate(
                [binned.counts.flatten(), binned.var_mean_log10_mass.flatten()]
            )
        )
    )
    sacc_data.to_canonical_order()
    return sacc_data


def _add_radius_bins(
    sacc_data: sacc.Sacc, radius_edges: npt.NDArray[np.float64]
) -> list[str]:
    """Add the radial bin tracers to a SACC object.

    :returns: Labels of the radial bin tracers
    """
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    radius_labels = [f"bin_radius_{i}" for i in range(len(radius_centers))]
    for label, lower, upper, center in zip(
        radius_labels, radius_edges[:-1], radius_edges[1:], radius_centers
    ):
        sacc_data.add_tracer(
            "bin_radius", label, float(lower), float(upper), float(center)
        )
    return radius_labels


def _add_profiles(
    sacc_data: sacc.Sacc,
    survey_name: str,
    z_labels: list[str],
    richness_labels: list[str],
    profiles: StackedProfiles,
) -> None:
    """Add the radial bin tracers and the stacked profiles to a SACC object."""
    radius_labels = _add_radius_bins(sacc_data, profiles.radius_edges)

    # pylint: disable=no-member
    for data_type, values in [
        (sacc.standard_types.cluster_delta_sigma, profiles.mean_delta_sigma),
        (sacc.standard_types.cluster_shear, profiles.mean_reduced_shear),
    ]:
        # pylint: enable=no-member
        for (i, z_label), (j, richness_label), (k, radius_label) in itertools.product(
            enumerate(z_labels), enumerate(richness_labels), enumerate(radius_labels)
        ):
            sacc_data.add_data_point(
                data_type,
                (survey_name, z_label, richness_label, radius_label),
                float(values[i, j, k]),
            )


def shear_sacc(
    survey_name: str, binned: BinnedClusters, profiles: StackedProfiles
) -> sacc.Sacc:
    """Build a SACC object with counts, mean mass, DeltaSigma and reduced shear.

    :param survey_name: Name of the survey tracer
    :param binned: Binned cluster data
    :param profiles: Stacked lensing profiles
    :returns: The SACC object
    """
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("survey", survey_name, SIM_AREA)
    z_labels, richness_labels = _add_counts_and_mass(
        sacc_data,
        survey_name,
        binned.z_edges,
        binned.richness_edges,
        binned.counts,
        binned.mean_log10_mass,
    )

    _add_profiles(sacc_data, survey_name, z_labels, richness_labels, profiles)

    sacc_data.add_covariance(
        np.diag(
            np.concatenate(
                [
                    binned.counts.flatten(),
                    binned.var_mean_log10_mass.flatten(),
                    profiles.var_delta_sigma.flatten(),
                    profiles.var_reduced_shear.flatten(),
                ]
            )
        )
    )
    sacc_data.to_canonical_order()
    return sacc_data


def sdss_sacc() -> sacc.Sacc:
    """Build a SACC object with the SDSS redMaPPer counts and mean log10 mass.

    Values are from Costanzi et al. (2019), arXiv:1810.09456: a single
    redshift bin 0.1 < z < 0.3 and five richness bins. The covariance is
    diagonal, combining the two quoted mass uncertainties in quadrature.

    :returns: The SACC object
    """
    area = 10263.037032448827
    z_edges = np.array([0.1, 0.3])
    richness_edges = np.log10(np.array([20.0, 27.9, 37.6, 50.3, 69.3, 140.0]))
    counts = np.array([[3711, 1788, 978, 476, 223]], dtype=np.float64)
    mean_log10_mass = np.array([[14.111, 14.263, 14.380, 14.609, 14.928]])
    std_log10_mass_1 = np.array([0.024, 0.030, 0.033, 0.036, 0.029])
    std_log10_mass_2 = np.array([0.026, 0.024, 0.026, 0.028, 0.036])
    std_counts = np.array([100.0, 61.0, 41.0, 27.0, 18.0])

    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("survey", SDSS_SURVEY_NAME, area)
    _add_counts_and_mass(
        sacc_data, SDSS_SURVEY_NAME, z_edges, richness_edges, counts, mean_log10_mass
    )
    sacc_data.add_covariance(
        np.diag(
            np.concatenate([std_counts**2, std_log10_mass_1**2 + std_log10_mass_2**2])
        )
    )
    sacc_data.to_canonical_order()
    return sacc_data
