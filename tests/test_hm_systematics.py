"""
Tests for the halo model theory systematics for
Weak Lensing.
"""

import os

import pytest

import numpy as np
import pyccl as ccl
import sacc

from firecrown.updatable import get_default_params_map
import firecrown.likelihood.weak_lensing as wl
import firecrown.metadata_types as mdt
from firecrown.likelihood.two_point import (
    TwoPoint,
    TracerNames,
)
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


@pytest.fixture(name="weak_lensing_source")
def fixture_weak_lensing_source() -> wl.WeakLensing:
    ia_systematic = wl.HMAlignmentSystematic()
    pzshift = wl.PhotoZShift(sacc_tracer="src0")
    return wl.WeakLensing(sacc_tracer="src0", systematics=[pzshift, ia_systematic])


@pytest.fixture(name="sacc_data")
def fixture_sacc_data() -> sacc.Sacc:
    # Load sacc file
    # This shouldn't be necessary, since we only use the n(z) from the sacc file
    saccfile = os.path.join(
        os.path.split(__file__)[0],
        "../examples/des_y1_3x2pt/sacc_data.fits",
    )
    return sacc.Sacc.load_fits(saccfile)


def test_hm_systematics(weak_lensing_source, sacc_data):
    # The following disabling of pylint warnings are TEMPORARY. Disabling warnings is
    # generally not a good practice. In this case, the warnings are indicating that this
    # test is too complicated.
    #
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-statements
    stats = [
        TwoPoint("galaxy_shear_xi_plus", weak_lensing_source, weak_lensing_source),
        TwoPoint("galaxy_shear_xi_minus", weak_lensing_source, weak_lensing_source),
    ]

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    src0_tracer = sacc_data.get_tracer("src0")
    z, nz = src0_tracer.z, src0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    mass_def = "200m"
    cM = "Duffy08"
    nM = "Tinker10"
    bM = "Tinker10"
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=mass_def)

    modeling_tools = ModelingTools(
        hm_calculator=hmc,
        cM_relation=cM,
        ccl_factory=CCLFactory(require_nonlinear_pk=True),
    )
    params = get_default_params_map(modeling_tools)
    modeling_tools.update(params)
    modeling_tools.prepare()

    # Bare CCL setup
    a_1h = 1.0e-3
    a_2h = 1.0

    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=mass_def)
    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(
        mass_def=mass_def, concentration=cM, a1h=a_1h, b=-2
    )
    NFW = ccl.halos.HaloProfileNFW(
        mass_def=mass_def, concentration=cM, truncated=True, fourier_analytic=True
    )
    a_arr = np.linspace(0.1, 1, 16)
    # Code that creates Pk2D objects:
    pk_GI_1h = ccl.halos.halomod_Pk2D(
        ccl_cosmo,
        hmc,
        NFW,
        prof2=sat_gamma_HOD,
        get_2h=False,
        a_arr=a_arr,
    )
    pk_II_1h = ccl.halos.halomod_Pk2D(
        ccl_cosmo,
        hmc,
        sat_gamma_HOD,
        get_2h=False,
        a_arr=a_arr,
    )
    # NLA
    C1rhocrit = 5e-14 * ccl.physical_constants.RHO_CRITICAL  # standard IA normalisation
    pk_GI_NLA = ccl.Pk2D.from_function(
        pkfunc=lambda k, a: -a_2h
        * C1rhocrit
        * ccl_cosmo["Omega_m"]
        / ccl_cosmo.growth_factor(a)
        * ccl_cosmo.nonlin_matter_power(k, a),
        is_logp=False,
    )
    pk_II_NLA = ccl.Pk2D.from_function(
        pkfunc=lambda k, a: (
            a_2h * C1rhocrit * ccl_cosmo["Omega_m"] / ccl_cosmo.growth_factor(a)
        )
        ** 2
        * ccl_cosmo.nonlin_matter_power(k, a),
        is_logp=False,
    )
    pk_GI = pk_GI_1h + pk_GI_NLA
    pk_II = pk_II_1h + pk_II_NLA

    # Set the parameters for our systematics
    params.update(
        {
            "ia_a_1h": a_1h,
            "ia_a_2h": a_2h,
            "src0_delta_z": 0.000,
        }
    )

    # Apply the systematics parameters
    likelihood.update(params)

    # Make things faster by only using a couple of ells
    for s in likelihood.statistics:
        s.ell_for_xi = {"minimum": 2, "midpoint": 5, "maximum": 60_000, "n_log": 10}

    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    _ = likelihood.compute_loglike(modeling_tools)

    # print(list(likelihood.statistics[0].cells.keys()))
    # pylint: disable=no-member

    s0 = likelihood.statistics[0].statistic
    assert isinstance(s0, TwoPoint)
    ells = s0.ells_for_xi
    cells_GG = s0.cells[TracerNames("shear", "shear")]
    cells_GI = s0.cells[TracerNames("shear", "intrinsic_hm")]
    cells_II = s0.cells[TracerNames("intrinsic_hm", "intrinsic_hm")]
    cells_cs_total = s0.cells[mdt.TRACER_NAMES_TOTAL]

    s1 = likelihood.statistics[1].statistic
    # del weak_lensing_source.cosmo_hash
    s1.compute_theory_vector(modeling_tools)
    assert isinstance(s1, TwoPoint)
    ells = s1.ells_for_xi
    cells_GG_m = s1.cells[TracerNames("shear", "shear")]
    cells_GI_m = s1.cells[TracerNames("shear", "intrinsic_hm")]
    cells_II_m = s1.cells[TracerNames("intrinsic_hm", "intrinsic_hm")]
    cells_cs_total_m = s1.cells[mdt.TRACER_NAMES_TOTAL]

    # pylint: enable=no-member
    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz), use_A_ia=False)
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_GI)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_II)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)

    cl_cs_theory = cl_GG + 2 * cl_GI + cl_II

    # print("IDS: ", id(s0), id(s1))

    assert np.allclose(cells_GG, cells_GG_m, atol=0, rtol=1e-127)

    assert np.allclose(cl_GG, cells_GG, atol=0, rtol=1e-7)
    assert np.allclose(cl_GG, cells_GG_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total_m, atol=0, rtol=1e-7)
