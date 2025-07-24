"""
Tests for the perturbation theory systematics for both
Weak Lensing and Number Counts.
"""

import os

import pytest

import numpy as np
import numpy.typing as npt
import pyccl as ccl
import pyccl.nl_pt as pt
import sacc

from firecrown.updatable import get_default_params_map
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
import firecrown.metadata_types as mdt
from firecrown.likelihood.two_point import (
    TwoPoint,
    TracerNames,
)
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter
import firecrown.parameters as fcp


@pytest.fixture(name="weak_lensing_source")
def fixture_weak_lensing_source() -> wl.WeakLensing:
    ia_systematic = wl.TattAlignmentSystematic()
    pzshift = wl.PhotoZShift(sacc_tracer="src0")
    return wl.WeakLensing(sacc_tracer="src0", systematics=[pzshift, ia_systematic])


@pytest.fixture(name="number_counts_source")
def fixture_number_counts_source() -> nc.NumberCounts:
    pzshift = nc.PhotoZShift(sacc_tracer="lens0")
    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")
    nl_bias = nc.PTNonLinearBiasSystematic(sacc_tracer="lens0")
    return nc.NumberCounts(
        sacc_tracer="lens0", has_rsd=True, systematics=[pzshift, magnification, nl_bias]
    )


@pytest.fixture(name="sacc_data")
def fixture_sacc_data() -> sacc.Sacc:
    # Load sacc file
    # This shouldn't be necessary, since we only use the n(z) from the sacc file
    saccfile = os.path.join(
        os.path.split(__file__)[0],
        "../examples/des_y1_3x2pt/sacc_data.fits",
    )
    return sacc.Sacc.load_fits(saccfile)


def test_pt_systematics(weak_lensing_source, number_counts_source, sacc_data):
    # The following disabling of pylint warnings are TEMPORARY. Disabling warnings is
    # generally not a good practice. In this case, the warnings are indicating that this
    # test is too complicated.
    #
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-statements
    stats = [
        TwoPoint("galaxy_shear_xi_plus", weak_lensing_source, weak_lensing_source),
        TwoPoint("galaxy_shear_xi_minus", weak_lensing_source, weak_lensing_source),
        TwoPoint("galaxy_shearDensity_xi_t", number_counts_source, weak_lensing_source),
        TwoPoint("galaxy_density_xi", number_counts_source, number_counts_source),
    ]

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    src0_tracer = sacc_data.get_tracer("src0")
    lens0_tracer = sacc_data.get_tracer("lens0")
    z, nz = src0_tracer.z, src0_tracer.nz
    lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    pt_calculator = pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=4,
        cosmo=ccl_cosmo,
    )
    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator,
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8),
    )
    params = get_default_params_map(modeling_tools)
    modeling_tools.update(params)
    modeling_tools.prepare()

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5

    b_1 = 2.0
    b_2 = 1.0
    b_s = 1.0

    mag_bias = 1.0

    c_1, c_d, c_2 = pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
    )

    # Code that creates Pk2D objects:
    ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_d))
    ptt_m = pt.PTMatterTracer()
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s)
    # IA
    pk_im = pt_calculator.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_m)
    pk_ii = pt_calculator.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_i)
    pk_gi = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_i)
    # Galaxies
    pk_gm = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_m)
    pk_gg = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_g)

    # Set the parameters for our systematics
    params.update(
        {
            "ia_a_1": a_1,
            "ia_a_2": a_2,
            "ia_a_d": a_d,
            "ia_zpiv_1": 0.0,
            "ia_zpiv_2": 0.0,
            "ia_zpiv_d": 0.0,
            "ia_alphaz_1": 0.0,
            "ia_alphaz_2": 0.0,
            "ia_alphaz_d": 0.0,
            "lens0_bias": b_1,
            "lens0_b_2": b_2,
            "lens0_b_s": b_s,
            "lens0_mag_bias": mag_bias,
            "src0_delta_z": 0.000,
            "lens0_delta_z": 0.000,
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

    # TODO:  We need to have a way to test systematics without requiring
    #  digging  into the innards of a likelihood object.
    s0 = likelihood.statistics[0].statistic
    assert isinstance(s0, TwoPoint)
    ells = s0.ells_for_xi
    cells_GG = s0.cells[TracerNames("shear", "shear")]
    cells_GI = s0.cells[TracerNames("intrinsic_pt", "shear")]
    cells_II = s0.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
    cells_cs_total = s0.cells[mdt.TRACER_NAMES_TOTAL]

    s1 = likelihood.statistics[1].statistic
    # del weak_lensing_source.cosmo_hash
    s1.compute_theory_vector(modeling_tools)
    assert isinstance(s1, TwoPoint)
    ells = s1.ells_for_xi
    cells_GG_m = s1.cells[TracerNames("shear", "shear")]
    cells_GI_m = s1.cells[TracerNames("shear", "intrinsic_pt")]
    cells_II_m = s1.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
    cells_cs_total_m = s1.cells[mdt.TRACER_NAMES_TOTAL]

    # print(list(likelihood.statistics[2].cells.keys()))
    s2 = likelihood.statistics[2].statistic
    assert isinstance(s2, TwoPoint)
    cells_gG = s2.cells[TracerNames("galaxies", "shear")]
    cells_gI = s2.cells[TracerNames("galaxies", "intrinsic_pt")]
    cells_mI = s2.cells[TracerNames("magnification+rsd", "intrinsic_pt")]

    # print(list(likelihood.statistics[3].cells.keys()))
    s3 = likelihood.statistics[3].statistic
    assert isinstance(s3, TwoPoint)
    cells_gg = s3.cells[TracerNames("galaxies", "galaxies")]
    cells_gm = s3.cells[TracerNames("galaxies", "magnification+rsd")]
    cells_mm = s3.cells[TracerNames("magnification+rsd", "magnification+rsd")]
    cells_gg_total = s3.cells[mdt.TRACER_NAMES_TOTAL]
    # pylint: enable=no-member
    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    t_g = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=False,
        dndz=(lens_z, lens_nz),
        bias=(lens_z, np.ones_like(lens_z)),
    )
    t_m = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=True,
        dndz=(lens_z, lens_nz),
        bias=None,
        mag_bias=(lens_z, mag_bias * np.ones_like(lens_z)),
    )
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)

    # Galaxies
    cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells, p_of_k_a=pk_gm)
    cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells, p_of_k_a=pk_gi)
    cl_gg = ccl.angular_cl(ccl_cosmo, t_g, t_g, ells, p_of_k_a=pk_gg)
    # Magnification
    cl_mI = ccl.angular_cl(ccl_cosmo, t_m, t_ia, ells, p_of_k_a=pk_im)
    cl_gm = ccl.angular_cl(ccl_cosmo, t_g, t_m, ells, p_of_k_a=pk_gm)
    cl_mm = ccl.angular_cl(ccl_cosmo, t_m, t_m, ells)

    cl_cs_theory = cl_GG + 2 * cl_GI + cl_II
    cl_gg_theory = cl_gg + 2 * cl_gm + cl_mm

    # print("IDS: ", id(s0), id(s1))

    assert np.allclose(cells_GG, cells_GG_m, atol=0, rtol=1e-127)

    assert np.allclose(cl_GG, cells_GG, atol=0, rtol=1e-7)
    assert np.allclose(cl_GG, cells_GG_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_gG, cells_gG, atol=0, rtol=1e-7)
    assert np.allclose(cl_gI, cells_gI, atol=0, rtol=1e-7)
    assert np.allclose(cl_gg, cells_gg, atol=0, rtol=1e-7)
    assert np.allclose(cl_mI, cells_mI, atol=0, rtol=1e-7)
    assert np.allclose(cl_gm, cells_gm, atol=0, rtol=1e-7)
    assert np.allclose(cl_mm, cells_mm, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_gg_theory, cells_gg_total, atol=0, rtol=1e-7)


def test_pt_mixed_systematics(sacc_data):
    # The following disabling of pylint warnings are TEMPORARY. Disabling warnings is
    # generally not a good practice. In this case, the warnings are indicating that this
    # test is too complicated.
    #
    # pylint: disable-msg=too-many-locals

    ia_systematic = wl.TattAlignmentSystematic()
    wl_source = wl.WeakLensing(sacc_tracer="src0", systematics=[ia_systematic])

    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")
    nc_source = nc.NumberCounts(
        sacc_tracer="lens0", has_rsd=True, systematics=[magnification]
    )

    stat = TwoPoint(
        source0=nc_source,
        source1=wl_source,
        sacc_data_type="galaxy_shearDensity_xi_t",
    )

    # Create the likelihood from the statistics
    likelihood = ConstGaussian(statistics=[stat])
    likelihood.read(sacc_data)

    src0_tracer = sacc_data.get_tracer("src0")
    lens0_tracer = sacc_data.get_tracer("lens0")
    z, nz = src0_tracer.z, src0_tracer.nz
    lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    pt_calculator = pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=4,
        cosmo=ccl_cosmo,
    )
    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator,
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8),
    )
    params = get_default_params_map(modeling_tools)
    modeling_tools.update(params)
    modeling_tools.prepare()

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5

    bias = 2.0
    mag_bias = 1.0

    c_1, c_d, c_2 = pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
    )

    # Code that creates Pk2D objects:
    ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_d))
    ptt_m = pt.PTMatterTracer()
    # IA
    pk_mi = pt_calculator.get_biased_pk2d(tracer1=ptt_m, tracer2=ptt_i)

    # Set the parameters for our systematics
    params.update(
        {
            "ia_a_1": a_1,
            "ia_a_2": a_2,
            "ia_a_d": a_d,
            "ia_zpiv_1": 0.0,
            "ia_zpiv_2": 0.0,
            "ia_zpiv_d": 0.0,
            "ia_alphaz_1": 0.0,
            "ia_alphaz_2": 0.0,
            "ia_alphaz_d": 0.0,
            "lens0_bias": bias,
            "lens0_mag_bias": mag_bias,
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

    # print(list(likelihood.statistics[2].cells.keys()))
    cells_gG = s0.cells[TracerNames("galaxies+magnification+rsd", "shear")]
    cells_gI = s0.cells[TracerNames("galaxies+magnification+rsd", "intrinsic_pt")]
    # pylint: enable=no-member

    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    t_g = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=True,
        dndz=(lens_z, lens_nz),
        bias=(lens_z, bias * np.ones_like(lens_z)),
        mag_bias=(lens_z, mag_bias * np.ones_like(lens_z)),
    )

    # Galaxies
    cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells)
    cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells, p_of_k_a=pk_mi)

    assert np.allclose(cl_gG, cells_gG, atol=0, rtol=1e-7)
    assert np.allclose(cl_gI, cells_gI, atol=0, rtol=1e-7)


def test_pt_mixed_systematics_zdep(sacc_data):
    # The following disabling of pylint warnings are TEMPORARY. Disabling warnings is
    # generally not a good practice. In this case, the warnings are indicating that this
    # test is too complicated.
    #
    # pylint: disable-msg=too-many-locals

    ia_systematic = wl.TattAlignmentSystematic()
    wl_source = wl.WeakLensing(sacc_tracer="src0", systematics=[ia_systematic])

    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")
    nc_source = nc.NumberCounts(
        sacc_tracer="lens0", has_rsd=True, systematics=[magnification]
    )

    stat = TwoPoint(
        source0=nc_source,
        source1=wl_source,
        sacc_data_type="galaxy_shearDensity_xi_t",
    )

    # Create the likelihood from the statistics
    likelihood = ConstGaussian(statistics=[stat])
    likelihood.read(sacc_data)

    src0_tracer = sacc_data.get_tracer("src0")
    lens0_tracer = sacc_data.get_tracer("lens0")
    z, nz = src0_tracer.z, src0_tracer.nz
    lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    pt_calculator = pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=4,
        cosmo=ccl_cosmo,
    )
    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator,
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8),
    )
    params = get_default_params_map(modeling_tools)
    modeling_tools.update(params)
    modeling_tools.prepare()

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5

    a_1_zpiv = a_2_zpiv = a_d_zpiv = 0.62
    a_1_alpha = a_2_alpha = a_d_alpha = 1.0

    bias = 2.0
    mag_bias = 1.0

    c_1, c_d, c_2 = pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
    )

    c_1_z = c_1 * ((1.0 + z) / (1.0 + a_1_zpiv)) ** a_1_alpha
    c_2_z = c_2 * ((1.0 + z) / (1.0 + a_2_zpiv)) ** a_2_alpha
    c_d_z = c_d * ((1.0 + z) / (1.0 + a_d_zpiv)) ** a_d_alpha
    # Code that creates Pk2D objects:
    ptt_i = pt.PTIntrinsicAlignmentTracer(
        c1=(z, c_1_z), c2=(z, c_2_z), cdelta=(z, c_d_z)
    )
    ptt_m = pt.PTMatterTracer()
    # IA
    pk_mi = pt_calculator.get_biased_pk2d(tracer1=ptt_m, tracer2=ptt_i)

    # Set the parameters for our systematics
    params.update(
        {
            "ia_a_1": a_1,
            "ia_a_2": a_2,
            "ia_a_d": a_d,
            "ia_zpiv_1": a_1_zpiv,
            "ia_zpiv_2": a_2_zpiv,
            "ia_zpiv_d": a_d_zpiv,
            "ia_alphaz_1": a_1_alpha,
            "ia_alphaz_2": a_2_alpha,
            "ia_alphaz_d": a_d_alpha,
            "lens0_bias": bias,
            "lens0_mag_bias": mag_bias,
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

    # print(list(likelihood.statistics[2].cells.keys()))
    cells_gG = s0.cells[TracerNames("galaxies+magnification+rsd", "shear")]
    cells_gI = s0.cells[TracerNames("galaxies+magnification+rsd", "intrinsic_pt")]
    # pylint: enable=no-member

    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    t_g = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=True,
        dndz=(lens_z, lens_nz),
        bias=(lens_z, bias * np.ones_like(lens_z)),
        mag_bias=(lens_z, mag_bias * np.ones_like(lens_z)),
    )

    # Galaxies
    cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells)
    cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells, p_of_k_a=pk_mi)

    assert np.allclose(cl_gG, cells_gG, atol=0, rtol=1e-7)
    assert np.allclose(cl_gI, cells_gI, atol=0, rtol=1e-7)


def test_pt_systematics_zdep(weak_lensing_source, number_counts_source, sacc_data):
    # The following disabling of pylint warnings are TEMPORARY. Disabling warnings is
    # generally not a good practice. In this case, the warnings are indicating that this
    # test is too complicated.
    #
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-statements
    stats = [
        TwoPoint("galaxy_shear_xi_plus", weak_lensing_source, weak_lensing_source),
        TwoPoint("galaxy_shear_xi_minus", weak_lensing_source, weak_lensing_source),
        TwoPoint("galaxy_shearDensity_xi_t", number_counts_source, weak_lensing_source),
        TwoPoint("galaxy_density_xi", number_counts_source, number_counts_source),
    ]

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)
    src0_tracer = sacc_data.get_tracer("src0")
    lens0_tracer = sacc_data.get_tracer("lens0")
    z, nz = src0_tracer.z, src0_tracer.nz
    lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    pt_calculator = pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=4,
        cosmo=ccl_cosmo,
    )
    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator,
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8),
    )
    params = get_default_params_map(modeling_tools)
    modeling_tools.update(params)
    modeling_tools.prepare()

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5

    a_1_zpiv = a_2_zpiv = a_d_zpiv = 0.62
    a_1_alpha = a_2_alpha = a_d_alpha = 1.0

    b_1 = 2.0
    b_2 = 1.0
    b_s = 1.0

    mag_bias = 1.0

    c_1, c_d, c_2 = pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
    )

    c_1_z = c_1 * ((1.0 + z) / (1.0 + a_1_zpiv)) ** a_1_alpha
    c_2_z = c_2 * ((1.0 + z) / (1.0 + a_2_zpiv)) ** a_2_alpha
    c_d_z = c_d * ((1.0 + z) / (1.0 + a_d_zpiv)) ** a_d_alpha
    # Code that creates Pk2D objects:
    ptt_i = pt.PTIntrinsicAlignmentTracer(
        c1=(z, c_1_z), c2=(z, c_2_z), cdelta=(z, c_d_z)
    )
    ptt_m = pt.PTMatterTracer()
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s)
    # IA
    pk_im = pt_calculator.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_m)
    pk_ii = pt_calculator.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_i)
    pk_gi = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_i)
    # Galaxies
    pk_gm = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_m)
    pk_gg = pt_calculator.get_biased_pk2d(tracer1=ptt_g, tracer2=ptt_g)

    # Set the parameters for our systematics
    params.update(
        {
            "ia_a_1": a_1,
            "ia_a_2": a_2,
            "ia_a_d": a_d,
            "ia_zpiv_1": a_1_zpiv,
            "ia_zpiv_2": a_2_zpiv,
            "ia_zpiv_d": a_d_zpiv,
            "ia_alphaz_1": a_1_alpha,
            "ia_alphaz_2": a_2_alpha,
            "ia_alphaz_d": a_d_alpha,
            "lens0_bias": b_1,
            "lens0_b_2": b_2,
            "lens0_b_s": b_s,
            "lens0_mag_bias": mag_bias,
            "src0_delta_z": 0.000,
            "lens0_delta_z": 0.000,
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

    # TODO:  We need to have a way to test systematics without requiring
    #  digging  into the innards of a likelihood object.
    s0 = likelihood.statistics[0].statistic
    assert isinstance(s0, TwoPoint)
    ells = s0.ells_for_xi
    cells_GG = s0.cells[TracerNames("shear", "shear")]
    cells_GI = s0.cells[TracerNames("intrinsic_pt", "shear")]
    cells_II = s0.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
    cells_cs_total = s0.cells[mdt.TRACER_NAMES_TOTAL]

    s1 = likelihood.statistics[1].statistic
    # del weak_lensing_source.cosmo_hash
    s1.compute_theory_vector(modeling_tools)
    assert isinstance(s1, TwoPoint)
    ells = s1.ells_for_xi
    cells_GG_m = s1.cells[TracerNames("shear", "shear")]
    cells_GI_m = s1.cells[TracerNames("shear", "intrinsic_pt")]
    cells_II_m = s1.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
    cells_cs_total_m = s1.cells[mdt.TRACER_NAMES_TOTAL]

    # print(list(likelihood.statistics[2].cells.keys()))
    s2 = likelihood.statistics[2].statistic
    assert isinstance(s2, TwoPoint)
    cells_gG = s2.cells[TracerNames("galaxies", "shear")]
    cells_gI = s2.cells[TracerNames("galaxies", "intrinsic_pt")]
    cells_mI = s2.cells[TracerNames("magnification+rsd", "intrinsic_pt")]

    # print(list(likelihood.statistics[3].cells.keys()))
    s3 = likelihood.statistics[3].statistic
    assert isinstance(s3, TwoPoint)
    cells_gg = s3.cells[TracerNames("galaxies", "galaxies")]
    cells_gm = s3.cells[TracerNames("galaxies", "magnification+rsd")]
    cells_mm = s3.cells[TracerNames("magnification+rsd", "magnification+rsd")]
    cells_gg_total = s3.cells[mdt.TRACER_NAMES_TOTAL]
    # pylint: enable=no-member
    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, np.ones_like(z)),
        use_A_ia=False,
    )
    t_g = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=False,
        dndz=(lens_z, lens_nz),
        bias=(lens_z, np.ones_like(lens_z)),
    )
    t_m = ccl.NumberCountsTracer(
        ccl_cosmo,
        has_rsd=True,
        dndz=(lens_z, lens_nz),
        bias=None,
        mag_bias=(lens_z, mag_bias * np.ones_like(lens_z)),
    )
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)

    # Galaxies
    cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells, p_of_k_a=pk_gm)
    cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells, p_of_k_a=pk_gi)
    cl_gg = ccl.angular_cl(ccl_cosmo, t_g, t_g, ells, p_of_k_a=pk_gg)
    # Magnification
    cl_mI = ccl.angular_cl(ccl_cosmo, t_m, t_ia, ells, p_of_k_a=pk_im)
    cl_gm = ccl.angular_cl(ccl_cosmo, t_g, t_m, ells, p_of_k_a=pk_gm)
    cl_mm = ccl.angular_cl(ccl_cosmo, t_m, t_m, ells)

    cl_cs_theory = cl_GG + 2 * cl_GI + cl_II
    cl_gg_theory = cl_gg + 2 * cl_gm + cl_mm

    # print("IDS: ", id(s0), id(s1))

    assert np.allclose(cells_GG, cells_GG_m, atol=0, rtol=1e-127)

    assert np.allclose(cl_GG, cells_GG, atol=0, rtol=1e-7)
    assert np.allclose(cl_GG, cells_GG_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI, atol=0, rtol=1e-7)
    assert np.allclose(cl_GI, cells_GI_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II, atol=0, rtol=1e-7)
    assert np.allclose(cl_II, cells_II_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_gG, cells_gG, atol=0, rtol=1e-7)
    assert np.allclose(cl_gI, cells_gI, atol=0, rtol=1e-7)
    assert np.allclose(cl_gg, cells_gg, atol=0, rtol=1e-7)
    assert np.allclose(cl_mI, cells_mI, atol=0, rtol=1e-7)
    assert np.allclose(cl_gm, cells_gm, atol=0, rtol=1e-7)
    assert np.allclose(cl_mm, cells_mm, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total, atol=0, rtol=1e-7)
    assert np.allclose(cl_cs_theory, cells_cs_total_m, atol=0, rtol=1e-7)
    assert np.allclose(cl_gg_theory, cells_gg_total, atol=0, rtol=1e-7)


def test_linear_bias_systematic(tools_with_vanilla_cosmology: ModelingTools):
    a = nc.LinearBiasSystematic("xxx")
    assert isinstance(a, nc.LinearBiasSystematic)
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None
    assert not a.is_updated()
    a.update(fcp.ParamsMap({"xxx_alphag": 1.0, "xxx_alphaz": 2.0, "xxx_z_piv": 1.5}))
    assert a.is_updated()
    assert a.alphag == 1.0
    assert a.alphaz == 2.0
    assert a.z_piv == 1.5

    orig_nca = nc.NumberCountsArgs(
        z=np.array([0.5, 1.0]),
        dndz=np.array([5.0, 4.0]),
        bias=np.array([1.0, 1.0]),
        mag_bias=(np.array([2.0, 3.0]), np.array([4.0, 5.0])),
        has_pt=False,
        has_hm=False,
        b_2=(np.array([5.0, 6.0]), np.array([6.0, 7.0])),
        b_s=(np.array([7.0, 8.0]), np.array([8.0, 9.0])),
    )

    nca = a.apply(tools_with_vanilla_cosmology, orig_nca)
    # Answer values determined by code inspection and hand calculation.
    expected_bias: npt.NDArray[np.float64] = np.array([0.27835299, 0.39158961])
    assert nca.bias is not None  # needed for mypy
    new_bias: npt.NDArray[np.float64] = nca.bias  # needed for mypy
    assert np.allclose(expected_bias, new_bias)

    a.reset()
    assert not a.is_updated()
    assert a.parameter_prefix == "xxx"
    assert a.alphag is None
    assert a.alphaz is None
    assert a.z_piv is None
