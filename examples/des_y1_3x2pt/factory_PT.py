#!/usr/bin/env python

"""Example factory function for DES Y1 3x2pt likelihood."""
from dataclasses import dataclass
import os

import numpy as np
import sacc
import pyccl as ccl
import pyccl.nl_pt

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
import firecrown.metadata_types as mdt
from firecrown.likelihood.two_point import (
    TwoPoint,
    TracerNames,
)
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.likelihood import Likelihood
from firecrown.ccl_factory import CCLFactory
from firecrown.updatable import get_default_params_map


saccfile = os.path.expanduser(
    os.path.expandvars("${FIRECROWN_DIR}/examples/des_y1_3x2pt/sacc_data.fits")
)


@dataclass
class CclSetup:
    """A package of related CCL parameters.

    We use this to reduce the number of variables used in the :meth:`run_likelihood`
    method.
    """

    a_1: float = 1.0
    a_2: float = 0.5
    a_d: float = 0.5
    b_1: float = 2.0
    b_2: float = 1.0
    b_s: float = 1.0
    mag_bias: float = 1.0


@dataclass
class CElls:
    """A package of related C_ell values.

    This is used to reduce the number of variables used in the
    :meth:`run_likelihood` method.
    """

    GG: np.ndarray
    GI: np.ndarray
    II: np.ndarray
    cs_total: np.ndarray
    gG: np.ndarray
    gI: np.ndarray
    mI: np.ndarray
    gg: np.ndarray
    gm: np.ndarray
    gg_total: np.ndarray

    def __init__(self, stat0: TwoPoint, stat2: TwoPoint, stat3: TwoPoint):
        self.GG = stat0.cells[TracerNames("shear", "shear")]
        self.GI = stat0.cells[TracerNames("shear", "intrinsic_pt")]
        self.II = stat0.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
        self.cs_total = stat0.cells[mdt.TRACER_NAMES_TOTAL]

        self.gG = stat2.cells[TracerNames("galaxies", "shear")]
        self.gI = stat2.cells[TracerNames("galaxies", "intrinsic_pt")]
        self.mI = stat2.cells[TracerNames("magnification+rsd", "intrinsic_pt")]

        self.gg = stat3.cells[TracerNames("galaxies", "galaxies")]
        self.gm = stat3.cells[TracerNames("galaxies", "magnification+rsd")]
        self.gg_total = stat3.cells[mdt.TRACER_NAMES_TOTAL]


def build_likelihood(_) -> tuple[Likelihood, ModelingTools]:
    """Likelihood factory function for DES Y1 3x2pt analysis."""
    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # Define the intrinsic alignment systematic. This will be added to the
    # lensing sources later
    ia_systematic = wl.TattAlignmentSystematic()

    # Define the photo-z shift systematic.
    src_pzshift = wl.PhotoZShift(sacc_tracer="src0")

    # Create the weak lensing source, specifying the name of the tracer in the
    # sacc file and a list of systematics
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[src_pzshift, ia_systematic])

    lens_pzshift = nc.PhotoZShift(sacc_tracer="lens0")
    magnification = nc.ConstantMagnificationBiasSystematic(sacc_tracer="lens0")
    nl_bias = nc.PTNonLinearBiasSystematic(sacc_tracer="lens0")
    lens0 = nc.NumberCounts(
        sacc_tracer="lens0",
        has_rsd=True,
        systematics=[lens_pzshift, magnification, nl_bias],
    )

    # Define the statistics we like to include in the likelihood
    # The only place the dict 'stats' gets used, other than setting values in
    # it, is to call 'values' on it. Thus we don't need a dict, we need a list
    # of the values. The keys assigned to the dict are never used.
    xip_src0_src0 = TwoPoint(
        source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_plus"
    )
    xim_src0_src0 = TwoPoint(
        source0=src0, source1=src0, sacc_data_type="galaxy_shear_xi_minus"
    )

    gammat_lens0_src0 = TwoPoint(
        source0=lens0,
        source1=src0,
        sacc_data_type="galaxy_shearDensity_xi_t",
    )

    wtheta_lens0_lens0 = TwoPoint(
        source0=lens0,
        source1=lens0,
        sacc_data_type="galaxy_density_xi",
    )

    # Create the likelihood from the statistics
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator, ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )
    # Note that the ordering of the statistics is relevant, because the data
    # vector, theory vector, and covariance matrix will be organized to
    # follow the order used here.
    likelihood = ConstGaussian(
        statistics=[xip_src0_src0, xim_src0_src0, gammat_lens0_src0, wtheta_lens0_lens0]
    )

    # Read the two-point data from the sacc file
    likelihood.read(sacc_data)

    # an object called "likelihood" must be defined
    print(
        "Using parameters:", list(likelihood.required_parameters().get_params_names())
    )

    # To allow this likelihood to be used in cobaya or cosmosis,
    # return the likelihood object
    return likelihood, modeling_tools


# We can also run the likelihood directly
def run_likelihood() -> None:
    """Produce plots using the likelihood function built by :meth:`build_likelihood`."""
    # pylint: enable=import-outside-toplevel

    likelihood, tools = build_likelihood(None)

    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(saccfile)

    src0_tracer = sacc_data.get_tracer("src0")
    lens0_tracer = sacc_data.get_tracer("lens0")
    z, nz = src0_tracer.z, src0_tracer.nz
    lens_z, lens_nz = lens0_tracer.z, lens0_tracer.nz

    # Setting cosmological parameters
    cosmo_params = get_default_params_map(tools)
    tools.update(cosmo_params)
    tools.prepare()
    ccl_cosmo = tools.get_ccl_cosmology()

    cs = CclSetup()
    c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=cs.a_1, a1delta=cs.a_d, a2=cs.a_2, Om_m2_for_c2=False
    )

    # Code that creates a Pk2D object:
    ptc = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
        cosmo=ccl_cosmo,
    )
    ptt_i = pyccl.nl_pt.PTIntrinsicAlignmentTracer(
        c1=(z, c_1), c2=(z, c_2), cdelta=(z, c_d)
    )
    ptt_m = pyccl.nl_pt.PTMatterTracer()
    ptt_g = pyccl.nl_pt.PTNumberCountsTracer(b1=cs.b_1, b2=cs.b_2, bs=cs.b_s)
    # IA
    pk_gi, pk_ii, pk_im = get_ia(ptc, ptt_g, ptt_i, ptt_m)
    # Galaxies
    pk_gg, pk_gm = get_galaxies(ptc, ptt_g, ptt_m)
    # Magnification: just a matter-matter P(k)
    pk_mm = ptc.get_biased_pk2d(ptt_m, tracer2=ptt_m)

    # Set the parameters for our systematics
    systematics_params = ParamsMap(
        {
            "ia_a_1": cs.a_1,
            "ia_a_2": cs.a_2,
            "ia_a_d": cs.a_d,
            "lens0_bias": cs.b_1,
            "lens0_b_2": cs.b_2,
            "lens0_b_s": cs.b_s,
            "lens0_mag_bias": cs.mag_bias,
            "src0_delta_z": 0.000,
            "lens0_delta_z": 0.000,
        }
    )

    # Apply the systematics parameters
    likelihood.update(systematics_params)

    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(tools)

    print(f"Log-like = {log_like:.1f}")

    assert isinstance(likelihood, ConstGaussian)
    assert likelihood.cov is not None

    stat0 = likelihood.statistics[0].statistic
    assert isinstance(stat0, TwoPoint)

    # x = likelihood.statistics[0].ell_or_theta_
    # y_data = likelihood.statistics[0].measured_statistic_

    # y_err = np.sqrt(np.diag(likelihood.cov))[: len(x)]
    # y_theory = likelihood.statistics[0].predicted_statistic_

    print(list(stat0.cells.keys()))

    stat2 = likelihood.statistics[2].statistic  # pylint: disable=no-member
    assert isinstance(stat2, TwoPoint)
    print(list(stat2.cells.keys()))

    stat3 = likelihood.statistics[3].statistic  # pylint: disable=no-member
    assert isinstance(stat3, TwoPoint)
    print(list(stat3.cells.keys()))

    plot_predicted_and_measured_statistics(
        ccl_cosmo,
        cs,
        lens_nz,
        lens_z,
        nz,
        pk_gg,
        pk_gi,
        pk_gm,
        pk_ii,
        pk_im,
        pk_mm,
        stat0,
        stat2,
        stat3,
        z,
    )


def get_galaxies(
    ptc: pyccl.nl_pt.EulerianPTCalculator,
    ptt_g: pyccl.nl_pt.PTNumberCountsTracer,
    ptt_m: pyccl.nl_pt.PTMatterTracer,
) -> tuple[pyccl.Pk2D, pyccl.Pk2D]:
    """
     Compute the predicted power spectra for galaxies.

    :param ptc: The perturbation theory calculator.
    :param ptt_g: The number counts tracer.
    :param ptt_m: The matter tracer.
    :return: The galaxy-matter and galaxy-galaxy power spectra.
    """
    pk_gm = ptc.get_biased_pk2d(ptt_g, tracer2=ptt_m)
    pk_gg = ptc.get_biased_pk2d(ptt_g, tracer2=ptt_g)
    return pk_gg, pk_gm


def get_ia(
    ptc: pyccl.nl_pt.EulerianPTCalculator,
    ptt_g: pyccl.nl_pt.PTNumberCountsTracer,
    ptt_i: pyccl.nl_pt.PTIntrinsicAlignmentTracer,
    ptt_m: pyccl.nl_pt.PTMatterTracer,
) -> tuple[pyccl.Pk2D, pyccl.Pk2D, pyccl.Pk2D]:
    """
    Compute the predicted power spectra for intrinsic alignments.

    :param ptc: The perturbation theory calculator.
    :param ptt_g: The number counts tracer.
    :param ptt_i: The intrinsic alignment tracer.
    :param ptt_m: The matter tracer.
    :return: The galaxy-intrinsic alignment, intrinsic-intrinsic, and
             intrinsic-matter power spectra.
    """
    pk_im = ptc.get_biased_pk2d(ptt_i, tracer2=ptt_m)
    pk_ii = ptc.get_biased_pk2d(ptt_i, tracer2=ptt_i)
    pk_gi = ptc.get_biased_pk2d(ptt_g, tracer2=ptt_i)
    return pk_gi, pk_ii, pk_im


def plot_predicted_and_measured_statistics(
    ccl_cosmo,
    cs,
    lens_nz,
    lens_z,
    nz,
    pk_gg,
    pk_gi,
    pk_gm,
    pk_ii,
    pk_im,
    pk_mm,
    stat0,
    stat2,
    stat3,
    z,
):
    """Plot the predictions and measurements."""
    # We do imports here to save a bit of time when importing this module but
    # not using the run_likelihood function.
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    ells = stat0.ells_for_xi
    cells = CElls(stat0, stat2, stat3)

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
        mag_bias=(lens_z, cs.mag_bias * np.ones_like(lens_z)),
    )
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)
    # Galaxies
    cl_gG, cl_gI, cl_gg = get_galaxies_angular_cl(
        ccl_cosmo, ells, pk_gg, pk_gi, pk_gm, t_g, t_ia, t_lens
    )
    # Magnification
    cl_gm, cl_mI, cl_mm = get_magnification_angular_cl(
        ccl_cosmo, ells, pk_gm, pk_im, pk_mm, t_g, t_ia, t_m
    )
    # The observed angular power spectrum is the sum of the two.
    cl_cs_theory = cl_GG + 2 * cl_GI + cl_II
    cl_gg_theory = cl_gg + 2 * cl_gm + cl_mm
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    fig.subplots_adjust(hspace=0)
    # ax[0].plot(x, y_theory, label="Total")
    assert isinstance(ax, np.ndarray)
    assert isinstance(ax[0], plt.Axes)
    assert isinstance(ax[1], plt.Axes)
    ax[0].plot(ells, cells.GG, label="GG firecrown")
    ax[0].plot(ells, cl_GG, ls="--", label="GG CCL")
    ax[0].plot(ells, -cells.GI, label="-GI firecrown")
    ax[0].plot(ells, -cl_GI, ls="--", label="-GI CCL")
    ax[0].plot(ells, cells.II, label="II firecrown")
    ax[0].plot(ells, cl_II, ls="--", label="II CCL")
    ax[0].plot(ells, -cells.gI, label="-Ig firecrown")
    ax[0].plot(ells, -cl_gI, ls="--", label="-Ig CCL")
    ax[0].plot(ells, cells.cs_total, label="total CS firecrown")
    ax[0].plot(ells, cl_cs_theory, ls="--", label="total CS CCL")
    ax[1].plot(ells, cells.gG, label="Gg firecrown")
    ax[1].plot(ells, cl_gG, ls="--", label="Gg CCL")
    ax[1].plot(ells, cells.gg, label="gg firecrown")
    ax[1].plot(ells, cl_gg, ls="--", label="gg CCL")
    ax[1].plot(ells, -cells.mI, label="-mI firecrown")
    ax[1].plot(ells, -cl_mI, ls="--", label="-mI CCL")
    ax[1].plot(ells, cells.gm, label="gm firecrown")
    ax[1].plot(ells, cl_gm, ls="--", label="gm CCL")
    ax[1].plot(ells, cells.gg_total, label="total gg firecrown")
    ax[1].plot(ells, cl_gg_theory, ls="--", label="total gg CCL")
    # ax[0].errorbar(x, y_data, y_err, ls="none", marker="o")
    ax[0].set_xscale("log")
    ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylabel(r"$C_\ell$")
    for a in ax:
        a.set_yscale("log")
        a.set_ylabel(r"$C_\ell$")
        a.legend(fontsize="small")
    fig.suptitle("PT Cls, including IA, galaxy bias, magnification")
    fig.savefig("pt_cls.png", facecolor="white", dpi=300)
    plt.show()


def get_magnification_angular_cl(
    ccl_cosmo: pyccl.Cosmology,
    ells: list[int],
    pk_gm: pyccl.Pk2D,
    pk_im: pyccl.Pk2D,
    pk_mm: pyccl.Pk2D,
    t_g: pyccl.Tracer,
    t_ia: pyccl.Tracer,
    t_m: pyccl.Tracer,
) -> tuple[pyccl.Pk2D, pyccl.Pk2D, pyccl.Pk2D]:
    """
    Compute the predicted power spectra for magnification.

    :param ccl_cosmo: The CCL cosmology object.
    :param ells: The angular wavenumbers at which to compute the power spectrum.
    :param pk_gm: The galaxy-magnification power spectrum.
    :param pk_im: The intrinsic alignment-magnification power spectrum.
    :param pk_mm: The magnification-magnification power spectrum.
    :param t_g: The galaxy number counts tracer.
    :param t_ia: The intrinsic alignment tracer.
    :param t_m: The magnification tracer.
    :return: The galaxy-magnification, intrinsic alignment-magnification, and
             magnification-magnification power spectra.
    """
    cl_mI = ccl.angular_cl(ccl_cosmo, t_m, t_ia, ells, p_of_k_a=pk_im)
    cl_gm = ccl.angular_cl(ccl_cosmo, t_g, t_m, ells, p_of_k_a=pk_gm)
    cl_mm = ccl.angular_cl(ccl_cosmo, t_m, t_m, ells, p_of_k_a=pk_mm)
    return cl_gm, cl_mI, cl_mm


def get_galaxies_angular_cl(
    ccl_cosmo: pyccl.Cosmology,
    ells: list[int],
    pk_gg: pyccl.Pk2D,
    pk_gi: pyccl.Pk2D,
    pk_gm: pyccl.Pk2D,
    t_g: pyccl.Tracer,
    t_ia: pyccl.Tracer,
    t_lens: pyccl.Tracer,
) -> tuple[pyccl.Pk2D, pyccl.Pk2D, pyccl.Pk2D]:
    """
    Compute the predicted power spectra for galaxies.

    :param ccl_cosmo: The CCL cosmology object.
    :param ells: The angular wavenumbers at which to compute the power spectrum.
    :param pk_gg: The galaxy-galaxy power spectrum.
    :param pk_gi: The galaxy-intrinsic alignment power spectrum.
    :param pk_gm: The galaxy-magnification power spectrum.
    :param t_g: The galaxy number counts tracer.
    :param t_ia: The intrinsic alignment tracer.
    :param t_lens: The lensing tracer.
    :return: The galaxy-galaxy, galaxy-intrinsic, and galaxy-lensing power spectra.
    """
    cl_gG = ccl.angular_cl(ccl_cosmo, t_g, t_lens, ells, p_of_k_a=pk_gm)
    cl_gI = ccl.angular_cl(ccl_cosmo, t_g, t_ia, ells, p_of_k_a=pk_gi)
    cl_gg = ccl.angular_cl(ccl_cosmo, t_g, t_g, ells, p_of_k_a=pk_gg)
    return cl_gG, cl_gI, cl_gg


if __name__ == "__main__":
    run_likelihood()
