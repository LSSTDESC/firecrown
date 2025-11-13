"""Example of a Firecrown likelihood using the DES Y1 cosmic shear data TATT."""

import os
import sacc
import pyccl as ccl
import pyccl.nl_pt

import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood import TwoPoint, ConstGaussian, Likelihood
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory
from firecrown.updatable import get_default_params_map
from firecrown.metadata_types import TracerNames, TRACER_NAMES_TOTAL

SACCFILE = os.path.expanduser(
    os.path.expandvars("${FIRECROWN_DIR}/examples/des_y1_3x2pt/sacc_data.hdf5")
)


def build_likelihood(_) -> tuple[Likelihood, ModelingTools]:
    """Build the likelihood for the DES Y1 cosmic shear data TATT."""
    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(SACCFILE)

    n_source = 1
    stats = define_stats(n_source)

    # Create the likelihood from the statistics
    pt_calculator = pyccl.nl_pt.EulerianPTCalculator(
        with_NC=False,
        with_IA=True,
        # with_dd=True,
        log10k_min=-4,
        log10k_max=2,
        nk_per_decade=20,
    )

    modeling_tools = ModelingTools(
        pt_calculator=pt_calculator, ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )
    likelihood = ConstGaussian(statistics=list(stats.values()))

    # Read the two-point data from the sacc file
    likelihood.read(sacc_data)

    # To allow this likelihood to be used in cobaya or cosmosis, define a
    # an object called "likelihood" must be defined
    print(
        "Using parameters:", list(likelihood.required_parameters().get_params_names())
    )

    return likelihood, modeling_tools


def define_stats(n_source):
    """Define the TwoPoint objects to be returned by this factory furnciton."""
    sources = define_sources(n_source)
    stats = {}
    for stat, sacc_stat in [
        ("xip", "galaxy_shear_xi_plus"),
        ("xim", "galaxy_shear_xi_minus"),
    ]:
        for i in range(n_source):
            for j in range(i, n_source):
                # Define two-point statistics, given two sources (from above) and
                # the type of statistic.
                stats[f"{stat}_src{i}_src{j}"] = TwoPoint(
                    source0=sources[f"src{i}"],
                    source1=sources[f"src{j}"],
                    sacc_data_type=sacc_stat,
                )
    return stats


def define_sources(n_source):
    """Return the sources to be used by the factory function."""
    result = {}
    # Define the intrinsic alignment systematic. This will be added to the
    # lensing restult later
    ia_systematic = wl.TattAlignmentSystematic(include_z_dependence=True)
    for i in range(n_source):
        # Define the photo-z shift systematic.
        pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")

        # Create the weak lensing source, specifying the name of the tracer in the
        # sacc file and a list of systematics
        result[f"src{i}"] = wl.WeakLensing(
            sacc_tracer=f"src{i}", systematics=[pzshift, ia_systematic]
        )
    return result


# We can also run the likelihood directly
def run_likelihood() -> None:
    """Run the likelihood."""
    likelihood, tools = build_likelihood(None)

    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(SACCFILE)

    src0_tracer = sacc_data.get_tracer("src0")
    z, nz = src0_tracer.z, src0_tracer.nz

    # Bare CCL setup
    a_1 = 1.0
    a_2 = 0.5
    a_d = 0.5
    # Set the parameters for our systematics
    systematics_params = {
        "ia_a_1": a_1,
        "ia_a_2": a_2,
        "ia_a_d": a_d,
        "ia_alphaz_1": 0.0,
        "ia_alphaz_2": 0.0,
        "ia_alphaz_d": 0.0,
        "ia_zpiv_1": 0.62,
        "ia_zpiv_2": 0.62,
        "ia_zpiv_d": 0.62,
        "src0_delta_z": 0.000,
        "src1_delta_z": 0.003,
        "src2_delta_z": -0.001,
        "src3_delta_z": 0.002,
    }
    # Prepare the cosmology object
    params = ParamsMap(get_default_params_map(tools).params | systematics_params)

    # Apply the systematics parameters
    likelihood.update(params)

    # Prepare the cosmology object
    tools.update(params)
    tools.prepare()
    ccl_cosmo = tools.get_ccl_cosmology()

    c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
        ccl_cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2=False
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
    # IAs x matter
    pk_im = ptc.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_m)
    pk_ii = ptc.get_biased_pk2d(tracer1=ptt_i, tracer2=ptt_i)

    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(tools)

    print(f"Log-like = {log_like:.1f}")

    # Plot the predicted and measured statistic
    assert isinstance(likelihood, ConstGaussian)
    two_point_0 = likelihood.statistics[0].statistic
    assert isinstance(two_point_0, TwoPoint)

    # x = two_point_0.ell_or_theta_
    # y_data = two_point_0.measured_statistic_

    assert isinstance(likelihood, ConstGaussian)
    assert likelihood.cov is not None

    # y_err = np.sqrt(np.diag(likelihood.cov))[: len(x)]
    # y_theory = two_point_0.predicted_statistic_

    print(list(two_point_0.cells.keys()))

    make_plot(ccl_cosmo, nz, pk_ii, pk_im, two_point_0, z)


def make_plot(ccl_cosmo, nz, pk_ii, pk_im, two_point_0, z):
    """Create and show a diagnostic plot."""
    import numpy as np  # pylint: disable-msg=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable-msg=import-outside-toplevel

    ells = two_point_0.ells_for_xi
    cells_gg = two_point_0.cells[TracerNames("shear", "shear")]
    cells_gi = two_point_0.cells[TracerNames("shear", "intrinsic_pt")]
    cells_ii = two_point_0.cells[TracerNames("intrinsic_pt", "intrinsic_pt")]
    cells_total = two_point_0.cells[TRACER_NAMES_TOTAL]
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
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_im)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_ii)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)
    # The observed angular power spectrum is the sum of the two.
    cl_theory = (
        cl_GG + 2 * cl_GI + cl_II
    )  # normally we would also have a third term, +cl_II).
    # plt.plot(x, y_theory, label="Total")
    plt.plot(ells, cells_gg, label="GG firecrown")
    plt.plot(ells, cl_GG, ls="--", label="GG CCL")
    plt.plot(ells, -cells_gi, label="-GI firecrown")
    plt.plot(ells, -cl_GI, ls="--", label="-GI CCL")
    plt.plot(ells, cells_ii, label="II firecrown")
    plt.plot(ells, cl_II, ls="--", label="II CCL")
    plt.plot(ells, cells_total, label="total firecrown")
    plt.plot(ells, cl_theory, ls="--", label="total CCL")
    # plt.errorbar(x, y_data, y_err, ls="none", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.legend()
    # plt.xlim(right=5e3)
    # plt.ylim(bottom=1e-12)
    plt.title("TATT IA")
    plt.savefig("tatt.png", facecolor="white", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_likelihood()
