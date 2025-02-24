"""Example of a Firecrown likelihood using the DES Y1 cosmic shear data and
the halo model for intrinsic alignments."""
import os
import sacc
import pyccl as ccl

import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.likelihood import Likelihood
from firecrown.ccl_factory import CCLFactory
from firecrown.updatable import get_default_params_map
from firecrown.metadata_types import TracerNames, TRACER_NAMES_TOTAL

saccfile = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_DIR}/examples/des_y1_3x2pt/sacc_data.fits"
    )
)


def build_likelihood(_) -> tuple[Likelihood, ModelingTools]:
    """Build the likelihood for the DES Y1 cosmic shear data TATT."""
    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(saccfile)

    # Define sources
    n_source = 1
    sources = {}

    # Define the intrinsic alignment systematic. This will be added to the
    # lensing sources later
    ia_systematic = wl.HMAlignmentSystematic()

    for i in range(n_source):
        # Define the photo-z shift systematic.
        pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")

        # Create the weak lensing source, specifying the name of the tracer in the
        # sacc file and a list of systematics
        sources[f"src{i}"] = wl.WeakLensing(
            sacc_tracer=f"src{i}", systematics=[pzshift, ia_systematic]
        )

    # Define the statistics we like to include in the likelihood
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

    # Create the likelihood from the statistics
    # Define the halo model components. This is one solution but maybe not the best!
    hmd_200m = ccl.halos.MassDef200m
    cM = "Duffy08"
    nM = "Tinker10"
    bM = "Tinker10"

    modeling_tools = ModelingTools(
        hm_definition=hmd_200m, hm_function=nM, bias_function=bM, cM_relation=cM,
        ccl_factory = CCLFactory(require_nonlinear_pk=True)
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


# We can also run the likelihood directly
def run_likelihood() -> None:
    """Run the likelihood."""
    import numpy as np  # pylint: disable-msg=import-outside-toplevel

    likelihood, tools = build_likelihood(None)

    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(saccfile)

    src0_tracer = sacc_data.get_tracer("src0")
    z, nz = src0_tracer.z, src0_tracer.nz  # pylint: disable-msg=invalid-name

    # Define halo model amplitudes
    a_1h = 1e-3  # 1-halo alignment amplitude.
    a_2h = 1.0  # 2-halo alignment amplitude.

    # Set the parameters for our systematics
    systematics_params = ParamsMap(
        {
            "ia_a_1h": a_1h,
            "ia_a_2h": a_2h,
            "src0_delta_z": 0.000,
        }
    )

    # Prepare the cosmology object
    params = ParamsMap(get_default_params_map(tools) | systematics_params)

    # Apply the systematics parameters
    likelihood.update(params)

    # Prepare the cosmology object
    tools.update(params)
    tools.prepare()
    ccl_cosmo = tools.get_ccl_cosmology()

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo.compute_nonlin_power()

    # Code that creates a Pk2D object:
    k_arr = np.geomspace(1e-3, 1e3, 128)  # For evaluating
    a_arr = np.linspace(0.1, 1, 16)
    cM = ccl.halos.ConcentrationDuffy08(mass_def="200m")
    nM = ccl.halos.MassFuncTinker10(mass_def="200m")
    bM = ccl.halos.HaloBiasTinker10(mass_def="200m")
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def="200m")
    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(
        mass_def="200m", concentration=cM, a1h=a_1h, b=-2
    )
    # NFW profile for matter (G)
    NFW = ccl.halos.HaloProfileNFW(
        mass_def="200m", concentration=cM, truncated=True, fourier_analytic=True
    )
    pk_GI_1h = ccl.halos.halomod_Pk2D(
        ccl_cosmo,
        hmc,
        NFW,
        prof2=sat_gamma_HOD,
        get_2h=False,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
    )
    pk_II_1h = ccl.halos.halomod_Pk2D(
        ccl_cosmo, hmc, sat_gamma_HOD, get_2h=False, lk_arr=np.log(k_arr), a_arr=a_arr
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

    # pylint: disable=no-member
    print(list(two_point_0.cells.keys()))

    make_plot(ccl_cosmo, nz, pk_GI, pk_II, two_point_0, z)


def make_plot(ccl_cosmo, nz, pk_GI, pk_II, two_point_0, z):
    """Create and show a diagnostic plot."""
    import numpy as np  # pylint: disable-msg=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable-msg=import-outside-topleve

    ells = two_point_0.ells_for_xi
    cells_gg = two_point_0.cells[TracerNames("shear", "shear")]
    cells_gi = two_point_0.cells[TracerNames("shear", "intrinsic_hm")]
    cells_ig = two_point_0.cells[TracerNames("intrinsic_hm", "shear")]
    cells_ii = two_point_0.cells[TracerNames("intrinsic_hm", "intrinsic_hm")]
    cells_total = two_point_0.cells[TracerNames("", "")]
    # pylint: enable=no-member

    # Code that computes effect from IA using that Pk2D object
    t_lens = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz), use_A_ia=False)
    t_ia = ccl.WeakLensingTracer(
        ccl_cosmo,
        has_shear=False,
        use_A_ia=False,
        dndz=(z, nz),
        ia_bias=(z, np.ones_like(z)),
    )
    # pylint: disable=invalid-name
    cl_GI = ccl.angular_cl(ccl_cosmo, t_lens, t_ia, ells, p_of_k_a=pk_GI)
    cl_IG = ccl.angular_cl(ccl_cosmo, t_ia, t_lens, ells, p_of_k_a=pk_GI)
    cl_II = ccl.angular_cl(ccl_cosmo, t_ia, t_ia, ells, p_of_k_a=pk_II)
    # The weak gravitational lensing power spectrum
    cl_GG = ccl.angular_cl(ccl_cosmo, t_lens, t_lens, ells)
    # The observed angular power spectrum is the sum of the two.
    cl_theory = cl_GG + cl_GI + cl_IG + cl_II
    # pylint: enable=invalid-name

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
    plt.title("Halo model IA")
    plt.savefig("halo_model.png", facecolor="white", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_likelihood()
