"""Example of a Firecrown likelihood using the DES Y1 cosmic shear data TATT."""

import os
from typing import Tuple

import pyccl
import sacc
import pyccl as ccl
import pyccl.nl_pt

import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap, create
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.likelihood.likelihood import Likelihood


SACCFILE = os.path.expanduser(
    os.path.expandvars(
        "${FIRECROWN_DIR}/examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data.fits"
    )
)


class vanDaalen19Baryonfication(PowerspectrumModifier):
    name: str = "delta_matter_baryons:delta_matter_baryons"

    def __init__(self, pk_to_modify: str = "delta_matter:delta_matter"):
        super().__init__()
        self.pk_to_modify = pk_to_modify
        self.vD19 = pyccl.baryons.BaryonsvanDaalen19()
        self.f_bar = create()

    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        self.vD19.update_parameters(fbar=self.f_bar)
        return self.vD19.include_baryonic_effects(
            cosmo=tools.get_ccl_cosmology(),
            pk=tools.get_pk(self.pk_to_modify)
        )


def build_likelihood(_) -> Tuple[Likelihood, ModelingTools]:
    """Build the likelihood for the DES Y1 cosmic shear data TATT."""
    # Load sacc file
    sacc_data = sacc.Sacc.load_fits(SACCFILE)

    n_source = 1
    stats = define_stats(n_source)

    # Create the likelihood from the statistics
    pk_modifier = vanDaalen19Baryonfication(pk_to_modify="delta_matter:delta_matter")

    modeling_tools = ModelingTools(pk_modifiers=[pk_modifier])
    likelihood = ConstGaussian(statistics=list(stats.values()))

    # Read the two-point data from the sacc file
    likelihood.read(sacc_data)

    # To allow this likelihood to be used in cobaya or cosmosis, define a
    # an object called "likelihood" must be defined
    print(
        "Using parameters:",
        list(likelihood.required_parameters().get_params_names()),
        list(modeling_tools.required_parameters().get_params_names())
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
    # Specify that the matter power spectrum with baryons should be used
    baryon_systematic = wl.SelectField(field="delta_matter_baryons")
    for i in range(n_source):
        # Define the photo-z shift systematic.
        pzshift = wl.PhotoZShift(sacc_tracer=f"src{i}")

        # Create the weak lensing source, specifying the name of the tracer in the
        # sacc file and a list of systematics
        result[f"src{i}"] = wl.WeakLensing(
            sacc_tracer=f"src{i}", systematics=[pzshift, baryon_systematic]
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

    # Define a ccl.Cosmology object using default parameters
    ccl_cosmo = ccl.CosmologyVanillaLCDM()
    ccl_cosmo.compute_nonlin_power()

    f_bar = 0.5

    vD19 = pyccl.BaryonsvanDaalen19(fbar=f_bar)
    pk_baryons = vD19.include_baryonic_effects(
        cosmo=ccl_cosmo,
        pk=ccl_cosmo.get_nonlin_power()
    )

    # Set the parameters for our systematics
    systematics_params = ParamsMap(
        {
            "f_bar": f_bar,
            "src0_delta_z": 0.000,
            "src1_delta_z": 0.003,
            "src2_delta_z": -0.001,
            "src3_delta_z": 0.002,
        }
    )

    # Apply the systematics parameters
    likelihood.update(systematics_params)

    # Prepare the cosmology object
    tools.update(systematics_params)
    tools.prepare(ccl_cosmo)

    # Compute the log-likelihood, using the ccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(tools)

    print(f"Log-like = {log_like:.1f}")

    # Plot the predicted and measured statistic
    two_point_0 = likelihood.statistics[0].statistic
    assert isinstance(two_point_0, TwoPoint)

    assert isinstance(likelihood, ConstGaussian)
    assert likelihood.cov is not None

    # Predict CCL Cl
    wl_tracer = ccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    ell = two_point_0.ells
    cl_dm = ccl.angular_cl(
        cosmo=ccl_cosmo,
        tracer1=wl_tracer, tracer2=wl_tracer,
        ell=ell,
    )
    cl_baryons = ccl.angular_cl(
        cosmo=ccl_cosmo,
        tracer1=wl_tracer, tracer2=wl_tracer,
        ell=ell,
        p_of_k_a=pk_baryons
    )

    # pylint: disable=no-member
    print(list(two_point_0.cells.keys()))

    make_plot(ell, cl_dm, cl_baryons, two_point_0)


def make_plot(ell, cl_dm, cl_baryons, two_point_0):
    """Create and show a diagnostic plot."""
    import matplotlib.pyplot as plt  # pylint: disable-msg=import-outside-toplevel

    cl_firecrown = two_point_0.cells[("shear", "shear")]

    plt.plot(ell, cl_firecrown/cl_dm, label="firecrown w/ baryons")
    plt.plot(ell, cl_baryons/cl_dm, ls="--", label="CCL w/ baryons")

    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^\mathrm{baryons} / C_\ell^\mathrm{DM-only}$")
    plt.legend()
    plt.savefig("plots/vD19_baryons.png", facecolor="white", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_likelihood()
