"""Example of a Firecrown likelihood using the DES Y1 cosmic shear data.

This example also includes a modified matter power spectrum.
"""

import os

import pyccl

from firecrown.likelihood.factories import load_sacc_data
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood import TwoPoint, ConstGaussian, Likelihood
from firecrown.updatable import ParamsMap, register_new_updatable_parameter
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.modeling_tools import CCLFactory
from firecrown.updatable import get_default_params_map
from firecrown.metadata_types import TracerNames

SACC_FILE = os.path.expanduser(
    os.path.expandvars("${FIRECROWN_DIR}/examples/des_y1_3x2pt/sacc_data.hdf5")
)


class vanDaalen19Baryonfication(PowerspectrumModifier):
    """A PowerspectrumModifier class.

    This class implements the van Daalen et al. 2019 baryon model.
    """

    name: str = "delta_matter_baryons:delta_matter_baryons"

    def __init__(self, pk_to_modify: str = "delta_matter:delta_matter"):
        super().__init__()
        self.pk_to_modify = pk_to_modify
        self.vD19 = pyccl.baryons.BaryonsvanDaalen19()
        self.f_bar = register_new_updatable_parameter(default_value=0.5)

    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        """Compute the 3D power spectrum P(k, z)."""
        self.vD19.update_parameters(fbar=self.f_bar)
        return self.vD19.include_baryonic_effects(
            cosmo=tools.get_ccl_cosmology(), pk=tools.get_pk(self.pk_to_modify)
        )


def build_likelihood(_) -> tuple[Likelihood, ModelingTools]:
    """Build the likelihood for the DES Y1 cosmic shear data TATT."""
    # Load sacc file
    sacc_data = load_sacc_data(SACC_FILE)

    n_source = 1
    stats = define_stats(n_source)

    # Define the power spectrum modification and add it to the ModelingTools
    pk_modifier = vanDaalen19Baryonfication(pk_to_modify="delta_matter:delta_matter")
    modeling_tools = ModelingTools(
        pk_modifiers=[pk_modifier], ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )

    # Create the likelihood from the statistics
    likelihood = ConstGaussian(statistics=list(stats.values()))

    # Read the two-point data from the sacc file
    likelihood.read(sacc_data)

    # To allow this likelihood to be used in cobaya or cosmosis, define a
    # an object called "likelihood" must be defined
    print(
        "Using parameters:",
        list(likelihood.required_parameters().get_params_names()),
        list(modeling_tools.required_parameters().get_params_names()),
    )

    return likelihood, modeling_tools


def define_stats(n_source):
    """Define the TwoPoint objects to be returned by this factory function."""
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
    sacc_data = load_sacc_data(SACC_FILE)

    src0_tracer = sacc_data.get_tracer("src0")
    z, nz = src0_tracer.z, src0_tracer.nz

    f_bar = 0.5
    # Set the parameters for our systematics
    systematics_params = {
        "f_bar": f_bar,
        "src0_delta_z": 0.000,
        "src1_delta_z": 0.003,
        "src2_delta_z": -0.001,
        "src3_delta_z": 0.002,
    }

    # Prepare the cosmology object
    params = ParamsMap(get_default_params_map(tools).params | systematics_params)

    tools.update(params)
    tools.prepare()

    ccl_cosmo = tools.get_ccl_cosmology()

    # Calculate the baryonic effects directly with CCL
    vD19 = pyccl.BaryonsvanDaalen19(fbar=f_bar)
    pk_baryons = vD19.include_baryonic_effects(
        cosmo=ccl_cosmo, pk=ccl_cosmo.get_nonlin_power()
    )

    # Apply the systematics parameters
    likelihood.update(params)

    # Compute the log-likelihood, using the pyccl.Cosmology object as the input
    log_like = likelihood.compute_loglike(tools)

    print(f"Log-like = {log_like:.1f}")

    # Plot the predicted and measured statistic
    assert isinstance(likelihood, ConstGaussian)
    two_point_0 = likelihood.statistics[0].statistic
    assert isinstance(two_point_0, TwoPoint)
    assert likelihood.cov is not None

    # Predict CCL Cl
    wl_tracer = pyccl.WeakLensingTracer(ccl_cosmo, dndz=(z, nz))
    ell = two_point_0.ells_for_xi
    cl_dm = pyccl.angular_cl(
        cosmo=ccl_cosmo,
        tracer1=wl_tracer,
        tracer2=wl_tracer,
        ell=ell,
    )
    cl_baryons = pyccl.angular_cl(
        cosmo=ccl_cosmo,
        tracer1=wl_tracer,
        tracer2=wl_tracer,
        ell=ell,
        p_of_k_a=pk_baryons,
    )

    # pylint: disable=no-member
    print(list(two_point_0.cells.keys()))

    make_plot(ell, cl_dm, cl_baryons, two_point_0)


def make_plot(ell, cl_dm, cl_baryons, two_point_0):
    """Create and show a diagnostic plot."""
    import matplotlib.pyplot as plt  # pylint: disable-msg=import-outside-toplevel

    cl_firecrown = two_point_0.cells[TracerNames("shear", "shear")]

    plt.plot(ell, cl_firecrown / cl_dm, label="firecrown w/ baryons")
    plt.plot(ell, cl_baryons / cl_dm, ls="--", label="CCL w/ baryons")

    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell^\mathrm{baryons} / C_\ell^\mathrm{DM-only}$")
    plt.legend()
    plt.savefig("plots/vD19_baryons.png", facecolor="white", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_likelihood()
