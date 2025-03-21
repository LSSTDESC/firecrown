"""Cobaya Likelihood Connector.

Module for providing a likelihood for use in Cobaya.

This module provides the class :class:`LikelihoodConnector`, which is an implementation
of a Cobaya likelihood.
"""

import numpy as np
import numpy.typing as npt

from cobaya.likelihood import Likelihood
import pyccl
from pyccl.cosmology import Pk2D

# See comment in compute_pyccl_args_options
# from pyccl.pyutils import loglin_spacing

from firecrown.connector.mapping import mapping_builder, MappingCAMB, Mapping
from firecrown.ccl_factory import CCLCalculatorArgs
from firecrown.likelihood.likelihood import load_likelihood, NamedParameters
from firecrown.likelihood.likelihood import Likelihood as FirecrownLikelihood
from firecrown.parameters import ParamsMap
from firecrown.ccl_factory import PoweSpecAmplitudeParameter, CCLCreationMode
from firecrown.updatable import get_default_params_map


def compute_pyccl_args_options(
    ccl_cosmo: pyccl.Cosmology,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float
]:
    """Creates a dictionary of pyccl arguments.

    This method uses the CCLFactory to create a pyccl object and returns the
    dictionary of precision options for the pyccl object.
    """
    # Here we follow the pyccl convention.
    spl = ccl_cosmo.cosmo.spline_params
    # Ideally, we would construct the background scale factor array in the same way as
    # CCL. This would ensure consistency with how pyccl operates in pure mode. However,
    # the C function ccl_cosmology_distances_from_input() updates spl.A_SPLINE_MIN based
    # on the minimum scale factor provided.
    #
    # If we build the array using the approach below (following CCL):
    #
    # a_bg = loglin_spacing(
    #    spl.A_SPLINE_MINLOG,
    #    spl.A_SPLINE_MIN,
    #    spl.A_SPLINE_MAX,
    #    spl.A_SPLINE_NLOG,
    #    spl.A_SPLINE_NA,
    # )
    #
    # then a[0] will be equal to spl.A_SPLINE_MINLOG. When
    # ccl_cosmology_distances_from_input() updates spl.A_SPLINE_MIN, it will set
    # spl.A_SPLINE_MIN = spl.A_SPLINE_MINLOG. This causes issues in later computations
    # (it creates an array with equal elements), particularly in the construction of the
    # growth function.
    #
    # To avoid this, we construct the array starting from spl.A_SPLINE_MIN instead:
    a_bg = np.linspace(spl.A_SPLINE_MIN, spl.A_SPLINE_MAX, spl.A_SPLINE_NA)

    z_bg = (1.0 / a_bg - 1.0).astype(np.float64)

    # We inspect the linear power spectrum from pyccl to get the maximum k and the
    # redshift grid.
    psp: Pk2D = ccl_cosmo.get_linear_power()
    a_arr, lk_arr, _ = psp.get_spline_arrays()
    Pk_kmax = np.exp(np.max(lk_arr))
    z_array = np.flip(1.0 / a_arr - 1.0)

    return a_bg, z_bg, z_array, Pk_kmax


class LikelihoodConnector(Likelihood):
    """A class implementing cobaya.likelihood.Likelihood."""

    input_style: str | None = None
    likelihood: FirecrownLikelihood
    firecrownIni: str
    derived_parameters: list[str] = []
    build_parameters: NamedParameters

    def initialize(self):
        """Initialize the likelihood object by loading its Firecrown configuration."""
        if not hasattr(self, "build_parameters"):
            build_parameters = NamedParameters()
        else:
            if isinstance(self.build_parameters, dict):
                build_parameters = NamedParameters(self.build_parameters)
            else:
                if not isinstance(self.build_parameters, NamedParameters):
                    raise TypeError(
                        "build_parameters must be a NamedParameters or dict"
                    )
                build_parameters = self.build_parameters

        self.likelihood, self.tools = load_likelihood(
            self.firecrownIni, build_parameters
        )

        self.external_obs: (
            dict[str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]]
            | None
        ) = None
        self.map: Mapping | None = None
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            # We need to request external Boltzmann code for observables and
            # cosmological parameters.
            assert self.input_style
            # We have to do some extra type-fiddling here because mapping_builder has a
            # declared return type of the base class.
            new_mapping = mapping_builder(input_style=self.input_style)
            assert isinstance(new_mapping, MappingCAMB)
            self.map = new_mapping

            # External observables are necessary in the default mode, so we need to
            # extract from CCL its precison parameters.
            params = get_default_params_map(self.tools)
            self.tools.update(params)
            self.tools.prepare()
            ccl_cosmo = self.tools.ccl_factory.get()
            self.a_bg, self.z_bg, z_array, Pk_kmax = compute_pyccl_args_options(
                ccl_cosmo
            )

            # We need to request external Boltzmann code for power spectra, if we want
            # both linear and nonlinear we need to pass (False, True) to the "nonlinear"
            # option.
            self.external_obs = {
                "omk": None,
                "Pk_grid": {
                    "k_max": Pk_kmax,
                    "z": z_array,
                    "nonlinear": (False, True),
                },
                "comoving_radial_distance": {"z": self.z_bg},
                "Hubble": {"z": self.z_bg},
            }
            self.tools.reset()

    def initialize_with_params(self) -> None:
        """Complete the initialization of a LikelihoodConnector object.

        Required by Cobaya.

        This version has nothing to do.
        """

    def initialize_with_provider(self, provider) -> None:
        """Set the obejct's provider.

        Required by Cobaya.

        :param provider: A Cobaya provider.
        """
        self.provider = provider

    def get_can_provide_params(self) -> list[str]:
        """Return the list of params provided.

        Required by Cobaya.

        Returns an empty list.
        """
        return self.derived_parameters

    def get_can_support_params(self) -> list[str]:
        """Return a list containing the names of the mapping's parameter names.

        Required by Cobaya.
        :return: The list of parameter names.
        """
        if self.map is None:
            return []
        return self.map.get_params_names(self.tools.ccl_factory.amplitude_parameter)

    def get_allow_agnostic(self) -> bool:
        """Is it allowed to pass all unassigned input parameters to this component.

        Required by Cobaya.

        Return False.
        """
        return False

    def get_requirements(
        self,
    ) -> dict[str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]]:
        """Returns a dictionary.

        Returns a dictionary with keys corresponding the contained likelihood's
        required parameter the values give the required options.

        Required by Cobaya.
        :return: a dictionary
        """
        required_params = (
            self.likelihood.required_parameters() + self.tools.required_parameters()
        )

        likelihood_requires: dict[
            str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]
        ] = {}
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            # We need to request external Boltzmann code for observables
            # and cosmological parameters.
            assert self.external_obs is not None
            likelihood_requires.update(self.external_obs)
            # Cosmological parameters differ from Cobaya's boltzmann interface, so we
            # need to remove them when using Calculator mode.
            required_params -= self.tools.ccl_factory.required_parameters()
            if (
                self.tools.ccl_factory.amplitude_parameter
                == PoweSpecAmplitudeParameter.SIGMA8
            ):
                # Tell Cobaya we want it to calculate sigma8, and that we do not
                # provide the way to do that; something else must know how to do
                # so, or an error will result.
                likelihood_requires["sigma8"] = None

        for param_name in required_params.get_params_names():
            likelihood_requires[param_name] = None

        return likelihood_requires

    def must_provide(self, **requirements) -> None:
        """Required by Cobaya.

        This version does nothing.
        """

    def calculate_args(
        self, params_values
    ) -> tuple[CCLCalculatorArgs, dict[str, float | list[float]]]:
        """Calculate the curr   ent cosmology, and set state["pyccl"] to the result.

        :param state: The state dictionary to update.
        :param params_values: The values of the parameters to use.
        """
        assert self.map is not None
        assert isinstance(self.map, MappingCAMB)
        self.map.set_params_from_camb(**params_values)
        pyccl_params_values = self.map.asdict()

        # This is the dictionary appropriate for CCL creation
        chi_arr = self.provider.get_comoving_radial_distance(self.z_bg)
        hoh0_arr = self.provider.get_Hubble(self.z_bg) / self.map.get_H0()
        k, z, pk = self.provider.get_Pk_grid(nonlinear=False)
        # Note: we havae to define self.a_Pk here because Cobaya does not allow
        # us to override the __init__ method.
        #
        # pylint: disable-next=attribute-defined-outside-init
        self.a_Pk = self.map.redshift_to_scale_factor(z)
        pk_a = self.map.redshift_to_scale_factor_p_k(pk)

        pyccl_args: CCLCalculatorArgs = {
            "background": {"a": self.a_bg, "chi": chi_arr, "h_over_h0": hoh0_arr},
            "pk_linear": {"a": self.a_Pk, "k": k, "delta_matter:delta_matter": pk_a},
        }
        return pyccl_args, pyccl_params_values

    def logp(self, **params_values) -> float:
        """Return the log of the calculated likelihood.

        Required by Cobaya.
        :params values: The values of the parameters to use.
        """
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            pyccl_args, pyccl_params = self.calculate_args(params_values)
            derived = params_values.pop("_derived", {})
            params = ParamsMap(params_values | pyccl_params)
            self.likelihood.update(params)
            self.tools.update(params)
            self.tools.prepare(calculator_args=pyccl_args)
        else:
            derived = params_values.pop("_derived", {})
            params = ParamsMap(params_values)
            self.likelihood.update(params)
            self.tools.update(params)
            self.tools.prepare()

        loglike = self.likelihood.compute_loglike(self.tools)

        derived_params_collection = self.likelihood.get_derived_parameters()
        assert derived_params_collection is not None
        for section, name, val in derived_params_collection:
            derived[f"{section}__{name}"] = val

        self.likelihood.reset()
        self.tools.reset()

        return loglike
