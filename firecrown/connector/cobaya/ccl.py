"""Cobaya CCL Connector.

Provide the class CCLConnector, which is an implementation of a Cobaya Theory.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cobaya.theory import Theory

from firecrown.connector.mapping import mapping_builder, MappingCAMB
from firecrown.ccl_factory import CCLCalculatorArgs


class CCLConnector(Theory):
    """A class implementing cobaya.theory.Theory."""

    input_style: str | None = None

    def initialize(self) -> None:
        """Initialize a CCLConnector object.

        Required by Cobaya.

        This is used instead of __init__, to provide default initialization.
        Cobaya does not allow us to override __init__.
        """
        assert self.input_style
        # We have to do some extra type-fiddling here because mapping_builder
        # has a declared return type of the base class.
        new_mapping = mapping_builder(input_style=self.input_style)
        assert isinstance(new_mapping, MappingCAMB)
        self.map = new_mapping

        self.a_bg = np.linspace(0.1, 1.0, 50)
        self.z_bg = 1.0 / self.a_bg - 1.0
        self.z_Pk = np.arange(0.0, 6.0, 1)
        self.Pk_kmax = 1.0

    def initialize_with_params(self) -> None:
        """Complete the initialization of a CCLConnector object.

        Required by Cobaya.

        Cobaya calls this after initialize(), so that work can be done after
        that point. This version has nothing to do.
        """

    def initialize_with_provider(self, provider) -> None:
        """Set the object's provider.

        Required by Cobaya.

        :param provider: A Cobaya provider.
        """
        self.provider = provider

    def get_can_provide_params(self) -> list[str]:
        """Return the list of params provided.

        Required by Cobaya.

        Returns an empty list.
        """
        return []

    def get_can_support_params(self) -> list[str]:
        """Return a list containing the names of the mapping's parameter names.

        Required by Cobaya.
        :return: The list of parameter names.
        """
        return self.map.get_params_names()

    def get_allow_agnostic(self) -> bool:
        """Is it allowed to pass all unassigned input parameters to this component.

        Required by Cobaya.

        This implementation always returns False.
        :return: False
        """
        return False

    def get_requirements(
        self,
    ) -> dict[str, None | dict[str, npt.NDArray[np.float64]] | dict[str, object]]:
        """Returns a dictionary.

        The dictionary contains the following keys:
             omk, Pk_grid, comoving_radial_distance, Hubble,
             and with values reflecting the current status of the object.
        Required by Cobaya.
        :return: a dictionary
        """
        pyccl_calculator_requires = {
            "omk": None,
            "Pk_grid": {"k_max": self.Pk_kmax, "z": self.z_Pk},
            "comoving_radial_distance": {"z": self.z_bg},
            "Hubble": {"z": self.z_bg},
        }

        return pyccl_calculator_requires

    def must_provide(self, **requirements) -> None:
        """Required by Cobaya.

        This version does nothing.
        """

    def calculate(self, state: dict, want_derived=True, **params_values) -> None:
        """Calculate the current cosmology, and set state["pyccl"] to the result.

        :param state: The state dictionary to update.
        :param want_derived: Whether to calculate derived parameters or not.
        :param params_values: The values of the parameters to use.
        """
        self.map.set_params_from_camb(**params_values)
        pyccl_params_values = self.map.asdict()

        # This is the dictionary appropriate for CCL creation
        chi_arr = self.provider.get_comoving_radial_distance(self.z_bg)
        hoh0_arr = self.provider.get_Hubble(self.z_bg) / self.map.get_H0()
        k, z, pk = self.provider.get_Pk_grid()

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
        state["pyccl_args"] = pyccl_args
        state["pyccl_params"] = pyccl_params_values

    def get_pyccl_args(self) -> CCLCalculatorArgs:
        """Return the current CCL arguments.

        :return: The current CCL arguments
        """
        return self.current_state["pyccl_args"]

    def get_pyccl_params(self) -> dict[str, float]:
        """Return the current cosmological parameters.

        :return: The current cosmological parameters.
        """
        return self.current_state["pyccl_params"]
