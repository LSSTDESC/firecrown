"""Cobaya CCL Connector


Provide the class CCLConnector, which is an implementation of a Cobaya Theory.

"""
from __future__ import annotations
from typing import Optional, Dict, List

import numpy as np
import pyccl

from cobaya.theory import Theory

from firecrown.connector.mapping import mapping_builder


class CCLConnector(Theory):
    """
    A class implementing cobaya.theory.Theory.
    """

    input_style: Optional[str] = None

    def initialize(self):
        """Required by Cobaya.

        This is used instead of __init__, to provide default initialization.
        Cobaya does not allow us to override __init__.
        """

        self.map = mapping_builder(input_style=self.input_style)

        self.a_bg = np.linspace(0.1, 1.0, 50)  # pylint: disable-msg=C0103
        self.z_bg = 1.0 / self.a_bg - 1.0  # pylint: disable-msg=C0103
        self.z_Pk = np.arange(0.2, 6.0, 1)  # pylint: disable-msg=C0103
        self.Pk_kmax = 1.0  # pylint: disable-msg=C0103

    def get_param(self, p: str) -> None:
        """Return the current value of the parameter named 'p'.

        This implementation always returns None.
        """
        return None

    def initialize_with_params(self):
        """Required by Cobaya.

        Cobaya calls this after initialize(), so that work can be done after
        that point. This version has nothing to do.
        """

    def initialize_with_provider(self, provider):
        """Required by Cobaya.

        Sets instance's provided to the given provider.
        """

        self.provider = provider

    def get_can_provide_params(self):
        """Required by Cobaya.

        Returns an empty list.
        """
        return []

    def get_can_support_params(self) -> List:
        """Required by Cobaya.

        Return a list containing the names of the mapping's parameter names.
        """
        return self.map.get_params_names()

    def get_allow_agnostic(self):
        """Required by Cobaya.

        Return False.
        """
        return False

    def get_requirements(self) -> Dict:
        """Required by Cobaya.

        Returns a dictionary with keys:
             omk, Pk_grid, comoving_radial_distance, Hubble,
             and with values reflecting the current status of the object.

        """

        pyccl_calculator_requires = {
            "omk": None,
            "Pk_grid": {"k_max": self.Pk_kmax, "z": self.z_Pk},
            "comoving_radial_distance": {"z": self.z_bg},
            "Hubble": {"z": self.z_bg},
        }

        return pyccl_calculator_requires

    def must_provide(self, **requirements):
        """Required by Cobaya.

        This version does nothing.
        """

    def calculate(self, state: Dict, want_derived=True, **params_values) -> None:
        """Calculate the current cosmology, and set state["pyccl"] to the result."""

        self.map.set_params_from_camb(**params_values)

        pyccl_params_values = self.map.asdict()
        # This is the dictionary appropriate for CCL creation

        chi_arr = self.provider.get_comoving_radial_distance(self.z_bg)
        hoh0_arr = self.provider.get_Hubble(self.z_bg) / self.map.get_H0()
        k, z, pk = self.provider.get_Pk_grid()  # pylint: disable-msg=C0103

        # pylint: disable-next=W0201,C0103
        self.a_Pk = self.map.redshift_to_scale_factor(z)
        pk_a = self.map.redshift_to_scale_factor_p_k(pk)

        cosmo = pyccl.CosmologyCalculator(
            **pyccl_params_values,
            background={"a": self.a_bg, "chi": chi_arr, "h_over_h0": hoh0_arr},
            pk_linear={
                "a": self.a_Pk,
                "k": k,
                "delta_matter:delta_matter": pk_a,
            },
            nonlinear_model="halofit",
        )
        state["pyccl"] = cosmo

    def get_pyccl(self) -> pyccl.Cosmology:
        """Return the current cosmology."""
        return self.current_state["pyccl"]
