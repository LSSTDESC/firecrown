import math
import numpy as np
import pyccl as ccl

from firecrown.convert import firecrown_convert_builder

from pprint import pprint

from cobaya.theory import Theory


class CCLConnector(Theory):
    """
    A class implementing cobaya.theory.Theory ...

    ...

    Attributes
    ----------
    ... : str
        ...

    Methods
    -------
    ...(...)
        ....
    """

    input_style: str = None

    def initialize(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """

        self.fc_params = firecrown_convert_builder(input_style=self.input_style)

        self.a_bg = np.linspace(0.1, 1.0, 50)
        self.z_bg = 1.0 / self.a_bg - 1.0
        self.z_Pk = np.arange(0.2, 6.0, 1)
        self.Pk_kmax = 1.0
        pass

    def get_param(self, p):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def initialize_with_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def initialize_with_provider(self, provider):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        self.provider = provider

    def get_can_provide_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return []

    def get_can_support_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self.fc_params.get_names()

    def get_allow_agnostic(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return False

    def get_requirements(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        ccl_calculator_requires = {
            "Pk_grid": {"k_max": self.Pk_kmax, "z": self.z_Pk},
            "comoving_radial_distance": {"z": self.z_bg},
            "Hubble": {"z": self.z_bg},
        }
        return ccl_calculator_requires

    def must_provide(self, **requirements):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def calculate(self, state, want_derived=True, **params_values):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """

        self.fc_params.set_params(**params_values)

        ccl_params_values = self.fc_params.get_params()
        # This is the dictionary appropriate for CCL creation

        chi_arr = self.provider.get_comoving_radial_distance(self.z_bg)
        hoh0_arr = self.provider.get_Hubble(self.z_bg) / self.fc_params.get_H0()
        k, z, pk = self.provider.get_Pk_grid()

        self.a_Pk = np.sort(1.0 / (1.0 + z))
        cosmo = ccl.CosmologyCalculator(
            **ccl_params_values,
            background={"a": self.a_bg, "chi": chi_arr, "h_over_h0": hoh0_arr},
            pk_linear={
                "a": self.a_Pk,
                "k": k,
                "delta_matter:delta_matter": pk[:, ::-1],
            },
            nonlinear_model="halofit"
        )
        state["ccl"] = cosmo

    def get_ccl(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self.current_state["ccl"]
