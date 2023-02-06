"""Basic module for Cosmology and cosmological tools definitions.

This module contains the Cosmology class, which is built around the
pyccl.Cosmology class. It also contains the CosmologyContainer class,
which is a container for a Cosmology instance and the parameters used
to create it. This is used by the likelihoods that need to access the
cosmology to compute the likelihood.
"""

from typing import Dict, Optional

import pyccl
import pyccl.nl_pt


class Cosmology:
    """A class that bundles together a pyccl.Cosmology object and associated
    objects, such as perturbation theory or halo model calculator workspaces."""

    def __init__(self, cosmo: pyccl.Cosmology):
        self.ccl_cosmo = cosmo
        self.pt_calculator: Optional[pyccl.nl_pt.PTCalculator] = None
        self.hm_calculator: Optional[pyccl.halomodel.HMCalculator] = None
        self.pk: Dict[str, pyccl.Pk2D] = {}

    def add_pk(self, name: str, pk: pyccl.Pk2D):
        """Add a pyccl.Pk2D to the table of power spectra."""
        self.pk[name] = pk

    def get_pk(self, name: str) -> pyccl.Pk2D:
        """Retrive a pyccl.Pk2D from the table of power spectra, or fall back
        to what the pyccl.Cosmology object can provide."""
        if name in self.pk:
            return self.pk[name]
        return self.ccl_cosmo.get_nonlin_power(name)

    def has_pk(self, name: str) -> bool:
        """Check if a power spectrum with name `name' is available."""
        # There should probably a pyccl.Cosmology method to check if a specific
        # power spectrum exists
        try:
            self.get_pk(name)
        except KeyError:
            return False
        return True

    def prepare(self, **kwargs):
        """Prepare the Cosmology for use in likelihoods.

        This method will prepare the Cosmology for use in likelihoods. This
        includes building the perturbation theory and halo model calculators
        if they are needed.

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to the perturbation theory and halo model
            calculators.
        """
        if self.pt_calculator is None:
            self.pt_calculator = pyccl.nl_pt.PTCalculator(self.ccl_cosmo, **kwargs)
        if self.hm_calculator is None:
            self.hm_calculator = pyccl.halomodel.HMCalculator(self.ccl_cosmo, **kwargs)
