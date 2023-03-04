"""Basic module for Cosmology and cosmological tools definitions.

This module contains the ModelingTools class, which is built around the
pyccl.Cosmology class. This is used by likelihoods that need to access
reusable objects, such as perturbation theory or halo model calculators.
"""

from typing import Dict, Optional, final
import pyccl.nl_pt


class ModelingTools:
    """A class that bundles together a pyccl.Cosmology object and associated
    objects, such as perturbation theory or halo model calculator workspaces."""

    def __init__(
        self,
        *,
        pt_calculator: Optional[pyccl.nl_pt.PTCalculator] = None,
        hm_definition: Optional[pyccl.halos.MassDef] = None,
        hm_function: Optional[str] = None,
        bias_function: Optional[str] = None,
        cM_relation: Optional[str] = None,
    ):
        self.ccl_cosmo: Optional[pyccl.Cosmology] = None
        self.pt_calculator: Optional[pyccl.nl_pt.PTCalculator] = pt_calculator
        # Developed this way because the halo model calculator requires cosmology as input.
        # This way we are passing the ingredients to create the HM calc.
        # Disadvantage is that other HMC parameters (e.g. mass sampling) are not tweakable.
        self.hm_definition: Optional[pyccl.halos.MassDef] = hm_definition
        self.hm_function: Optional[str] = hm_function
        self.bias_function: Optional[str] = bias_function
        self.cM_relation: Optional[str] = cM_relation
        self.powerspectra: Dict[str, pyccl.Pk2D] = {}

    def add_pk(self, name: str, powerspectrum: pyccl.Pk2D):
        """Add a pyccl.Pk2D to the table of power spectra."""

        if name in self.powerspectra:
            raise KeyError(f"Power spectrum {name} already exists")

        self.powerspectra[name] = powerspectrum

    def get_pk(self, name: str) -> pyccl.Pk2D:
        """Retrive a pyccl.Pk2D from the table of power spectra, or fall back
        to what the pyccl.Cosmology object can provide."""

        if self.ccl_cosmo is None:
            raise RuntimeError("Cosmology has not been set")

        if name in self.powerspectra:
            return self.powerspectra[name]

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

    def prepare(self, ccl_cosmo: pyccl.Cosmology) -> None:
        """Prepare the Cosmology for use in likelihoods.

        This method will prepare the ModelingTools for use in likelihoods. This
        includes building the perturbation theory and halo model calculators
        if they are needed.

        :param ccl_cosmo: the current CCL cosmology object

        """

        if self.ccl_cosmo is not None:
            raise RuntimeError("Cosmology has already been set")
        self.ccl_cosmo = ccl_cosmo

        if self.pt_calculator is not None:
            pk_lin_z0 = pyccl.linear_matter_power(ccl_cosmo, self.pt_calculator.ks, 1.0)
            self.pt_calculator.update_pk(pk_lin_z0)

    @final
    def reset(self) -> None:
        """Resets all CCL objects in ModelingTools."""

        self.ccl_cosmo = None

    def get_ccl_cosmology(self) -> pyccl.Cosmology:
        """Return the CCL cosmology object."""

        if self.ccl_cosmo is None:
            raise RuntimeError("Cosmology has not been set")
        return self.ccl_cosmo

    def get_pt_calculator(self) -> pyccl.nl_pt.PTCalculator:
        """Return the perturbation theory calculator object."""

        if self.pt_calculator is None:
            raise RuntimeError("A PT calculator has not been set")
        return self.pt_calculator

    def get_hm_calculator(self) -> pyccl.halos.HMCalculator:
        """Return the halo model calculator object."""

        if self.hm_definition is None:
            raise RuntimeError("A halo mass definition has not been set")
        if self.hm_function is None:
            raise RuntimeError("A halo mass function has not been set")
        if self.bias_function is None:
            raise RuntimeError("A halo bias function has not been set")
        cosmo = self.get_ccl_cosmology()
        nM = pyccl.halos.mass_function_from_name(self.hm_function)(cosmo, mass_def=self.hm_definition)
        bM = pyccl.halos.halo_bias_from_name(self.bias_function)(cosmo, mass_def=self.hm_definition)

        return pyccl.halos.HMCalculator(cosmo, nM, bM, self.hm_definition)

    def get_cM_relation(self) -> pyccl.halos.Concentration:
        """Return the concentration-mass relation."""

        if self.cM_relation is None:
            raise RuntimeError("A concentration-mass relation has not been set")
        if self.hm_definition is None:
            raise RuntimeError("A halo mass definition has not been set")

        return pyccl.halos.concentration_from_name(self.cM_relation)(self.hm_definition)
