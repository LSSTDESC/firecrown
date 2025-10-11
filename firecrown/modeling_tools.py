"""Basic Cosmology and cosmological tools definitions.

:mod:`modeling_tools` contains the :class:`ModelingTools` class, which is
built around the :class:`pyccl.Cosmology` class. This is used by likelihoods
that need to access reusable objects, such as perturbation theory or halo model
calculators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

import pyccl.nl_pt

from firecrown.ccl_factory import CCLCalculatorArgs, CCLFactory
from firecrown.models.cluster import ClusterAbundance, ClusterDeltaSigma
from firecrown.updatable import Updatable, UpdatableCollection


class ModelingTools(Updatable):
    """Modeling tools for likelihoods.

    A class that bundles together a :class:`pyccl.Cosmology` object and associated
    objects, such as perturbation theory or halo model calculator workspaces.
    """

    def __init__(
        self,
        *,
        pt_calculator: None | pyccl.nl_pt.EulerianPTCalculator = None,
        hm_calculator: None | pyccl.halos.HMCalculator = None,
        cM_relation: None | str = None,
        pk_modifiers: None | Collection[PowerspectrumModifier] = None,
        cluster_abundance: None | ClusterAbundance = None,
        cluster_deltasigma: None | ClusterDeltaSigma = None,
        ccl_factory: None | CCLFactory = None,
    ):
        super().__init__()
        self.ccl_cosmo: None | pyccl.Cosmology = None
        self.pt_calculator: None | pyccl.nl_pt.EulerianPTCalculator = pt_calculator
        self.hm_calculator: None | pyccl.halos.HMCalculator = hm_calculator
        self.cM_relation: None | str = cM_relation
        pk_modifiers = pk_modifiers if pk_modifiers is not None else []
        self.pk_modifiers: UpdatableCollection = UpdatableCollection(pk_modifiers)
        self.powerspectra: dict[str, pyccl.Pk2D] = {}
        self._prepared: bool = False
        self.cluster_abundance = cluster_abundance
        self.cluster_deltasigma = cluster_deltasigma
        self.ccl_factory = CCLFactory() if ccl_factory is None else ccl_factory

    def add_pk(self, name: str, powerspectrum: pyccl.Pk2D) -> None:
        """Add a :class:`pyccl.Pk2D` to the table of power spectra.

        :param name: the name of the power spectrum
        :param powerspectrum: the power spectrum
        """
        if name in self.powerspectra:
            raise KeyError(f"Power spectrum {name} already exists")

        self.powerspectra[name] = powerspectrum

    def get_pk(self, name: str) -> pyccl.Pk2D:
        """Access a power spectrum from the table of power spectra.

        Either retrieve a pyccl.Pk2D from the table of power spectra, or fall back
        to what the pyccl.Cosmology object can provide.
        :param name: the name of the desired power spectrum
        """
        if self.ccl_cosmo is None:
            raise RuntimeError("Cosmology has not been set")

        if name in self.powerspectra:
            return self.powerspectra[name]

        return self.ccl_cosmo.get_nonlin_power(name)

    def has_pk(self, name: str) -> bool:
        """Check if a power spectrum with name `name` is available."""
        # There should probably a pyccl.Cosmology method to check if a specific
        # power spectrum exists
        try:
            self.get_pk(name)
        except KeyError:
            return False
        return True

    def prepare(self, *, calculator_args: None | CCLCalculatorArgs = None) -> None:
        """Prepare the Cosmology for use in likelihoods.

        This method will prepare the ModelingTools for use in likelihoods. This
        includes building the perturbation theory and halo model calculators
        if they are needed.

        :param calculator_args: the CCL calculator arguments
        """
        if not self.is_updated():
            raise RuntimeError("ModelingTools has not been updated.")

        if self._prepared:
            raise RuntimeError("ModelingTools has already been prepared")

        if self.ccl_cosmo is not None:
            raise RuntimeError("Cosmology has already been set")

        self.ccl_cosmo = self.ccl_factory.create(calculator_args)

        if self.pt_calculator is not None:
            self.pt_calculator.update_ingredients(self.ccl_cosmo)

        for pkm in self.pk_modifiers:
            self.add_pk(name=pkm.name, powerspectrum=pkm.compute_p_of_k_z(tools=self))

        if self.cluster_abundance is not None:
            self.cluster_abundance.update_ingredients(self.ccl_cosmo)

        if self.cluster_deltasigma is not None:
            self.cluster_deltasigma.update_ingredients(self.ccl_cosmo)
        self._prepared = True

    def _reset(self) -> None:
        """Resets all CCL objects in ModelingTools.

        This method is called by the Updatable base class when the object is
        destroyed. It also resets the power spectra, the cosmology and the
        _prepared state variable.
        """
        self.ccl_cosmo = None
        # Also reset the power spectra
        # TODO: is that always needed?
        self.powerspectra = {}
        self._prepared = False

    def get_ccl_cosmology(self) -> pyccl.Cosmology:
        """Return the CCL cosmology object."""
        if self.ccl_cosmo is None:
            raise RuntimeError("Cosmology has not been set")
        return self.ccl_cosmo

    def get_pt_calculator(self) -> pyccl.nl_pt.EulerianPTCalculator:
        """Return the perturbation theory calculator object."""
        if self.pt_calculator is None:
            raise RuntimeError("A PT calculator has not been set")
        return self.pt_calculator

    def get_hm_calculator(self) -> pyccl.halos.HMCalculator:
        """Return the halo model calculator object."""
        if self.hm_calculator is None:
            raise RuntimeError("A Halo Model calculator has not been set")
        return self.hm_calculator

    def get_cM_relation(self) -> str:
        """Return the concentration-mass relation."""
        if self.cM_relation is None:
            raise RuntimeError("A concentration-mass relation has not been set")
        return self.cM_relation


class PowerspectrumModifier(Updatable, ABC):
    """Abstract base class for power spectrum modifiers."""

    name: str = "base:base"

    @abstractmethod
    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        """Compute the 3D power spectrum P(k, z)."""
