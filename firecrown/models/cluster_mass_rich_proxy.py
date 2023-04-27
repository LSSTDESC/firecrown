"""Cluster Mass Richness proxy module

Define the Cluster Mass Richness proxy module and its arguments.
"""
from __future__ import annotations
from typing import List, Tuple, final
import itertools

import numpy as np
from scipy import special
from .. import sacc_support
from ..sacc_support import sacc

from ..parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from .cluster_mass import ClusterMass, ClusterMassArgument
from .. import parameters


class ClusterMassRichPointArgument(ClusterMassArgument):
    """Argument for the Cluster Mass Richness proxy."""

    def __init__(
        self,
        richness: ClusterMassRich,
        logMl: float,
        logMu: float,
        logM_obs: float,
    ):
        super().__init__(logMl, logMu)
        self.richness: ClusterMassRich = richness
        self.logM_obs: float = logM_obs

    @property
    def dim(self) -> int:
        """Return the dimension of the argument."""
        return 0

    def get_logM_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster mass argument."""
        return (self.logMl, self.logMu)

    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster mass proxy argument."""
        return []

    def p(self, logM: float, z: float, *_) -> float:
        """Return the probability of the point argument."""

        lnM_obs = self.logM_obs * np.log(10.0)

        lnM_mu, sigma = self.richness.cluster_mass_lnM_obs_mu_sigma(logM, z)
        x = lnM_obs - lnM_mu
        chisq = np.dot(x, x) / (2.0 * sigma**2)
        likelihood = np.exp(-chisq) / (np.sqrt(2.0 * np.pi * sigma**2))
        return likelihood * np.log(10.0)


class ClusterMassRichBinArgument(ClusterMassArgument):
    """Argument for the Cluster Mass Richness proxy."""

    def __init__(
        self,
        richness: ClusterMassRich,
        logMl: float,
        logMu: float,
        logM_obs_lower: float,
        logM_obs_upper: float,
    ):
        super().__init__(logMl, logMu)
        self.richness: ClusterMassRich = richness
        self.logM_obs_lower: float = logM_obs_lower
        self.logM_obs_upper: float = logM_obs_upper

    @property
    def dim(self) -> int:
        """Return the dimension of the argument."""
        return 0

    def get_logM_bounds(self) -> Tuple[float, float]:
        """Return the bounds of the cluster mass argument."""
        return (self.logMl, self.logMu)

    def get_proxy_bounds(self) -> List[Tuple[float, float]]:
        """Return the bounds of the cluster mass proxy argument."""
        return []

    def p(self, logM: float, z: float, *_) -> float:
        """Return the probability of the binned argument."""

        lnM_obs_mu, sigma = self.richness.cluster_mass_lnM_obs_mu_sigma(logM, z)
        x_min = (lnM_obs_mu - self.logM_obs_lower * np.log(10.0)) / (
            np.sqrt(2.0) * sigma
        )
        x_max = (lnM_obs_mu - self.logM_obs_upper * np.log(10.0)) / (
            np.sqrt(2.0) * sigma
        )

        if x_max > 4.0:
            #  pylint: disable-next=no-member
            return (special.erfc(x_min) - special.erfc(x_max)) / 2.0
        else:
            #  pylint: disable-next=no-member
            return (special.erf(x_min) - special.erf(x_max)) / 2.0


class ClusterMassRich(ClusterMass):
    """Cluster Mass Richness proxy."""

    def __init__(
        self, pivot_mass, pivot_redshift, logMl: float = 13.0, logMu: float = 16.0
    ):
        """Initialize the ClusterMassRich object."""

        super().__init__()
        self.pivot_mass = pivot_mass
        self.pivot_redshift = pivot_redshift
        self.log_pivot_mass = pivot_mass * np.log(10.0)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)
        self.logMl = logMl
        self.logMu = logMu

        # Updatable parameters
        self.mu_p0 = parameters.create()
        self.mu_p1 = parameters.create()
        self.mu_p2 = parameters.create()
        self.sigma_p0 = parameters.create()
        self.sigma_p1 = parameters.create()
        self.sigma_p2 = parameters.create()

        self.logM_obs_min = 0.0  # pylint: disable-msg=invalid-name
        self.logM_obs_max = np.inf  # pylint: disable-msg=invalid-name

    @final
    def _update_cluster_mass(self, params: ParamsMap):
        """Perform any updates necessary after the parameters have being updated.

        This implementation has nothing to do."""

    @final
    def _reset_cluster_mass(self) -> None:
        """Reset the ClusterMass object.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def read(self, sacc_data: sacc.Sacc):
        """Method to read the data for this source from the SACC file."""

    def cluster_mass_lnM_obs_mu_sigma(self, logM, z):
        """Return the mean and standard deviation of the observed mass."""

        lnM = logM * np.log(10)

        lnM_obs_mu = (
            self.mu_p0
            + self.mu_p1 * (lnM - self.log_pivot_mass)
            + self.mu_p2 * (np.log1p(z) - self.log1p_pivot_redshift)
        )
        sigma = (
            self.sigma_p0
            + self.sigma_p1 * (lnM - self.log_pivot_mass)
            + self.sigma_p2 * (np.log1p(z) - self.log1p_pivot_redshift)
        )
        # sigma = abs(sigma)
        return [lnM_obs_mu, sigma]

    def gen_bins_by_array(self, logM_obs_bins: np.ndarray) -> List[ClusterMassArgument]:
        """Generate bins by an array of bin edges."""

        if len(logM_obs_bins) < 2:
            raise ValueError("logM_obs_bins must have at least two elements")

        return [
            ClusterMassRichBinArgument(
                self, self.logMl, self.logMu, logM_obs_lower, logM_obs_upper
            )
            for logM_obs_lower, logM_obs_upper in itertools.pairwise(logM_obs_bins)
        ]

    def point_arg(self, logM_obs: float) -> ClusterMassArgument:
        """Return the argument generator of the cluster mass function."""

        return ClusterMassRichPointArgument(self, self.logMl, self.logMu, logM_obs)

    def gen_bin_from_tracer(self, tracer: sacc.BaseTracer) -> ClusterMassArgument:
        """Return the argument for the given tracer."""

        if not isinstance(tracer, sacc_support.BinRichnessTracer):
            raise ValueError("Tracer must be a BinRichnessTracer")

        return ClusterMassRichBinArgument(
            self, self.logMl, self.logMu, tracer.richness_lower, tracer.richness_upper
        )
