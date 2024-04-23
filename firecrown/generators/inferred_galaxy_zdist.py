"""This module deals with the generation of inferred galaxy redshift distributions.
"""

from itertools import combinations_with_replacement, chain
from typing import TypedDict, TypeVar, Type, Self
from dataclasses import dataclass
from enum import Enum, auto
import re

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, erf, erfc

import yaml
from yaml import CLoader as Loader
from yaml import CDumper as Dumper

import sacc
from sacc.data_types import required_tags

from firecrown.metadata.two_point import InferredGalaxyZDist

Y1_ALPHA = 0.94
Y1_BETA = 2.0
Y1_Z0 = 0.26

Y10_ALPHA = 0.90
Y10_BETA = 2.0
Y10_Z0 = 0.28


class ZDistLSSTSRD:
    """LSST Inferred galaxy redshift distributions.


    Inferred galaxy redshift distribution based on the LSST Science Requirements
    Document (SRD).
    """

    def __init__(self, alpha: float, beta: float, z0: float) -> None:
        """Initialize the LSST Inferred galaxy redshift distribution.

        :param alpha: The alpha parameter of the distribution.
        :param beta: The beta parameter of the distribution.
        :param z0: The z0 parameter of the distribution.
        """
        self.alpha = alpha
        self.beta = beta
        self.z0 = z0

    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        """Create a ZDistLSSTSRD object from a YAML file.

        :param filename: The path to the YAML file.
        :return: A ZDistLSSTSRD object.
        """
        with open(filename, "r", encoding="utf8") as f:
            data = yaml.load(f, Loader=Loader)
            return cls(**data)

    @classmethod
    def year_1(
        cls, alpha: float = Y1_ALPHA, beta: float = Y1_BETA, z0: float = Y1_Z0
    ) -> Self:
        """Create a ZDistLSSTSRD object for the first year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 1.

        :param alpha: The alpha parameter using the default value of 0.94.
        :param beta: The beta parameter using the default value of 2.0.
        :param z0: The z0 parameter using the default value of 0.26.
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0)

    @classmethod
    def year_10(
        cls, alpha: float = Y10_ALPHA, beta: float = Y10_BETA, z0: float = Y10_Z0
    ) -> Self:
        """Create a ZDistLSSTSRD object for the tenth year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 10.

        :param alpha: The alpha parameter using the default value of 0.90.
        :param beta: The beta parameter using the default value of 2.0.
        :param z0: The z0 parameter using the default value of 0.28.
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0)

    def distribution(self, z: npt.NDArray) -> npt.NDArray:
        """Generate the inferred galaxy redshift distribution."""
        norma = self.alpha / (self.z0 * gamma((1.0 + self.beta) / self.alpha))

        return (
            norma * (z / self.z0) ** self.beta * np.exp(-((z / self.z0) ** self.alpha))
        )

    def _integrated_gaussian(
        self, zpl: float, zpu: float, sigma_z: float, z: npt.NDArray
    ) -> npt.NDArray:
        """Generate the integrated Gaussian distribution."""
        denom = np.sqrt(2.0) * sigma_z * (1.0 + z)
        return (erf((z - zpl) / denom) - erf((z - zpu) / denom)) / erfc(-z / denom)

    def binned_distribution(
        self, zpl: float, zpu: float, sigma_z: float, z: npt.NDArray
    ) -> npt.NDArray:
        """Generate the inferred galaxy redshift distribution in bins."""
        true_dist = self.distribution(z)
        return self._integrated_gaussian(zpl, zpu, sigma_z, z) * true_dist
