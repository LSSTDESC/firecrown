"""Shared types for test fixtures."""

from typing import TypedDict

from numcosmo_py import Nc


class NumCosmoCosmologyFixture(TypedDict):
    """Values provided by the NumCosmo cosmology fixtures."""

    cosmo: Nc.HICosmo
    p_ml: Nc.PowspecML
    p_mnl: Nc.PowspecMNL
    dist: Nc.Distance
