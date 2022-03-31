"""The base likelihood class

Some Notes:

    -
"""

from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod
from typing import final
import numpy as np
import pyccl
import sacc
import importlib
import importlib.util
import os

class Likelihood(object):
    """The log-likelihood (e.g., a Gaussian, T-distribution, etc.).

    Parameters
    ----------
    data_vector : list of str
        A list of the statistics in the config file in the order you want them
        to appear in the covariance matrix.

    Attributes
    ----------
    cov : array-like, shape (n, n)
        The covariance matrix for the data vector.
    inv_cov : array-like, shape (n, n)
        The inverse of the covariance matrix.
    """

    def set_params_names(self, params_names: List[str]):
        self.params_names = params_names

    def get_params_names(self):
        if hasattr(self, "params_names"):
            return self.params_names
        else:
            return []

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Read the covariance matrirx for this likelihood from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    @abstractmethod
    def compute_loglike(self, cosmo: pyccl.Cosmology, params: Dict[str, float]):
        """Compute the log-likelihood of generic CCL data.

        Parameters
        ----------
        cosmo : a `pyccl.Cosmology` object
            A cosmology.
        params : dict
            Dictionary mapping parameters to their values.

        Returns
        -------
        loglike : float
            The computed log-likelihood.
        """
        pass

def load_likelihood(firecrownIni):
    filename, file_extension = os.path.splitext(firecrownIni)

    ext = file_extension.lower()

    if ext == ".py":
        inifile = os.path.basename(firecrownIni)
        inipath = os.path.dirname(firecrownIni)
        modname, _ = os.path.splitext(inifile)

        spec = importlib.util.spec_from_file_location(modname, firecrownIni)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "likelihood"):
            raise ValueError(
                f"Firecrown initialization file {firecrownIni} does not define a likelihood."
            )

        likelihood = mod.likelihood
    else:
        raise ValueError(
            f"Unrecognized Firecrown initialization file {firecrownIni}."
        )

    return likelihood
