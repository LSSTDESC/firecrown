"""The base likelihood class

Some Notes:

    -
"""

from __future__ import annotations
from typing import List
from abc import abstractmethod
import pyccl
import sacc
import importlib
import importlib.util
import os

from ..parameters import ParamsMap
from ..updatable import Updatable


class Likelihood(Updatable):
    """Likelihood is an abstract classes. Concrete subclasses represent specific
    likelihood forms (e.g. gaussian with constant covariance matrix, or Student's t,
    etc.).

    Concrete subclasses must have an implementation of both *read* and
    *compute_loglike*. Note that abstract subclasses of Likelihood might implement
    these methods, and provide other abstract methods for their subclasses to implement.
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
    def compute_loglike(self, cosmo: pyccl.Cosmology, params: ParamsMap):
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


def load_likelihood(firecrown_ini):
    filename, file_extension = os.path.splitext(firecrown_ini)

    ext = file_extension.lower()

    if ext == ".py":
        inifile = os.path.basename(firecrown_ini)
        modname, _ = os.path.splitext(inifile)

        spec = importlib.util.spec_from_file_location(modname, firecrown_ini)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "likelihood"):
            raise ValueError(
                f"Firecrown initialization file {firecrown_ini} does not define a likelihood."
            )

        likelihood = mod.likelihood
    else:
        raise ValueError(f"Unrecognized Firecrown initialization file {firecrown_ini}.")

    return likelihood
