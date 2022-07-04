"""The base likelihood class

Some Notes:

    -
"""

from __future__ import annotations
from typing import List, Optional
from abc import abstractmethod
import pyccl
import sacc
import importlib
import importlib.util
import os
import sys
import numpy as np
import numpy.typing as npt

from ..updatable import Updatable, UpdatableCollection


class Likelihood(Updatable):
    """Likelihood is an abstract class. Concrete subclasses represent specific
    likelihood forms (e.g. gaussian with constant covariance matrix, or Student's t,
    etc.).

    Concrete subclasses must have an implementation of both *read* and
    *compute_loglike*. Note that abstract subclasses of Likelihood might implement
    these methods, and provide other abstract methods for their subclasses to implement.
    """

    def __init__(self):
        self.params_names: Optional[List[str]] = None
        self.predicted_data_vector: Optional[npt.NDArray[np.double]] = None
        self.measured_data_vector: Optional[npt.NDArray[np.double]] = None
        self.inv_cov: Optional[npt.NDArray[np.double]] = None
        self.statistics: UpdatableCollection = UpdatableCollection()

    def set_params_names(self, params_names: List[str]):
        self.params_names = params_names

    def get_params_names(self):
        if hasattr(self, "params_names"):
            return self.params_names
        else:
            return []

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Read the covariance matrirx for this likelihood from the SACC file."""
        pass

    @abstractmethod
    def compute_loglike(self, cosmo: pyccl.Cosmology):
        """Compute the log-likelihood of generic CCL data.

        Parameters
        ----------
        cosmo : a `pyccl.Cosmology` object
            A cosmology.

        Returns
        -------
        loglike : float
            The computed log-likelihood.
        """
        pass


def load_likelihood(filename: str) -> Likelihood:
    _, file_extension = os.path.splitext(filename)

    ext = file_extension.lower()

    if ext != ".py":
        raise ValueError(f"Unrecognized Firecrown initialization file {filename}.")

    inifile = os.path.basename(filename)
    modname, _ = os.path.splitext(inifile)
    script_path = os.path.dirname(os.path.abspath(filename))

    spec = importlib.util.spec_from_file_location(modname, filename, submodule_search_locations=[script_path])

    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod

    if spec.loader is None:
        raise ImportError(f"Spec for module '{modname}' has no loader.")

    try:
        spec.loader.exec_module(mod)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {filename}") from e

    if not hasattr(mod, "likelihood"):
        raise ValueError(
            f"Firecrown initialization file {filename} does not define a likelihood."
        )

    likelihood = mod.likelihood
    return likelihood
