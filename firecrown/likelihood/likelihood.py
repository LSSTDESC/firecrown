"""Basic likelihood infrastructure


This module provides the base class :python:`Likelihood`, which is the class
from which all concrete firecrown likelihoods must descend.

It also provides the function :python:`load_likelihood` which reads a
likelihood script to create an object of some subclass of :python:`Likelihood`.

"""

from __future__ import annotations
from typing import List, Optional, final
from abc import abstractmethod
import importlib
import importlib.util
import os
import sys
import numpy as np
import numpy.typing as npt

import pyccl
import pyccl.nl_pt

import sacc

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
        """Default initialization for a base Likelihood object."""
        super().__init__()

        self.params_names: Optional[List[str]] = None
        self.predicted_data_vector: Optional[npt.NDArray[np.double]] = None
        self.measured_data_vector: Optional[npt.NDArray[np.double]] = None
        self.inv_cov: Optional[npt.NDArray[np.double]] = None
        self.statistics: UpdatableCollection = UpdatableCollection()

    def set_params_names(self, params_names: List[str]) -> None:
        """Set the parameter names for this Likelihood."""
        self.params_names = params_names

    def get_params_names(self) -> Optional[List[str]]:
        """Return the parameter names of this Likelihood."""

        # TODO: This test for the presence of the instance variable
        # params_names seems unnecessary; we set the instance variable
        # to None in the initializer. Since we would return an empty list
        # if we did *not* have an instance variable, should we just make
        # the default value used in the initializer an empty list?
        if hasattr(self, "params_names"):
            return self.params_names
        return []

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Read the covariance matrix for this likelihood from the SACC file."""

    @abstractmethod
    def compute_loglike(self, cosmo: pyccl.Cosmology) -> float:
        """Compute the log-likelihood of generic CCL data."""


class CosmologyContainer:
    def __init__(self, cosmo: pyccl.Cosmology):
        self.ccl_cosmo = cosmo
        self.pt_calculator: pyccl.nl_pt.PTCalculator = None
        self.hm_calculator: pyccl.halomodel.HMCalculator = None

    def add_pk(self, name: str, pk: pyccl.Pk2D):
        self.pk[name] = pk

    def get_pk(self, name: str) -> pyccl.Pk2D:
        if name in self.pk:
            return self.pk[name]
        return self.ccl_cosmo.get_nonlin_power(name)


class CosmoSystematicsLikelihood(Likelihood):
    def __init__(self, systematics):
        super().__init__()
        self.systematics = UpdatableCollection(systematics)

    @final
    def apply_systematics(self, cosmo: pyccl.Cosmology) -> CosmologyContainer:
        cosmo_container = CosmologyContainer(cosmo)
        for systematic in self.systematics:
            cosmo_container = systematic.apply(cosmo_container)
        return cosmo_container


# This should probably live somewhere else and derive from some base Systematics
# calss
class PTSystematic:
    def __init__(self, *ptc_args, **ptc_kwargs):
        self.ptc = pyccl.nl_pt.PTCalculator(*self.ptc_args, **self.ptc_kwargs)

    def apply(self, cosmo_container: CosmologyContainer) -> CosmologyContainer:
        # P_lin(k) at z=0
        pk_lin_z0 = pyccl.linear_matter_power(cosmo_container.ccl_cosmo, self.ptc.ks, 1.)
        # Compute the perturbative quantities
        self.ptc.update_pk(pk_lin_z0)

        cosmo_container.pt_calculator = self.ptc
        return cosmo_container


def load_likelihood(filename: str) -> Likelihood:
    """Loads a likelihood script and returns an instance

    :param filename: script filename
    """
    _, file_extension = os.path.splitext(filename)

    ext = file_extension.lower()

    if ext != ".py":
        raise ValueError(f"Unrecognized Firecrown initialization file {filename}.")

    inifile = os.path.basename(filename)
    modname, _ = os.path.splitext(inifile)
    script_path = os.path.dirname(os.path.abspath(filename))

    spec = importlib.util.spec_from_file_location(
        modname, filename, submodule_search_locations=[script_path]
    )

    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod

    if spec.loader is None:
        raise ImportError(f"Spec for module '{modname}' has no loader.")

    try:
        spec.loader.exec_module(mod)
    except FileNotFoundError as error:
        raise ImportError(f"{error.strerror}: {filename}") from error

    if not hasattr(mod, "likelihood"):
        raise ValueError(
            f"Firecrown initialization file {filename} does not define a likelihood."
        )

    likelihood = mod.likelihood
    assert isinstance(likelihood, Likelihood)

    return likelihood
