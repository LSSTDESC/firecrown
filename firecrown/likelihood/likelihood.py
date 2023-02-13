"""Basic likelihood infrastructure


This module provides the base class :python:`Likelihood`, which is the class
from which all concrete firecrown likelihoods must descend.

It also provides the function :python:`load_likelihood` which reads a
likelihood script to create an object of some subclass of :python:`Likelihood`.

"""

from __future__ import annotations
from typing import List, Dict, Tuple, Union, Optional
from abc import abstractmethod
import warnings
import importlib
import importlib.util
import os
import sys
import numpy as np
import numpy.typing as npt

import sacc

from ..updatable import Updatable, UpdatableCollection
from ..modeling_tools import ModelingTools


class Likelihood(Updatable):
    """Likelihood is an abstract class. Concrete subclasses represent specific
    likelihood forms (e.g. gaussian with constant covariance matrix, or Student's t,
    etc.).

    Concrete subclasses must have an implementation of both *read* and
    *compute_loglike*. Note that abstract subclasses of Likelihood might implement
    these methods, and provide other abstract methods for their subclasses to implement.
    """

    def __init__(self) -> None:
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
    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood of generic CCL data."""


def load_likelihood(
    filename: str, build_parameters: Dict[str, Union[str, int, bool, float, np.ndarray]]
) -> Tuple[Likelihood, ModelingTools]:
    """Loads a likelihood script and returns an instance

    :param filename: script filename
    :param build_parameters: a dictionary containing the factory function parameters
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

    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_likelihood"):
        if not hasattr(mod, "likelihood"):
            raise AttributeError(
                f"Firecrown initialization script {filename} does not define "
                f"a `build_likelihood` factory function."
            )
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "The use of a likelihood variable in Firecrown's initialization "
            "script is deprecated. Any parameters passed to the likelihood "
            "will be ignored. The script should define a `build_likelihood` "
            "factory function.",
            category=DeprecationWarning,
        )
        likelihood = mod.likelihood
    else:
        if not callable(mod.build_likelihood):
            raise TypeError(
                "The factory function `build_likelihood` must be a callable."
            )
        build_return = mod.build_likelihood(build_parameters)
        if isinstance(build_return, tuple):
            likelihood, tools = build_return
        else:
            likelihood = build_return
            tools = ModelingTools()

    if not isinstance(likelihood, Likelihood):
        raise TypeError(
            f"The returned likelihood must be a Firecrown's `Likelihood` type, "
            f"received {type(likelihood)} instead."
        )

    return likelihood, tools
