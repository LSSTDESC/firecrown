"""Basic likelihood infrastructure


This module provides the base class :python:`Likelihood`, which is the class
from which all concrete firecrown likelihoods must descend.

It also provides the function :python:`load_likelihood` which reads a
likelihood script to create an object of some subclass of :python:`Likelihood`.

"""

from __future__ import annotations
from typing import Mapping, Tuple, Union, Optional
from abc import abstractmethod
import types
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

    def __init__(self, parameter_prefix: Optional[str] = None) -> None:
        """Default initialization for a base Likelihood object."""
        super().__init__(parameter_prefix=parameter_prefix)

        self.predicted_data_vector: Optional[npt.NDArray[np.double]] = None
        self.measured_data_vector: Optional[npt.NDArray[np.double]] = None
        self.inv_cov: Optional[npt.NDArray[np.double]] = None
        self.statistics: UpdatableCollection = UpdatableCollection()

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Read the covariance matrix for this likelihood from the SACC file."""

    @abstractmethod
    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood of generic CCL data."""


class NamedParameters:
    """Provides access to a set of parameters of a given set of types.

    Access to the parameters is provided by a type-safe interface. Each of the
    access functions assures that the parameter value it returns is of the
    specified type.

    """

    def __init__(
        self,
        mapping: Optional[
            Mapping[
                str,
                Union[
                    str,
                    int,
                    bool,
                    float,
                    npt.NDArray[np.int64],
                    npt.NDArray[np.float64],
                ],
            ]
        ] = None,
    ):
        """Initialize the object from the supplied mapping of values."""
        if mapping is None:
            self.data = {}
        else:
            self.data = dict(mapping)

    def get_bool(self, name: str, default_value: Optional[bool] = None) -> bool:
        """Return the named parameter as a bool."""
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, bool)
        return val

    def get_string(self, name: str, default_value: Optional[str] = None) -> str:
        """Return the named parameter as a string."""
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, str)
        return val

    def get_int(self, name: str, default_value: Optional[int] = None) -> int:
        """Return the named parameter as an int."""
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, int)
        return val

    def get_float(self, name: str, default_value: Optional[float] = None) -> float:
        """Return the named parameter as a float."""
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, float)
        return val

    def get_int_array(self, name: str) -> npt.NDArray[np.int64]:
        """Return the named parameter as a numpy array of int."""
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.int64)
        assert val.dtype == np.int64
        return val

    def get_float_array(self, name: str) -> npt.NDArray[np.float64]:
        """Return the named parameter as a numpy array of float."""
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.float64)
        assert val.dtype == np.float64
        return val

    def to_set(self):
        """Return the contained data as a set."""
        return set(self.data)


def load_likelihood_from_module_type(
    module: types.ModuleType, build_parameters: NamedParameters
) -> Tuple[Likelihood, ModelingTools]:
    """Loads a likelihood and returns a tuple of the likelihood and
    the modeling tools.

    This function is used by both :python:`load_likelihood_from_script` and
    :python:`load_likelihood_from_module`. It is not intended to be called
    directly.

    :param module: a loaded module
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    """

    if not hasattr(module, "build_likelihood"):
        if not hasattr(module, "likelihood"):
            raise AttributeError(
                f"Firecrown initialization module {module.__name__} in "
                f"{module.__file__} does not define "
                f"a `build_likelihood` factory function."
            )
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "The use of a likelihood variable in Firecrown's initialization "
            "module is deprecated. Any parameters passed to the likelihood "
            "will be ignored. The module should define a `build_likelihood` "
            "factory function.",
            category=DeprecationWarning,
        )
        likelihood = module.likelihood
        tools = ModelingTools()
    else:
        if not callable(module.build_likelihood):
            raise TypeError(
                "The factory function `build_likelihood` must be a callable."
            )
        build_return = module.build_likelihood(build_parameters)
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

    if not isinstance(tools, ModelingTools):
        raise TypeError(
            f"The returned tools must be a Firecrown's `ModelingTools` type, "
            f"received {type(tools)} instead."
        )

    return likelihood, tools


def load_likelihood_from_script(
    filename: str, build_parameters: NamedParameters
) -> Tuple[Likelihood, ModelingTools]:
    """Loads a likelihood script and returns a tuple of the likelihood and
    the modeling tools.

    :param filename: script filename
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
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

    # Apparently, the spec can be None if the file extension is not .py
    # However, we already checked for that, so this should never happen.
    # if spec is None:
    #    raise ImportError(f"Could not load spec for module '{modname}' at: {filename}")
    # Instead, we just assert that it is not None.
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod

    # Apparently, the spec.loader can be None if the file extension is not
    # recognized. However, we already checked for that, so this should never
    # happen.
    # if spec.loader is None:
    #     raise ImportError(f"Spec for module '{modname}' has no loader.")
    # Instead, we just assert that it is not None.
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return load_likelihood_from_module_type(mod, build_parameters)


def load_likelihood_from_module(
    module: str, build_parameters: NamedParameters
) -> Tuple[Likelihood, ModelingTools]:
    """Loads a likelihood and returns a tuple of the likelihood and
    the modeling tools.

    :param module: module name
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    """

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(
            f"Unrecognized Firecrown initialization module {module}."
        ) from exc

    return load_likelihood_from_module_type(mod, build_parameters)


def load_likelihood(
    likelihood_name: str, build_parameters: NamedParameters
) -> Tuple[Likelihood, ModelingTools]:
    """Loads a likelihood and returns a tuple of the likelihood and
    the modeling tools.

    :param likelihood_name: script filename or module name
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    """

    try:
        return load_likelihood_from_script(likelihood_name, build_parameters)
    except ValueError:
        try:
            return load_likelihood_from_module(likelihood_name, build_parameters)
        except ValueError as exc:
            raise ValueError(
                f"Unrecognized Firecrown initialization file or module "
                f"{likelihood_name}."
            ) from exc
