"""Basic likelihood infrastructure.

This module provides the base class :class:`Likelihood`, which is the class
from which all concrete firecrown likelihoods must descend.

It also provides the function :meth:`load_likelihood` which reads a
likelihood script to create an object of some subclass of :class:`Likelihood`.

How to use a :class:`Likelihood` object
.......................................

The class :class:`Likelihood` is designed to support repeated calculations
of the likelihood of the observation of some specific data, given a specified
theory.

The data for which the likelihood is being calculated is set when the
:meth:`read` method of the likelihood is called. It is expected that this
will be done only once in the lifetime of any likelihood object. In the
specific case of a :class:`GaussFamily` likelihood, these data include both
a *data vector* and a *covariance matrix*, which must be present in the
:class:`Sacc` object given to the :meth:`read` method.

The theory predictions that are used in the calculation of a likelihood are
expected to change for different calls to the :meth:`compute_loglike` method.
In order to prepare a :class:`Likelihood` object for each call to
:meth:`compute_loglike`, the following sequence of calls must be made (note
that this is done by the Firecrown infrastructure when you are using Firecrown
with any of the supported sampling frameworks):

#. create the `Likelihood` object `like` using the concrete class name as a
   factory function
#. call :meth:`read` passing in the :class:`sacc.Sacc` object containing all
   the necessary data
#. for each call to :meth:`calculate_loglike`, prepare `like` for the new
   calculation:

    #. call :meth:`update` on the :class:`ParamsMap` object you are using the
       this likelihood.
    #. call :meth:`prepare` on the :class:`ModelingTools` object you are using.
    #. call :meth:`update` on `like`, passing in the :class:`ParamsMap` you
       just updated.
#. call :meth:`calculate_loglike` passing the current :class:`ModelingTools`
   object.
#. call :meth:`reset` to free any held resources and to prepare `like` for the
   next cycle.

Note that repeated calls to :meth:`update` on a :class:`Likelihood` object, if
there is no intervening call of :meth:`reset`, have no effect. This is necessary
cause the same object many be used in several places in any given :class:`Likelihood`,
but object should only be updated once.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import warnings
from abc import abstractmethod
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import sacc

from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import Updatable


class Likelihood(Updatable):
    """Likelihood is an abstract class.

    Concrete subclasses represent specific likelihood forms (e.g. gaussian with
    constant covariance matrix, or Student's t, etc.).

    Concrete subclasses must have an implementation of both :meth:`read` and
    :meth:`compute_loglike`. Note that abstract subclasses of Likelihood might implement
    these methods, and provide other abstract methods for their subclasses to implement.
    """

    def __init__(self, parameter_prefix: None | str = None) -> None:
        """Default initialization for a base Likelihood object.

        :params parameter_prefix: The prefix to prepend to all parameter names
        """
        super().__init__(parameter_prefix=parameter_prefix)

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file.

        :param sacc_data: The SACC data object to be read
        """

    def make_realization_vector(self) -> npt.NDArray[np.float64]:
        """Create a new realization of the model.

        This new realization uses the previously computed theory vector and covariance
        matrix.

        :return: the new realization of the theory vector
        """
        raise NotImplementedError(
            "This class does not implement make_realization_vector."
        )

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True, strict: bool = True
    ) -> sacc.Sacc:
        """Create a new realization of the model.

        This realization uses the previously computed theory vector and covariance
        matrix.

        :param sacc_data: The SACC data object containing the covariance matrix
        :param add_noise: If True, add noise to the realization. If False, return
            only the theory vector.
        :param strict: If True, check that the indices of the realization cover
            all the indices of the SACC data object.

        :return: the new SACC object containing the new realization
        """

    @abstractmethod
    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood of generic CCL data.

        :param tools: the ModelingTools to be used in calculating the likelihood
        :return: the log-likelihood
        """


class NamedParameters:
    """Provides access to a set of parameters of a given set of types.

    Access to the parameters is provided by a type-safe interface. Each of the
    access functions assures that the parameter value it returns is of the
    specified type.

    """

    def __init__(
        self,
        mapping: (
            None
            | Mapping[
                str,
                str
                | int
                | bool
                | float
                | npt.NDArray[np.int64]
                | npt.NDArray[np.float64],
            ]
        ) = None,
    ):
        """Initialize the object from the supplied mapping of values.

        :param mapping: the mapping from strings to values used for initialization
        """
        if mapping is None:
            self.data = {}
        else:
            self.data = dict(mapping)

    def get_bool(self, name: str, default_value: None | bool = None) -> bool:
        """Return the named parameter as a bool.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, bool)
        return val

    def get_string(self, name: str, default_value: None | str = None) -> str:
        """Return the named parameter as a string.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, str)
        return val

    def get_int(self, name: str, default_value: None | int = None) -> int:
        """Return the named parameter as an int.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, int)
        return val

    def get_float(self, name: str, default_value: None | float = None) -> float:
        """Return the named parameter as a float.

        :param name: the name of the parameter to be returned
        :param default_value: the default value if the parameter is not found
        :return: the value of the parameter (or the default value)
        """
        if default_value is None:
            val = self.data[name]
        else:
            val = self.data.get(name, default_value)

        assert isinstance(val, float)
        return val

    def get_int_array(self, name: str) -> npt.NDArray[np.int64]:
        """Return the named parameter as a numpy array of int.

        :param name: the name of the parameter to be returned
        :return: the value of the parameter
        """
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.int64)
        assert val.dtype == np.int64
        return val

    def get_float_array(self, name: str) -> npt.NDArray[np.float64]:
        """Return the named parameter as a numpy array of float.

        :param name: the name of the parameter to be returned
        :return: the value of the parameter
        """
        tmp = self.data[name]
        assert isinstance(tmp, np.ndarray)
        val = tmp.view(dtype=np.float64)
        assert val.dtype == np.float64
        return val

    def to_set(
        self,
    ) -> set[
        str | int | bool | float | npt.NDArray[np.int64] | npt.NDArray[np.float64]
    ]:
        """Return the contained data as a set.

        :return: the value of the parameter as a set
        """
        return set(self.data)

    def set_from_basic_dict(
        self,
        basic_dict: dict[
            str,
            str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
        ],
    ) -> None:
        """Set the contained data from a dictionary of basic types.

        :param basic_dict: the mapping from strings to values used for initialization
        """
        for key, value in basic_dict.items():
            if isinstance(value, (str, float, int, bool)):
                self.data = dict(self.data, **{key: value})
            elif isinstance(value, Sequence):
                if all(isinstance(v, float) for v in value):
                    self.data = dict(self.data, **{key: np.array(value)})
                elif all(isinstance(v, bool) for v in value):
                    self.data = dict(
                        self.data, **{key: np.array(value, dtype=np.int64)}
                    )
                elif all(isinstance(v, int) for v in value):
                    self.data = dict(
                        self.data, **{key: np.array(value, dtype=np.int64)}
                    )
                else:
                    raise ValueError(f"Invalid type for sequence value: {value}")
            else:
                raise ValueError(f"Invalid type for value: {value}")

    def convert_to_basic_dict(
        self,
    ) -> dict[
        str,
        str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
    ]:
        """Convert a NamedParameters object to a dictionary of built-in types.

        :return: a dictionary containing the parameters as built-in Python types
        """
        basic_dict: dict[
            str,
            str | float | int | bool | Sequence[float] | Sequence[int] | Sequence[bool],
        ] = {}

        for key, value in self.data.items():
            if isinstance(value, (str, float, int, bool)):
                basic_dict[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype == np.int64:
                    basic_dict[key] = value.ravel().tolist()
                elif value.dtype == np.float64:
                    basic_dict[key] = value.ravel().tolist()
                else:
                    raise ValueError(f"Invalid type for sequence value: {value}")
            else:
                raise ValueError(f"Invalid type for value: {value}")
        return basic_dict


def load_likelihood_from_module_type(
    module: types.ModuleType,
    build_parameters: NamedParameters,
    build_likelihood_name: str = "build_likelihood",
) -> tuple[Likelihood, ModelingTools]:
    """Loads a likelihood from a module type.

    After loading, this method returns a tuple of the likelihood and
    the modeling tools.

    This function is used by both :meth:`load_likelihood_from_script` and
    :meth:`load_likelihood_from_module`. It is not intended to be called
    directly.

    :param module: a loaded module
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    :return: a tuple of the likelihood and the modeling tools
    """
    if not hasattr(module, build_likelihood_name):
        if not hasattr(module, "likelihood"):
            raise AttributeError(
                f"Firecrown initialization module {module.__name__} in "
                f"{module.__file__} does not define "
                f"a `{build_likelihood_name}` factory function."
            )
        warnings.warn(
            f"The use of a likelihood variable in Firecrown's initialization "
            f"module is deprecated. Any parameters passed to the likelihood "
            f"will be ignored. The module should define the `{build_likelihood_name}` "
            f"factory function.",
            category=DeprecationWarning,
        )
        likelihood = module.likelihood
        tools = ModelingTools()
    else:
        build_likelihood = getattr(module, build_likelihood_name)
        if not callable(build_likelihood):
            raise TypeError(
                f"The factory function `{build_likelihood_name}` must be a callable."
            )
        build_return = build_likelihood(build_parameters)
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
) -> tuple[Likelihood, ModelingTools]:
    """Loads a likelihood script.

    After loading, this method returns a tuple of the likelihood and
    the modeling tools.

    :param filename: script filename
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    :return: a tuple of the likelihood and the modeling tools
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
) -> tuple[Likelihood, ModelingTools]:
    """Loads a likelihood from a module.

    After loading, this method returns a tuple of the likelihood and
    the modeling tools.

    :param module: module name
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    :return: a tuple of the likelihood and the modeling tools
    """
    try:
        # Try importing the entire string as a module first
        try:
            mod = importlib.import_module(module)
            func = "build_likelihood"
        except ImportError as sub_exc:
            # If it fails, split and try importing as module.function
            if "." not in module:
                raise sub_exc
            module_name, func = module.rsplit(".", 1)
            mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Unrecognized Firecrown initialization module '{module}'. "
            f"The module must be either a module_name or a module_name.func "
            f"where func is the factory function."
        ) from exc

    return load_likelihood_from_module_type(
        mod, build_parameters, build_likelihood_name=func
    )


def load_likelihood(
    likelihood_name: str, build_parameters: NamedParameters
) -> tuple[Likelihood, ModelingTools]:
    """Loads a likelihood from the provided likelihood_name.

    After loading, this method returns a tuple of the likelihood and
    the modeling tools.

    :param likelihood_name: script filename or module name
    :param build_parameters: a NamedParameters object containing the factory
        function parameters
    :return: a tuple of the likelihood and the modeling tools
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
