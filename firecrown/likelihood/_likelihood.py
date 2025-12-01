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

# Import base classes from _base.py
from firecrown.likelihood._base import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools


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

    for name, obj in (("likelihood", likelihood), ("tools", tools)):
        if obj.is_updated():
            warnings.warn(
                f"The factory function returned a {name} object that is already in "
                f"an updated state. Any parameters currently set in the {name} will "
                f"be ignored. The object will be reset automatically.",
                category=UserWarning,
            )
            obj.reset()

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

    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod

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
