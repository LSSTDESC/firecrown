"""The classes in this file define the firecrown-CCL API.

Some Notes:

 - Each subclass which inherits from a given class is expected to define any
   methods defined in the parent with the same call signature. See the base
   class docstrings for additional instructions.
 - If a base class includes a class-level doc string, then
   the `__init__` function of the subclass should define at least those
   arguments and/or keyword arguments in the class-level doc string.
 - Attributes ending with an underscore are set after the call to
   `apply`/`compute`/`render`.
 - Attributes define in the `__init__` method should be considered constant
   and not changed after instantiation.
 - Objects inheriting from `Systematic` should only adjust source/statistic
   properties ending with an underscore.
 - The `read` methods are called after all objects are made and are used to
   read any additional data.
"""

from __future__ import annotations
from typing import Dict, Optional
from abc import ABC, abstractmethod
import numpy as np
import pyccl
import sacc
import importlib
import importlib.util
import os

import firecrown
from ..parser_constants import FIRECROWN_RESERVED_NAMES


class Statistic(ABC):
    """A statistic (e.g., two-point function, mass function, etc.).

    Parameters
    ----------
    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    """

    # Why does this exist? It is not marked as abstract, so derived classes are
    # not required to implement it. It has no behavior, so any code calling it
    # and expecting some side effect will not be satisfied.
    def read(self, sacc_data: sacc.Sacc, sources) -> None:
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        sources : dict
            A dictionary mapping sources to their objects. These sources do
            not have to have been rendered.
        """
        pass

    def update_params(self, params):
        pass

    @abstractmethod
    def compute(
            self,
            cosmo: pyccl.Cosmology,
            params: Dict,
            sources: Dict,
            systematics: Optional[Dict] = None,
    ) -> float:
        """Compute a statistic from sources, applying any systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        sources : dict
            A dictionary mapping sources to their objects. The sources must
            already have been rendered by calling `render` on them.
        systematics : dict
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        raise NotImplementedError("Method `compute` is not implemented!")


class Systematic():
    """The systematic (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    pass

    # def apply(self, cosmo: pyccl.Cosmology, params: Dict, source_or_statistic):
    #     """Apply systematics to a source.
    #
    #     Parameters
    #     ----------
    #     cosmo : pyccl.Cosmology
    #         A pyccl.Cosmology object.
    #     params : dict
    #         A dictionary mapping parameter names to their current values.
    #     source_or_statistic : a source or statistic object
    #         The source or statistic to which apply systematics.
    #     """
    #     raise NotImplementedError("Method `apply` is not implemented!")


class Source(ABC):
    """The source (e.g., a sample of lenses).

    Parameters
    ----------
    scale : 1.0, optional
        The default scale for this source.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.
    """

    @abstractmethod
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    def update_params(self, params):
        pass

    def render(self, cosmo, params, systematics=None):
        """Render a source by applying systematics.

        This method should compute the final scale factor for the source
        as `scale_` and then apply any systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        systematics : dict, optional
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        raise NotImplementedError("Method `render` is not implemented!")


class LogLike(object):
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

    def set_sources(self, sources):
        self.sources = sources

    def set_statistics(self, statistics):
        self.statistics = statistics

    def set_systematics(self, systematics):
        self.systematics = systematics

    def set_params_names(self, params_names):
        self.params_names = params_names

    def get_params_names(self):
        if hasattr(self, "params_names"):
            return self.params_names
        else:
            return []

    def read(self, sacc_data, sources, statistics):
        """Read the covariance matrirx for this likelihood from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        sources : dict
            A dictionary mapping sources to their objects. These sources do
            not have to have been rendered.
        statistics : dict
            A dictionary mapping statistics to their objects. These statistics do
            not have to have been rendered.
        """
        pass

    def compute(self, data, theory, **kwargs):
        """Compute the log-likelihood.

        Parameters
        ----------
        data : dict of arrays
            A dictionary mapping the names of the statistics to their
            values in the data.
        theory : dict of arrays
            A dictionary mapping the names of the statistics to their
            predictions.
        **kwargs : extra keyword arguments
            Any extra keyword arguments can be used by subclasses.

        Returns
        -------
        loglike : float
            The log-likelihood.
        """
        raise NotImplementedError(
            "Method `compute_loglike` is not implemented!")

    def assemble_data_vector(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the log-likelihood.

        Parameters
        ----------
        data : dict of arrays
            A dictionary mapping the names of the statistics to their
            values.

        Returns
        -------
        data_vector : array-like
            The data vector.
        """
        dv = [np.atleast_1d(data[stat]) for stat in self.data_vector]
        return np.concatenate(dv, axis=0)

    def compute_loglike(self, cosmo: pyccl.core.Cosmology,
                        parameters):
        """Compute the log-likelihood of generic CCL data.

        Parameters
        ----------
        cosmo : a `pyccl.Cosmology` object
            A cosmology.
        parameters : dict
            Dictionary mapping parameters to their values.

        Returns
        -------
        loglike : float
            The computed log-likelihood.
        """
        for src in self.sources.values():
            # TODO: Our test suite does not currently run this code.
            # The following assert does not cause any test to fail.
            # assert (False)
            src.update_params(parameters)
            src.render(cosmo, parameters, systematics=self.systematics)

        _data = {}
        _theory = {}
        for name, stat in self.statistics.items():
            stat.update_params(parameters)
            stat.compute(cosmo, parameters, self.sources,
                         systematics=self.systematics)
            _data[name] = stat.measured_statistic_
            _theory[name] = stat.predicted_statistic_

        return self.compute(_data, _theory)


def load_likelihood(firecrownIni):
    filename, file_extension = os.path.splitext(firecrownIni)

    ext = file_extension.lower()

    if ext == ".yaml":
        config, data = firecrown.parse(firecrownIni)

        analyses = set(data.keys()) - set(
            FIRECROWN_RESERVED_NAMES + ["priors"]
        )

        if len(analyses) != 1:
            raise ValueError("Only a single likelihood per file is supported")

        for analysis in analyses:
            likelihood = data[analysis]["data"]["likelihood"]
            likelihood.set_sources(data[analysis]["data"]["sources"])
            likelihood.set_systematics(
                data[analysis]["data"]["systematics"]
            )
            likelihood.set_statistics(
                data[analysis]["data"]["statistics"]
            )

    elif ext == ".py":
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
