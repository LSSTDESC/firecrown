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
from typing import final
import numpy as np
import pyccl
import sacc
import importlib
import importlib.util
import os

import firecrown
from ..parser_constants import FIRECROWN_RESERVED_NAMES

def get_params_hash (params: Dict[str, float]):
    return repr(sorted(params.items()))

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

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    @final
    def update_params(self, params):
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.update_params(params)
        self._update_params(params)
    
    @abstractmethod
    def _update_params(self, params):
        pass

    @abstractmethod
    def compute(
        self,
        cosmo: pyccl.Cosmology,
        params: Dict[str, float],
    ) -> (np.ndarray, np.ndarray):
        """Compute a statistic from sources, applying any systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        """
        raise NotImplementedError("Method `compute` is not implemented!")


class Systematic():
    """The systematic (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    def read(self, sacc_data: sacc.Sacc):
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

    @final
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    @abstractmethod
    def get_scale(self) -> float:
        pass

    @final
    def update_params(self, params: Dict[str, float]):
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.update_params(params)
        self._update_params(params)
    
    @abstractmethod
    def _update_params(self, params: Dict[str, float]):
        pass

    @abstractmethod
    def create_tracer(self, cosmo: pyccl.Cosmology, params: Dict[str, float]):
        pass

    @final
    def get_tracer(self, cosmo: pyccl.Cosmology, params: Dict[str, float]):
        cur_hash = hash ((cosmo, get_params_hash (params)))
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracer
        else:
            self.tracer, _ = self.create_tracer (cosmo, params)
            self.cosmo_hash = cur_hash
            return self.tracer

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

