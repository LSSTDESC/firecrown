"""Two point statistic support.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
import copy
import functools
import warnings

import numpy as np
import scipy.interpolate

import pyccl
import pyccl.nl_pt

from ....likelihood.likelihood import CosmologyBundle

from .statistic import Statistic
from .source.source import Source, Systematic
from ....parameters import ParamsMap, RequiredParameters, DerivedParameterCollection

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "galaxy_density_cl": "cl",
    "galaxy_density_xi": "NN",
    "galaxy_shearDensity_cl_e": "cl",
    "galaxy_shearDensity_xi_t": "NG",
    "galaxy_shear_cl_ee": "cl",
    "galaxy_shear_xi_minus": "GG-",
    "galaxy_shear_xi_plus": "GG+",
    "cmbGalaxy_convergenceDensity_xi": "NN",
    "cmbGalaxy_convergenceShear_xi_t": "NG",
}

ELL_FOR_XI_DEFAULTS = dict(minimum=2, midpoint=50, maximum=6e4, n_log=200)


def _ell_for_xi(*, minimum, midpoint, maximum, n_log):
    """Build an array of ells to sample the power spectrum for real-space
    predictions.
    """
    return np.concatenate(
        (
            np.linspace(minimum, midpoint - 1, midpoint - minimum),
            np.logspace(np.log10(midpoint), np.log10(maximum), n_log),
        )
    )


def _generate_ell_or_theta(*, minimum, maximum, n, binning="log"):
    if binning == "log":
        edges = np.logspace(np.log10(minimum), np.log10(maximum), n + 1)
        return np.sqrt(edges[1:] * edges[:-1])
    edges = np.linspace(minimum, maximum, n + 1)
    return (edges[1:] + edges[:-1]) / 2.0


@functools.lru_cache(maxsize=128)
def _cached_angular_cl(cosmo, tracers, ells, p_of_k_a=None):
    return pyccl.angular_cl(cosmo, *tracers, np.array(ells), p_of_k_a=p_of_k_a)


class TwoPoint(Statistic):
    """A two-point statistic (e.g., shear correlation function, galaxy-shear
    correlation function, etc.).

    Parameters
    ----------
    sacc_data_type : str
        The kind of two-point statistic. This must be a valid SACC data type that
        maps to one of the CCL correlation function kinds or a power spectra.
        Possible options are

          - galaxy_density_cl : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_density_xi : maps to 'gg' (a CCL angular position corr. function)
          - galaxy_shearDensity_cl_e : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_shearDensity_xi_t : maps to 'gl' (a CCL angular cross-correlation
            between position and shear)
          - galaxy_shear_cl_ee : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_shear_xi_minus : maps to 'l-' (a CCL angular shear corr.
            function xi-)
          - galaxy_shear_xi_plus : maps to 'l+' (a CCL angular shear corr.
            function xi-)
          - cmbGalaxy_convergenceDensity_xi : maps to 'gg' (a CCL angular position
            corr. function)
          - cmbGalaxy_convergenceShear_xi_t : maps to 'gl' (a CCL angular cross-
            correlation between position and shear)

    source0 : Source
        The first sources needed to compute this statistic.
    source1 : Source
        The second sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    ell_or_theta : dict, optional
        A dictionary of options for generating the ell or theta values at which
        to compute the statistics. This option can be used to have firecrown
        generate data without the corresponding 2pt data in the input SACC file.
        The options are:

         - min : float - The start of the binning.
         - max : float - The end of the binning.
         - n : int - The number of bins. Note that the edges of the bins start
           at `min` and end at `max`. The actual bin locations will be at the
           (possibly geometric) midpoint of the bin.
         - binning : str, optional - Pass 'log' to get logarithmic spaced bins and 'lin'
           to get linearly spaced bins. Default is 'log'.

    ell_or_theta_min : float, optional
        The minimum ell or theta value to keep. This minimum is applied after
        the ell or theta values are read and/or generated.
    ell_or_theta_max : float, optional
        The maximum ell or theta value to keep. This maximum is applied after
        the ell or theta values are read and/or generated.
    ell_for_xi : dict, optional
        A dictionary of options for making the ell values at which to compute
        Cls for use in real-space integrations. The possible keys are:

         - min : int, optional - The minimum angulare wavenumber to use for
           real-space integrations. Default is 2.
         - mid : int, optional - The midpoint angular wavenumber to use for
           real-space integrations. The angular wavenumber samples are linearly
           spaced at integers between `min` and `mid`. Default is 50.
         - max : float, optional - The maximum angular wavenumber to use for
           real-space integrations. The angular wavenumber samples are
           logarithmically spaced between `mid` and `max`. Default is 6e4.
         - n_log : int, optional - The number of logarithmically spaced angular
           wavenumber samples between `mid` and `max`. Default is 200.

    Attributes
    ----------
    ccl_kind : str
        The CCL correlation function kind or 'cl' for power spectra corresponding
        to the SACC data type.
    sacc_tracers : 2-tuple of str
        A tuple of the SACC tracer names for this 2pt statistic. Set after a
        call to read.
    ell_or_theta_ : np.ndarray
        The final array of ell/theta values for the statistic. Set after
        `compute` is called.
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    scale_ : float
        The final scale factor applied to the statistic. Set after `compute`
        is called. Note that this scale factor is already applied.

    """

    def __init__(
        self,
        sacc_data_type,
        source0: Source,
        source1: Source,
        systematics: Optional[List[Systematic]] = None,
        ell_for_xi=None,
        ell_or_theta=None,
        ell_or_theta_min=None,
        ell_or_theta_max=None,
    ):
        super().__init__()

        assert isinstance(source0, Source)
        assert isinstance(source1, Source)

        self.sacc_data_type = sacc_data_type
        self.source0 = source0
        self.source1 = source1
        self.systematics = systematics or []
        self.ell_for_xi = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            self.ell_for_xi.update(ell_for_xi)
        self.ell_or_theta = ell_or_theta
        self.ell_or_theta_min = ell_or_theta_min
        self.ell_or_theta_max = ell_or_theta_max

        self.data_vector = None
        self.theory_vector = None

        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(
                f"The SACC data type {sacc_data_type}'%s' is not " f"supported!"
            )

    @final
    def _update(self, params: ParamsMap):
        self.source0.update(params)
        self.source1.update(params)

    @final
    def _reset(self) -> None:
        self.source0.reset()
        self.source1.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        return self.source0.required_parameters() + self.source1.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = derived_parameters + self.source0.get_derived_parameters()
        derived_parameters = derived_parameters + self.source1.get_derived_parameters()
        return derived_parameters

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """

        self.source0.read(sacc_data)
        self.source1.read(sacc_data)

        tracers = [self.source0.sacc_tracer, self.source1.sacc_tracer]

        if self.ccl_kind == "cl":
            _ell_or_theta, _stat = sacc_data.get_ell_cl(
                self.sacc_data_type, *tracers, return_cov=False
            )
        else:
            _ell_or_theta, _stat = sacc_data.get_theta_xi(
                self.sacc_data_type, *tracers, return_cov=False
            )

        if self.ell_or_theta is None and (len(_ell_or_theta) == 0 or len(_stat) == 0):
            raise RuntimeError(
                f"Tracers '{tracers}' for data type '{self.sacc_data_type}' "
                f"have no 2pt data in the SACC file and no input ell or "
                f"theta values were given!"
            )
        if self.ell_or_theta is not None and len(_ell_or_theta) > 0 and len(_stat) > 0:
            warnings.warn(
                f"Tracers '{tracers}' have 2pt data and you have specified "
                "`ell_or_theta` in the configuration. `ell_or_theta` is being ignored!",
                warnings.UserWarning,
                stacklevel=2,
            )

        # at this point we default to the values in the sacc file
        if len(_ell_or_theta) == 0 or len(_stat) == 0:
            _ell_or_theta = _generate_ell_or_theta(**self.ell_or_theta)
            _stat = np.zeros_like(_ell_or_theta)
            self.sacc_inds = None
        else:
            self.sacc_inds = np.atleast_1d(
                sacc_data.indices(self.sacc_data_type, tuple(tracers))
            )

        if self.ell_or_theta_min is not None:
            q = np.where(_ell_or_theta >= self.ell_or_theta_min)
            _ell_or_theta = _ell_or_theta[q]
            _stat = _stat[q]
            if self.sacc_inds is not None:
                self.sacc_inds = self.sacc_inds[q]

        if self.ell_or_theta_max is not None:
            q = np.where(_ell_or_theta <= self.ell_or_theta_max)
            _ell_or_theta = _ell_or_theta[q]
            _stat = _stat[q]
            if self.sacc_inds is not None:
                self.sacc_inds = self.sacc_inds[q]

        self.theory_window_function = sacc_data.get_bandpower_windows(self.sacc_inds)
        if self.theory_window_function is not None:
            ell_config = {**ELL_FOR_XI_DEFAULTS}
            ell_config["maximum"] = self.theory_window_function.values[-1]
            ell_config["minimum"] = max(
                ell_config["minimum"], self.theory_window_function.values[0]
            )
            _ell_or_theta = _ell_for_xi(**ell_config)

        # I don't think we need these copies, but being safe here.
        self._ell_or_theta = _ell_or_theta.copy()
        self.data_vector = _stat.copy()
        self.measured_statistic_ = self.data_vector
        self.sacc_tracers = tracers

    def compute(self, cosmo: CosmologyBundle) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a two-point statistic from sources."""
        self.ell_or_theta_ = self._ell_or_theta.copy()

        tracers0 = self.source0.get_tracers(cosmo)
        tracers1 = self.source1.get_tracers(cosmo)
        scale0 = self.source0.get_scale()
        scale1 = self.source1.get_scale()

        if self.ccl_kind == "cl":
            self.ells = self.ell_or_theta_
        else:
            self.ells = _ell_for_xi(**self.ell_for_xi)
        self.cells = {}

        # Loop over the tracers and compute all possible combinations
        # of them
        for tracer0 in tracers0:
            for tracer1 in tracers1:
                pk_name = f"{tracer0.field}:{tracer1.field}"
                if (tracer0.field, tracer1.field) in self.cells:
                    # Already computed this combination, skipping
                    continue
                if cosmo.has_pk(pk_name):
                    # Use existing power spectrum
                    pk = cosmo.get_pk(pk_name)
                elif tracer0.has_pt and tracer1.has_pt:
                    # Compute perturbation power spectrum
                    pk = pyccl.nl_pt.get_pt_pk2d(cosmo.ccl_cosmo,
                                                 tracer0.pt_tracer, tracer2=tracer1.pt_tracer,
                                                 nonlin_pk_type="nonlinear",
                                                 ptc=cosmo.pt_calculator, update_ptc=False)
                elif tracer0.has_pt or tracer1.has_pt:
                    # Mixed pt and non-pt combination: skip
                    continue
                elif tracer0.has_hm and tracer1.has_hm:
                    # Compute halo model power spectrum
                    raise NotImplementedError("Halo model power spectra not supported yet")
                else:
                    raise ValueError(f"No power spectrum for {pk_name} can be found.")

                self.cells[(tracer0.field, tracer1.field)] = _cached_angular_cl(
                    cosmo.ccl_cosmo, (tracer0.ccl_tracer, tracer1.ccl_tracer), tuple(self.ells.tolist()), p_of_k_a=pk
                ) * scale0 * scale1

        # Add up all the contributions to the cells
        self.cells["total"] = sum(self.cells.values())
        theory_vector = self.cells["total"]

        if not self.ccl_kind == "cl":
            theory_vector = (
                pyccl.correlation(
                    cosmo.ccl_cosmo, self.ells, theory_vector, self.ell_or_theta_ / 60, type=self.ccl_kind
                )
            )

        if self.theory_window_function is not None:

            def log_interpolator(x, y):
                if np.all(y > 0):
                    # use log-log interpolation
                    intp = scipy.interpolate.InterpolatedUnivariateSpline(
                        np.log(x), np.log(y), ext=2
                    )
                    return lambda x_, intp=intp: np.exp(intp(np.log(x_)))
                else:
                    # only use log for x
                    intp = scipy.interpolate.InterpolatedUnivariateSpline(
                        np.log(x), y, ext=2
                    )
                    return lambda x_, intp=intp: intp(np.log(x_))

            theory_interpolator = log_interpolator(self.ell_or_theta_, theory_vector)
            ell = self.theory_window_function.values
            # Deal with ell=0 and ell=1
            theory_vector_interpolated = np.zeros(ell.size)
            theory_vector_interpolated[2:] = theory_interpolator(ell[2:])

            theory_vector = np.einsum(
                "lb, l -> b",
                self.theory_window_function.weight,
                theory_vector_interpolated,
            )
            self.ell_or_theta_ = np.einsum(
                "lb, l -> b", self.theory_window_function.weight, ell
            )

        self.predicted_statistic_ = theory_vector

        assert self.data_vector is not None

        return np.array(self.data_vector), np.array(theory_vector)
