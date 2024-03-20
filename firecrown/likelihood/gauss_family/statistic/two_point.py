"""Two point statistic support.
"""

from __future__ import annotations

import dataclasses
from typing import Optional
import copy
import functools
import warnings

import numpy as np
import numpy.typing as npt
import sacc.windows
import scipy.interpolate

import pyccl
import pyccl.nl_pt

from ....modeling_tools import ModelingTools

from .statistic import Statistic, DataVector, TheoryVector
from .source.source import Source, Tracer

# only supported types are here, anything else will throw
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

ELL_FOR_XI_DEFAULTS = {"minimum": 2, "midpoint": 50, "maximum": 60_000, "n_log": 200}


def _ell_for_xi(
    *, minimum: int, midpoint: int, maximum: int, n_log: int
) -> npt.NDArray[np.float64]:
    """Build an array of ells to sample the power spectrum for real-space
    predictions.

    The result will contain each integral value from min to mid.
    Starting from mid, and going up to max, there will be n_log
    logarithmically spaced values. All values are rounded to the nearest
    integer.
    """
    assert minimum >= 0
    assert minimum < midpoint
    assert midpoint < maximum
    lower_range = np.linspace(minimum, midpoint - 1, midpoint - minimum)
    upper_range = np.logspace(np.log10(midpoint), np.log10(maximum), n_log)
    concatenated = np.concatenate((lower_range, upper_range))
    # Round the results to the nearest integer values.
    # N.B. the dtype of the result is np.dtype[float64]
    return np.unique(np.around(concatenated))


def _generate_ell_or_theta(*, minimum, maximum, n, binning="log"):
    if binning == "log":
        edges = np.logspace(np.log10(minimum), np.log10(maximum), n + 1)
        return np.sqrt(edges[1:] * edges[:-1])
    edges = np.linspace(minimum, maximum, n + 1)
    return (edges[1:] + edges[:-1]) / 2.0


@functools.lru_cache(maxsize=128)
def _cached_angular_cl(cosmo, tracers, ells, p_of_k_a=None):
    return pyccl.angular_cl(
        cosmo, tracers[0], tracers[1], np.array(ells), p_of_k_a=p_of_k_a
    )


def make_log_interpolator(x, y):
    """Return a function object that does 1D spline interpolation.

    If all the y values are greater than 0, the function
    interpolates log(y) as a function of log(x).
    Otherwise, the function interpolates y as a function of log(x).
    The resulting interpolater will not extrapolate; if called with
    an out-of-range argument it will raise a ValueError.
    """
    # TODO: There is no code in Firecrown, neither test nor example, that uses
    # this in any way.
    if np.all(y > 0):
        # use log-log interpolation
        intp = scipy.interpolate.InterpolatedUnivariateSpline(
            np.log(x), np.log(y), ext=2
        )
        return lambda x_, intp=intp: np.exp(intp(np.log(x_)))
    # only use log for x
    intp = scipy.interpolate.InterpolatedUnivariateSpline(np.log(x), y, ext=2)
    return lambda x_, intp=intp: intp(np.log(x_))


@dataclasses.dataclass(frozen=True)
class TracerNames:
    """The names of the two tracers in the sacc file."""

    name1: str
    name2: str

    def __getitem__(self, item):
        """Get the name of the tracer at the given index."""
        if item == 0:
            return self.name1
        if item == 1:
            return self.name2
        raise IndexError

    def __iter__(self):
        """Iterate through the data members. This is to allow automatic
        unpacking."""
        yield self.name1
        yield self.name2


TRACER_NAMES_TOTAL = TracerNames("", "")  # special name to represent total


def read_ell_or_theta_and_stat(
    ccl_kind: str, sacc_data_type: str, sacc_data: sacc.Sacc, tracers: TracerNames
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read either ell_cl or theta_xi data from sacc_data, and
    return that and associated stat data.
    """
    method = sacc_data.get_ell_cl if ccl_kind == "cl" else sacc_data.get_theta_xi
    ell_or_theta, stat = method(sacc_data_type, *tracers, return_cov=False)
    assert len(ell_or_theta) == len(stat)
    return ell_or_theta, stat


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

         - minimum : int, optional - The minimum angular wavenumber to use for
           real-space integrations. Default is 2.
         - midpoint : int, optional - The midpoint angular wavenumber to use
           for real-space integrations. The angular wavenumber samples are
           linearly spaced at integers between `minimum` and `midpoint`. Default
           is 50.
         - maximum : int, optional - The maximum angular wavenumber to use for
           real-space integrations. The angular wavenumber samples are
           logarithmically spaced between `midpoint` and `maximum`. Default is
           60,000.
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
    ell_or_theta_ : npt.NDArray[np.float64]
        The final array of ell/theta values for the statistic. Set after
        `compute` is called.
    measured_statistic_ : npt.NDArray[np.float64]
        The measured value for the statistic.
    predicted_statistic_ : npt.NDArray[np.float64]
        The final prediction for the statistic. Set after `compute` is called.

    """

    def __init__(
        self,
        sacc_data_type,
        source0: Source,
        source1: Source,
        *,
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
        self.ell_for_xi: dict[str, int] = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            self.ell_for_xi.update(ell_for_xi)
        # What is the difference between the following 3 instance variables?
        #        ell_or_theta
        #        _ell_or_theta
        #        ell_or_theta_
        self.ell_or_theta = ell_or_theta
        self.ell_or_theta_min = ell_or_theta_min
        self.ell_or_theta_max = ell_or_theta_max
        self.theory_window_function: Optional[sacc.windows.BandpowerWindow] = None

        self.data_vector: Optional[DataVector] = None
        self.theory_vector: Optional[TheoryVector] = None
        self._ell_or_theta: Optional[npt.NDArray[np.float64]] = None
        self.ell_or_theta_: Optional[npt.NDArray[np.float64]] = None

        self.sacc_tracers: TracerNames
        self.ells: Optional[npt.NDArray[np.float64]] = None
        self.cells: dict[TracerNames, npt.NDArray[np.float64]] = {}
        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(
                f"The SACC data type {sacc_data_type}'%s' is not " f"supported!"
            )

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file.

        :param sacc_data: The data in the sacc format.
        """

        tracers = self.initialize_sources(sacc_data)

        _ell_or_theta, _stat = read_ell_or_theta_and_stat(
            self.ccl_kind, self.sacc_data_type, sacc_data, tracers
        )

        # If we have no data from our construction, and the SACC object also contains
        # no data, we have a failure...
        if self.ell_or_theta is None and (len(_ell_or_theta) == 0 or len(_stat) == 0):
            raise RuntimeError(
                f"Tracers '{tracers}' for data type '{self.sacc_data_type}' "
                f"have no 2pt data in the SACC file and no input ell or "
                f"theta values were given!"
            )
        # If we have data from our construction, and also have data in the SACC object,
        # emit a warning and use the information read from the SACC object.
        if self.ell_or_theta is not None and len(_ell_or_theta) > 0 and len(_stat) > 0:
            warnings.warn(
                f"Tracers '{tracers}' have 2pt data and you have specified "
                "`ell_or_theta` in the configuration. `ell_or_theta` is being ignored!",
                stacklevel=2,
            )

        # at this point we default to the values in the sacc file
        _ell_or_theta, _stat = self._calculate_stat_stuff(
            _ell_or_theta, _stat, sacc_data, tracers
        )

        # I don't think we need these copies, but being safe here.
        self._ell_or_theta = _ell_or_theta.copy()
        self.data_vector = DataVector.create(_stat)
        self.data_vector = self.data_vector
        self.sacc_tracers = tracers

        super().read(sacc_data)

    # TODO: Inline this function after it has been refactored.
    def _calculate_stat_stuff(
        self,
        ell_or_theta: npt.NDArray[np.float64],
        stat: npt.NDArray[np.float64],
        sacc_data: sacc.Sacc,
        tracers: TracerNames,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert len(ell_or_theta) == len(stat)
        common_length = len(ell_or_theta)
        ell_or_theta, stat = self.phase_1(
            sacc_data, tracers, common_length, ell_or_theta, stat
        )
        ell_or_theta = self.set_window_function(ell_or_theta, sacc_data)
        assert len(ell_or_theta) == len(stat)
        return ell_or_theta, stat

    def phase_1(
        self, sacc_data, tracers: TracerNames, common_length, ell_or_theta, stat
    ):
        """Temporary method to support refactoring of TwoPoint.read."""
        # Depending on the value of common_length, calculate either:
        #    1) ell_or_theta and stat, or
        #    2) self.sacc_indices
        if common_length == 0:
            ell_or_theta = _generate_ell_or_theta(**self.ell_or_theta)
            stat = np.zeros_like(ell_or_theta)
        else:
            self.sacc_indices = np.atleast_1d(
                sacc_data.indices(self.sacc_data_type, tracers)
            )
            assert self.sacc_indices is not None  # Needed for mypy
            assert len(self.sacc_indices) == common_length

        # If we have set self.ell_or_theta_min, filter ell_or_theta, stat, and
        # possibly self.sacc_indices
        if self.ell_or_theta_min is not None:
            locations = np.where(ell_or_theta >= self.ell_or_theta_min)
            ell_or_theta = ell_or_theta[locations]
            stat = stat[locations]
            if self.sacc_indices is not None:
                self.sacc_indices = self.sacc_indices[locations]

        # If we have set self.ell_or_theta_max, filter ell_or_theta, stat, and
        # possibly self.sacc_indices
        if self.ell_or_theta_max is not None:
            locations = np.where(ell_or_theta <= self.ell_or_theta_max)
            ell_or_theta = ell_or_theta[locations]
            stat = stat[locations]
            if self.sacc_indices is not None:
                self.sacc_indices = self.sacc_indices[locations]
        return ell_or_theta, stat

    def set_window_function(
        self, ell_or_theta: npt.NDArray[np.float64], sacc_data: sacc.Sacc
    ) -> npt.NDArray[np.float64]:
        """Set the window function for this statistic."""
        assert self.sacc_indices is not None
        self.theory_window_function = sacc_data.get_bandpower_windows(self.sacc_indices)
        if self.theory_window_function is not None:
            ell_or_theta = self.calculate_ell_or_theta()
            # Normalise the weights to 1:
            norm = self.theory_window_function.weight.sum(axis=0)
            self.theory_window_function.weight /= norm
        return ell_or_theta

    def initialize_sources(self, sacc_data: sacc.Sacc) -> TracerNames:
        """Initialize this TwoPoint's sources, and return the tracer names."""
        self.source0.read(sacc_data)
        if self.source0 is not self.source1:
            self.source1.read(sacc_data)
        assert self.source0.sacc_tracer is not None
        assert self.source1.sacc_tracer is not None
        tracers = (self.source0.sacc_tracer, self.source1.sacc_tracer)
        return TracerNames(*tracers)

    def calculate_ell_or_theta(self) -> npt.NDArray[np.float64]:
        """See _ell_for_xi.

        This method mixes together:
           1. the default parameters in ELL_FOR_XI_DEFAULTS
           2. the first and last values in self.theory_window_function.values
        and then calls _ell_for_xi with those arguments, returning whatever it
        returns.

        It is an error to call this function if self.theory_window_function has
        not been set. That is done in `read`, but might result in the value
        being re-set to None.:w
        """
        assert self.theory_window_function is not None
        ell_config = {
            **ELL_FOR_XI_DEFAULTS,
            "maximum": self.theory_window_function.values[-1],
        }
        ell_config["minimum"] = max(
            ell_config["minimum"], self.theory_window_function.values[0]
        )
        return _ell_for_xi(**ell_config)

    def get_data_vector(self) -> DataVector:
        """Return this statistic's data vector."""
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a two-point statistic from sources."""

        assert self._ell_or_theta is not None
        self.ell_or_theta_ = self._ell_or_theta.copy()

        tracers0 = self.source0.get_tracers(tools)
        tracers1 = self.source1.get_tracers(tools)
        scale0 = self.source0.get_scale()
        scale1 = self.source1.get_scale()

        if self.ccl_kind == "cl":
            self.ells = self.ell_or_theta_
        else:
            self.ells = _ell_for_xi(
                minimum=int(self.ell_for_xi["minimum"]),
                midpoint=int(self.ell_for_xi["midpoint"]),
                maximum=int(self.ell_for_xi["maximum"]),
                n_log=int(self.ell_for_xi["n_log"]),
            )

        # TODO: we should not be adding a new instance variable outside of
        # __init__. Why is `self.cells` an instance variable rather than a
        # local variable? It is used in at least two of the example codes:
        # both the PT and the TATT examples in des_y1_3x2pt access this data
        # member to print out results when the likelihood is "run directly"
        # by calling `run_likelihood`.

        self.cells = {}

        # Loop over the tracers and compute all possible combinations
        # of them.
        for tracer0 in tracers0:
            for tracer1 in tracers1:
                pk_name = f"{tracer0.field}:{tracer1.field}"
                tn = TracerNames(tracer0.tracer_name, tracer1.tracer_name)
                if tn in self.cells:
                    # Already computed this combination, skipping
                    continue
                pk = self.calculate_pk(pk_name, tools, tracer0, tracer1)

                self.cells[tn] = (
                    _cached_angular_cl(
                        tools.get_ccl_cosmology(),
                        (tracer0.ccl_tracer, tracer1.ccl_tracer),
                        tuple(self.ells.tolist()),
                        p_of_k_a=pk,
                    )
                    * scale0
                    * scale1
                )

        # Add up all the contributions to the cells
        self.cells[TRACER_NAMES_TOTAL] = np.array(sum(self.cells.values()))
        theory_vector = self.cells[TRACER_NAMES_TOTAL]

        if not self.ccl_kind == "cl":
            theory_vector = pyccl.correlation(
                tools.get_ccl_cosmology(),
                ell=self.ells,
                C_ell=theory_vector,
                theta=self.ell_or_theta_ / 60,
                type=self.ccl_kind,
            )

        if self.theory_window_function is not None:

            if not self.ccl_kind == "cl":
                raise ValueError("You cannot mix theory window function with xi.")

            # TODO: There is no code in Firecrown, neither test nor example,
            # that exercises a theory window function in any way.
            theory_interpolator = make_log_interpolator(
                self.ell_or_theta_, theory_vector
            )
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

        assert self.data_vector is not None
        return TheoryVector.create(theory_vector)

    def calculate_pk(
        self, pk_name: str, tools: ModelingTools, tracer0: Tracer, tracer1: Tracer
    ):
        """Return the power spectrum named by pk_name."""
        if tools.has_pk(pk_name):
            # Use existing power spectrum
            pk = tools.get_pk(pk_name)
        elif tracer0.has_pt or tracer1.has_pt:
            if not (tracer0.has_pt and tracer1.has_pt):
                # Mixture of PT and non-PT tracers
                # Create a dummy matter PT tracer for the non-PT part
                # FIXME: What if we are doing GGL, and need galaxies as tracers?
                matter_pt_tracer = pyccl.nl_pt.PTMatterTracer()
                if not tracer0.has_pt:
                    tracer0.pt_tracer = matter_pt_tracer
                else:
                    tracer1.pt_tracer = matter_pt_tracer
            # Compute perturbation power spectrum

            pt_calculator = tools.get_pt_calculator()
            pk = pt_calculator.get_biased_pk2d(
                tracer1=tracer0.pt_tracer,
                tracer2=tracer1.pt_tracer,
            )
        elif tracer0.has_hm or tracer1.has_hm:
            # Compute halo model power spectrum
            a_arr = np.linspace(0.1, 1, 16)  # Fix a_arr because normalization is zero for a<~0.07
            ccl_cosmo = tools.get_ccl_cosmology()
            hmc = tools.get_hm_calculator()
            IA_bias_exponent = 2  # Square IA bias if both tracers are HM (doing II correlation).
            if not (tracer0.has_hm and tracer1.has_hm):
                IA_bias_exponent = 1  # IA bias if not both tracers are HM (doing GI correlation).
                if 'galaxies' in [tracer0.field, tracer1.field]:
                    other_profile = pyccl.halos.HaloProfileHOD(mass_def=tools.hm_definition,
                                                               concentration=tools.get_cM_relation(),
                                                               truncated=True, fourier_analytic=True)
                else:
                    other_profile = pyccl.halos.HaloProfileNFW(mass_def=tools.hm_definition,
                                                               concentration=tools.get_cM_relation(),
                                                               truncated=True, fourier_analytic=True)
                other_profile.ia_a_2h = -1.  # used in GI contribution, which is negative.
                if not tracer0.has_hm:
                    profile0 = other_profile
                    profile1 = tracer1.halo_profile
                else:
                    profile0 = tracer0.halo_profile
                    profile1 = other_profile
            else:
                profile0 = tracer0.halo_profile
                profile1 = tracer1.halo_profile
            # Compute here the 1-halo power spectrum
            pk_1h = pyccl.halos.halomod_Pk2D(cosmo=ccl_cosmo, hmc=hmc,
                                             prof=profile0, prof2=profile1,
                                             a_arr=a_arr, get_2h=False)
            # Compute here the 2-halo power spectrum
            C1rhocrit = 5e-14 * pyccl.physical_constants.RHO_CRITICAL # standard IA normalisation
            pk_2h = pyccl.Pk2D.from_function(
                pkfunc=lambda k, a: profile0.ia_a_2h*profile1.ia_a_2h*
                                    (C1rhocrit*ccl_cosmo['Omega_m']/ccl_cosmo.growth_factor(a))
                                    **IA_bias_exponent*
                                    ccl_cosmo.nonlin_matter_power(k, a),
                is_logp=False)
            pk = pk_1h + pk_2h
        else:
            raise ValueError(f"No power spectrum for {pk_name} can be found.")
        return pk
