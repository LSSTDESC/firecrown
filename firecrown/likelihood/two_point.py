"""Two point statistic support."""

from __future__ import annotations

import copy
import warnings
from typing import Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc.windows
import scipy.interpolate

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown.generators.two_point import LogLinearElls
from firecrown.likelihood.source import Source, Tracer
from firecrown.likelihood.weak_lensing import (
    WeakLensingFactory,
    WeakLensing,
)
from firecrown.likelihood.number_counts import (
    NumberCountsFactory,
    NumberCounts,
)
from firecrown.likelihood.statistic import (
    DataVector,
    Statistic,
    TheoryVector,
)
from firecrown.metadata.two_point_types import (
    TRACER_NAMES_TOTAL,
    InferredGalaxyZDist,
    Galaxies,
    Measurement,
)

from firecrown.metadata.two_point import (
    TracerNames,
    TwoPointCells,
    TwoPointCWindow,
    TwoPointXiTheta,
    TwoPointCellsIndex,
    TwoPointXiThetaIndex,
    Window,
    extract_window_function,
    check_two_point_consistence_harmonic,
    check_two_point_consistence_real,
    measurements_from_index,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import UpdatableCollection

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
) -> npt.NDArray[np.int64]:
    """Create an array of ells to sample the power spectrum.

    This is used for for real-space predictions. The result will contain
    each integral value from min to mid. Starting from mid, and going up
    to max, there will be n_log logarithmically spaced values.

    All values are rounded to the nearest integer.
    """
    return LogLinearElls(
        minimum=minimum, midpoint=midpoint, maximum=maximum, n_log=n_log
    ).generate()


def _generate_ell_or_theta(*, minimum, maximum, n, binning="log"):
    if binning == "log":
        edges = np.logspace(np.log10(minimum), np.log10(maximum), n + 1)
        return np.sqrt(edges[1:] * edges[:-1])
    edges = np.linspace(minimum, maximum, n + 1)
    return (edges[1:] + edges[:-1]) / 2.0


# @functools.lru_cache(maxsize=128)
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


def calculate_ells_for_interpolation(w: Window) -> npt.NDArray[np.int64]:
    """See _ell_for_xi.

    This method mixes together:
        1. the default parameters in ELL_FOR_XI_DEFAULTS
        2. the first and last values in w.

    and then calls _ell_for_xi with those arguments, returning whatever it
    returns.
    """
    ell_config = {
        **ELL_FOR_XI_DEFAULTS,
        "maximum": w.ells[-1],
    }
    ell_config["minimum"] = max(ell_config["minimum"], w.ells[0])
    return _ell_for_xi(**ell_config)


class EllOrThetaConfig(TypedDict):
    """A dictionary of options for generating the ell or theta.

    This dictionary contains the minimum, maximum and number of
    bins to generate the ell or theta values at which to compute the statistics.

    :param minimum: The start of the binning.
    :param maximum: The end of the binning.
    :param n: The number of bins.
    :param binning: Pass 'log' to get logarithmic spaced bins and 'lin' to get linearly
        spaced bins. Default is 'log'.

    """

    minimum: float
    maximum: float
    n: int
    binning: str


def generate_ells_cells(ell_config: EllOrThetaConfig):
    """Generate ells or theta values from the configuration dictionary."""
    ells = _generate_ell_or_theta(**ell_config)
    Cells = np.zeros_like(ells)

    return ells, Cells


def generate_theta_xis(theta_config: EllOrThetaConfig):
    """Generate theta and xi values from the configuration dictionary."""
    thetas = _generate_ell_or_theta(**theta_config)
    xis = np.zeros_like(thetas)

    return thetas, xis


def apply_ells_min_max(
    ells: npt.NDArray[np.int64],
    Cells: npt.NDArray[np.float64],
    indices: None | npt.NDArray[np.int64],
    ell_min: None | int,
    ell_max: None | int,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.float64], None | npt.NDArray[np.int64]
]:
    """Apply the minimum and maximum ell values to the ells and Cells."""
    if ell_min is not None:
        locations = np.where(ells >= ell_min)
        ells = ells[locations]
        Cells = Cells[locations]
        if indices is not None:
            indices = indices[locations]

    if ell_max is not None:
        locations = np.where(ells <= ell_max)
        ells = ells[locations]
        Cells = Cells[locations]
        if indices is not None:
            indices = indices[locations]

    return ells, Cells, indices


def apply_theta_min_max(
    thetas: npt.NDArray[np.float64],
    xis: npt.NDArray[np.float64],
    indices: None | npt.NDArray[np.int64],
    theta_min: None | float,
    theta_max: None | float,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], None | npt.NDArray[np.int64]
]:
    """Apply the minimum and maximum theta values to the thetas and xis."""
    if theta_min is not None:
        locations = np.where(thetas >= theta_min)
        thetas = thetas[locations]
        xis = xis[locations]
        if indices is not None:
            indices = indices[locations]

    if theta_max is not None:
        locations = np.where(thetas <= theta_max)
        thetas = thetas[locations]
        xis = xis[locations]
        if indices is not None:
            indices = indices[locations]

    return thetas, xis, indices


def use_source_factory(
    inferred_galaxy_zdist: InferredGalaxyZDist,
    measurement: Measurement,
    wl_factory: WeakLensingFactory | None = None,
    nc_factory: NumberCountsFactory | None = None,
) -> WeakLensing | NumberCounts:
    """Apply the factory to the inferred galaxy redshift distribution."""
    source: WeakLensing | NumberCounts
    if measurement not in inferred_galaxy_zdist.measurements:
        raise ValueError(
            f"Measurement {measurement} not found in inferred galaxy redshift "
            f"distribution {inferred_galaxy_zdist.bin_name}!"
        )

    match measurement:
        case Galaxies.COUNTS:
            assert nc_factory is not None
            source = nc_factory.create(inferred_galaxy_zdist)
        case (
            Galaxies.SHEAR_E
            | Galaxies.SHEAR_T
            | Galaxies.SHEAR_MINUS
            | Galaxies.SHEAR_PLUS
        ):
            assert wl_factory is not None
            source = wl_factory.create(inferred_galaxy_zdist)
        case _:
            raise ValueError(f"Measurement {measurement} not supported!")
    return source


def use_source_factory_metadata_only(
    sacc_tracer: str,
    measurement: Measurement,
    wl_factory: WeakLensingFactory | None = None,
    nc_factory: NumberCountsFactory | None = None,
) -> WeakLensing | NumberCounts:
    """Apply the factory to create a source from metadata only."""
    source: WeakLensing | NumberCounts
    match measurement:
        case Galaxies.COUNTS:
            assert nc_factory is not None
            source = nc_factory.create_from_metadata_only(sacc_tracer)
        case (
            Galaxies.SHEAR_E
            | Galaxies.SHEAR_T
            | Galaxies.SHEAR_MINUS
            | Galaxies.SHEAR_PLUS
        ):
            assert wl_factory is not None
            source = wl_factory.create_from_metadata_only(sacc_tracer)
        case _:
            raise ValueError(f"Measurement {measurement} not supported!")
    return source


class TwoPoint(Statistic):
    """A two-point statistic.

    For example, shear correlation function, galaxy-shear correlation function, etc.

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

         - minimun : float - The start of the binning.
         - maximun : float - The end of the binning.
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
        sacc_data_type: str,
        source0: Source,
        source1: Source,
        *,
        ell_for_xi: None | dict[str, int] = None,
        ell_or_theta: None | EllOrThetaConfig = None,
        ell_or_theta_min: None | float | int = None,
        ell_or_theta_max: None | float | int = None,
    ) -> None:
        super().__init__()

        assert isinstance(source0, Source)
        assert isinstance(source1, Source)

        self.sacc_data_type: str
        self.ccl_kind: str
        self.source0: Source = source0
        self.source1: Source = source1

        self.ell_for_xi_config: dict[str, int]
        self.ell_or_theta_config: None | EllOrThetaConfig
        self.ell_or_theta_min: None | float | int
        self.ell_or_theta_max: None | float | int
        self.window: None | Window
        self.data_vector: None | DataVector
        self.theory_vector: None | TheoryVector
        self.sacc_tracers: TracerNames
        self.ells: None | npt.NDArray[np.int64]
        self.thetas: None | npt.NDArray[np.float64]
        self.mean_ells: None | npt.NDArray[np.float64]
        self.ells_for_xi: None | npt.NDArray[np.int64]
        self.cells: dict[TracerNames, npt.NDArray[np.float64]]

        self._init_empty_default_attribs()
        if ell_for_xi is not None:
            self.ell_for_xi_config.update(ell_for_xi)
        self.ell_or_theta_config = ell_or_theta
        self.ell_or_theta_min = ell_or_theta_min
        self.ell_or_theta_max = ell_or_theta_max

        self._set_ccl_kind(sacc_data_type)

    def _init_empty_default_attribs(self):
        """Initialize the empty and default attributes."""
        self.ell_for_xi_config = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        self.ell_or_theta_config = None
        self.ell_or_theta_min = None
        self.ell_or_theta_max = None

        self.window = None

        self.data_vector = None
        self.theory_vector = None

        self.ells = None
        self.thetas = None
        self.mean_ells = None
        self.ells_for_xi = None

        self.cells = {}

    def _set_ccl_kind(self, sacc_data_type):
        """Set the CCL kind for this statistic."""
        self.sacc_data_type = sacc_data_type
        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(f"The SACC data type {sacc_data_type} is not supported!")

    @classmethod
    def _from_metadata(
        cls,
        *,
        sacc_data_type: str,
        source0: Source,
        source1: Source,
        metadata: TwoPointCells | TwoPointCWindow | TwoPointXiTheta,
    ) -> TwoPoint:
        """Create a TwoPoint statistic from a TwoPointCells metadata object."""
        two_point = cls(sacc_data_type, source0, source1)
        match metadata:
            case TwoPointCells():
                two_point._init_from_cells(metadata)
            case TwoPointCWindow():
                two_point._init_from_cwindow(metadata)
            case TwoPointXiTheta():
                two_point._init_from_xi_theta(metadata)
            case _:
                raise ValueError(f"Metadata of type {type(metadata)} is not supported!")
        return two_point

    def _init_from_cells(self, metadata: TwoPointCells):
        """Initialize the TwoPoint statistic from a TwoPointCells metadata object."""
        self.sacc_tracers = metadata.XY.get_tracer_names()
        self.ells = metadata.ells
        self.window = None
        if metadata.Cell is not None:
            self.sacc_indices = metadata.Cell.indices
            self.data_vector = DataVector.create(metadata.Cell.data)
        self.ready = True

    def _init_from_cwindow(self, metadata: TwoPointCWindow):
        """Initialize the TwoPoint statistic from a TwoPointCWindow metadata object."""
        self.sacc_tracers = metadata.XY.get_tracer_names()
        self.window = metadata.window
        if self.window.ells_for_interpolation is None:
            self.window.ells_for_interpolation = calculate_ells_for_interpolation(
                self.window
            )
        if metadata.Cell is not None:
            self.sacc_indices = metadata.Cell.indices
            self.data_vector = DataVector.create(metadata.Cell.data)
        self.ready = True

    def _init_from_xi_theta(self, metadata: TwoPointXiTheta):
        """Initialize the TwoPoint statistic from a TwoPointXiTheta metadata object."""
        self.sacc_tracers = metadata.XY.get_tracer_names()
        self.thetas = metadata.thetas
        self.window = None
        self.ells_for_xi = _ell_for_xi(**self.ell_for_xi_config)
        if metadata.xis is not None:
            self.sacc_indices = metadata.xis.indices
            self.data_vector = DataVector.create(metadata.xis.data)
        self.ready = True

    @classmethod
    def _from_metadata_any(
        cls,
        metadata: Sequence[TwoPointCells | TwoPointCWindow | TwoPointXiTheta],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of TwoPoint statistics.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointCells, TwoPointCWindow or TwoPointXiTheta metadata objects.
        The metadata objects are used to initialize the TwoPoint statistics.
        """
        two_point_list = [
            cls._from_metadata(
                sacc_data_type=cell.get_sacc_name(),
                source0=use_source_factory(
                    cell.XY.x,
                    cell.XY.x_measurement,
                    wl_factory=wl_factory,
                    nc_factory=nc_factory,
                ),
                source1=use_source_factory(
                    cell.XY.y,
                    cell.XY.y_measurement,
                    wl_factory=wl_factory,
                    nc_factory=nc_factory,
                ),
                metadata=cell,
            )
            for cell in metadata
        ]

        return UpdatableCollection(two_point_list)

    @classmethod
    def _from_metadata_only_any(
        cls,
        metadata: Sequence[TwoPointCellsIndex | TwoPointXiThetaIndex],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of TwoPoint statistics.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointCells, TwoPointCWindow or TwoPointXiTheta metadata objects.
        The metadata objects are used to initialize the TwoPoint statistics.
        """
        two_point_list = []
        for cell_index in metadata:
            n1, a, n2, b = measurements_from_index(cell_index)
            two_point = cls(
                sacc_data_type=cell_index["data_type"],
                source0=use_source_factory_metadata_only(
                    n1, a, wl_factory=wl_factory, nc_factory=nc_factory
                ),
                source1=use_source_factory_metadata_only(
                    n2, b, wl_factory=wl_factory, nc_factory=nc_factory
                ),
            )
            two_point_list.append(two_point)

        return UpdatableCollection(two_point_list)

    @classmethod
    def from_metadata_harmonic(
        cls,
        metadata: Sequence[TwoPointCells | TwoPointCWindow],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
        check_consistence: bool = False,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of harmonic space TwoPoint statistics.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointCells or TwoPointCWindow metadata objects. The metadata objects
        are used to initialize the TwoPoint statistics.

        :param metadata: The metadata objects to initialize the TwoPoint statistics.
        :param wl_factory: The weak lensing factory to use.
        :param nc_factory: The number counts factory to use.
        :param check_consistence: Whether to check the consistence of the
            metadata and data.
        """
        if check_consistence:
            check_two_point_consistence_harmonic(metadata)
        return cls._from_metadata_any(metadata, wl_factory, nc_factory)

    @classmethod
    def from_metadata_real(
        cls,
        metadata: Sequence[TwoPointXiTheta],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
        check_consistence: bool = False,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of real space TwoPoint statistics.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointXiTheta metadata objects. The metadata objects are used to
        initialize the TwoPoint statistics.

        :param metadata: The metadata objects to initialize the TwoPoint statistics.
        :param wl_factory: The weak lensing factory to use.
        :param nc_factory: The number counts factory to use.
        :param check_consistence: Whether to check the consistence of the
            metadata and data.
        """
        if check_consistence:
            check_two_point_consistence_real(metadata)
        return cls._from_metadata_any(metadata, wl_factory, nc_factory)

    @classmethod
    def from_metadata_only_harmonic(
        cls,
        metadata: list[TwoPointCellsIndex],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
    ) -> UpdatableCollection[TwoPoint]:
        """Create a TwoPoint from metadata only."""
        return cls._from_metadata_only_any(metadata, wl_factory, nc_factory)

    @classmethod
    def from_metadata_only_real(
        cls,
        metadata: list[TwoPointXiThetaIndex],
        wl_factory: WeakLensingFactory | None = None,
        nc_factory: NumberCountsFactory | None = None,
    ) -> UpdatableCollection[TwoPoint]:
        """Create a TwoPoint from metadata only."""
        return cls._from_metadata_only_any(metadata, wl_factory, nc_factory)

    def read_ell_cells(
        self, sacc_data_type: str, sacc_data: sacc.Sacc, tracers: TracerNames
    ) -> (
        None
        | tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
    ):
        """Read and return ell and Cell."""
        ells, Cells = sacc_data.get_ell_cl(sacc_data_type, *tracers, return_cov=False)
        # As version 0.13 of sacc, the method get_ell_cl returns the
        # ell values and the Cl values in arrays of the same length.
        assert len(ells) == len(Cells)
        common_length = len(ells)
        sacc_indices = None

        if common_length == 0:
            return None
        sacc_indices = np.atleast_1d(sacc_data.indices(self.sacc_data_type, tracers))
        assert sacc_indices is not None  # Needed for mypy
        assert len(sacc_indices) == common_length

        return ells, Cells, sacc_indices

    def read_theta_xis(
        self, sacc_data_type: str, sacc_data: sacc.Sacc, tracers: TracerNames
    ) -> (
        None
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
    ):
        """Read and return theta and xi."""
        thetas, xis = sacc_data.get_theta_xi(sacc_data_type, *tracers, return_cov=False)
        # As version 0.13 of sacc, the method get_theta_xi returns the
        # theta values and the xi values in arrays of the same length.
        assert len(thetas) == len(xis)

        common_length = len(thetas)
        if common_length == 0:
            return None
        sacc_indices = np.atleast_1d(sacc_data.indices(self.sacc_data_type, tracers))
        assert sacc_indices is not None  # Needed for mypy
        assert len(sacc_indices) == common_length
        return thetas, xis, sacc_indices

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file.

        :param sacc_data: The data in the sacc format.
        """
        self.sacc_tracers = self.initialize_sources(sacc_data)

        if self.ccl_kind == "cl":
            self.read_harmonic_space(sacc_data)
        else:
            self.read_real_space(sacc_data)

        super().read(sacc_data)

    def read_real_space(self, sacc_data: sacc.Sacc):
        """Read the data for this statistic from the SACC file."""
        thetas_xis_indices = self.read_theta_xis(
            self.sacc_data_type, sacc_data, self.sacc_tracers
        )
        # We do not support window functions for real space statistics
        if thetas_xis_indices is not None:
            thetas, xis, sacc_indices = thetas_xis_indices
            if self.ell_or_theta_config is not None:
                # If we have data from our construction, and also have data in the
                # SACC object, emit a warning and use the information read from the
                # SACC object.
                warnings.warn(
                    f"Tracers '{self.sacc_tracers}' have 2pt data and you have "
                    f"specified `theta` in the configuration. `theta` is being "
                    f"ignored!",
                    stacklevel=2,
                )
        else:
            if self.ell_or_theta_config is None:
                # The SACC file has no data points, just a tracer, in this case we
                # are building the statistic from scratch. In this case the user
                # must have set the dictionary ell_or_theta, containing the
                # minimum, maximum and number of bins to generate the ell values.
                raise RuntimeError(
                    f"Tracers '{self.sacc_tracers}' for data type "
                    f"'{self.sacc_data_type}' "
                    "have no 2pt data in the SACC file and no input theta values "
                    "were given!"
                )
            thetas, xis = generate_theta_xis(self.ell_or_theta_config)
            sacc_indices = None
        assert isinstance(self.ell_or_theta_min, (float, type(None)))
        assert isinstance(self.ell_or_theta_max, (float, type(None)))
        thetas, xis, sacc_indices = apply_theta_min_max(
            thetas, xis, sacc_indices, self.ell_or_theta_min, self.ell_or_theta_max
        )
        self.ells_for_xi = _ell_for_xi(**self.ell_for_xi_config)
        self.thetas = thetas
        self.sacc_indices = sacc_indices
        self.data_vector = DataVector.create(xis)

    def read_harmonic_space(self, sacc_data: sacc.Sacc):
        """Read the data for this statistic from the SACC file."""
        ells_cells_indices = self.read_ell_cells(
            self.sacc_data_type, sacc_data, self.sacc_tracers
        )
        if ells_cells_indices is not None:
            ells, Cells, sacc_indices = ells_cells_indices
            if self.ell_or_theta_config is not None:
                # If we have data from our construction, and also have data in the
                # SACC object, emit a warning and use the information read from the
                # SACC object.
                warnings.warn(
                    f"Tracers '{self.sacc_tracers}' have 2pt data and you have "
                    f"specified `ell` in the configuration. `ell` is being ignored!",
                    stacklevel=2,
                )
            window = extract_window_function(sacc_data, sacc_indices)
            if window is not None:
                # When using a window function, we do not calculate all Cl's.
                # For this reason we have a default set of ells that we use
                # to compute Cl's, and we have a set of ells used for
                # interpolation.
                window.ells_for_interpolation = calculate_ells_for_interpolation(window)

        else:
            if self.ell_or_theta_config is None:
                # The SACC file has no data points, just a tracer, in this case we
                # are building the statistic from scratch. In this case the user
                # must have set the dictionary ell_or_theta, containing the
                # minimum, maximum and number of bins to generate the ell values.
                raise RuntimeError(
                    f"Tracers '{self.sacc_tracers}' for data type "
                    f"'{self.sacc_data_type}' "
                    "have no 2pt data in the SACC file and no input ell values "
                    "were given!"
                )
            ells, Cells = generate_ells_cells(self.ell_or_theta_config)
            sacc_indices = None

            # When generating the ells and Cells we do not have a window function
            window = None
        assert isinstance(self.ell_or_theta_min, (int, type(None)))
        assert isinstance(self.ell_or_theta_max, (int, type(None)))
        ells, Cells, sacc_indices = apply_ells_min_max(
            ells, Cells, sacc_indices, self.ell_or_theta_min, self.ell_or_theta_max
        )
        self.ells = ells
        self.window = window
        self.sacc_indices = sacc_indices
        self.data_vector = DataVector.create(Cells)

    def initialize_sources(self, sacc_data: sacc.Sacc) -> TracerNames:
        """Initialize this TwoPoint's sources, and return the tracer names."""
        self.source0.read(sacc_data)
        if self.source0 is not self.source1:
            self.source1.read(sacc_data)
        assert self.source0.sacc_tracer is not None
        assert self.source1.sacc_tracer is not None
        tracers = (self.source0.sacc_tracer, self.source1.sacc_tracer)
        return TracerNames(*tracers)

    def get_data_vector(self) -> DataVector:
        """Return this statistic's data vector."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector_real_space(self, tools: ModelingTools) -> TheoryVector:
        """Compute a two-point statistic in real space.

        This method computes the two-point statistic in real space. It first computes
        the Cl's in harmonic space and then translates them to real space using CCL.
        """
        tracers0 = self.source0.get_tracers(tools)
        tracers1 = self.source1.get_tracers(tools)
        scale0 = self.source0.get_scale()
        scale1 = self.source1.get_scale()

        assert self.ccl_kind != "cl"
        assert self.thetas is not None
        assert self.ells_for_xi is not None

        cells_for_xi = self.compute_cells(
            self.ells_for_xi, scale0, scale1, tools, tracers0, tracers1
        )

        theory_vector = pyccl.correlation(
            tools.get_ccl_cosmology(),
            ell=self.ells_for_xi,
            C_ell=cells_for_xi,
            theta=self.thetas / 60,
            type=self.ccl_kind,
        )
        return TheoryVector.create(theory_vector)

    def compute_theory_vector_harmonic_space(
        self, tools: ModelingTools
    ) -> TheoryVector:
        """Compute a two-point statistic in harmonic space.

        This method computes the two-point statistic in harmonic space. It computes
        either the Cl's at the ells provided by the SACC file or the ells required
        for the window function.
        """
        tracers0 = self.source0.get_tracers(tools)
        tracers1 = self.source1.get_tracers(tools)
        scale0 = self.source0.get_scale()
        scale1 = self.source1.get_scale()

        assert self.ccl_kind == "cl"
        assert (self.ells is not None) or (self.window is not None)

        if self.window is not None:
            # If a window function is provided, we need to compute the Cl's
            # for the ells used in the window function. To do this, we will
            # first compute the Cl's for the ells used in the interpolation
            # and then interpolate the results to the ells used in the window
            # function.
            assert self.window.ells_for_interpolation is not None
            cells_for_interpolation = self.compute_cells(
                self.window.ells_for_interpolation,
                scale0,
                scale1,
                tools,
                tracers0,
                tracers1,
            )

            # TODO: There is no code in Firecrown, neither test nor example,
            # that exercises a theory window function in any way.
            cell_interpolator = make_log_interpolator(
                self.window.ells_for_interpolation, cells_for_interpolation
            )
            # Deal with ell=0 and ell=1
            cells_interpolated = np.zeros(self.window.ells.size)
            cells_interpolated[2:] = cell_interpolator(self.window.ells[2:])

            # Here we left multiply the computed Cl's by the window function
            # to get the final Cl's.
            theory_vector = np.einsum(
                "lb, l -> b",
                self.window.weights,
                cells_interpolated,
            )
            # We also compute the mean ell value associated with each bin.
            self.mean_ells = np.einsum(
                "lb, l -> b", self.window.weights, self.window.ells
            )

            assert self.data_vector is not None
            return TheoryVector.create(theory_vector)

        # If we get here, we are working in harmonic space without a window function.
        assert self.ells is not None
        theory_vector = self.compute_cells(
            self.ells,
            scale0,
            scale1,
            tools,
            tracers0,
            tracers1,
        )

        return TheoryVector.create(theory_vector)

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a two-point statistic from sources."""
        if self.ccl_kind == "cl":
            return self.compute_theory_vector_harmonic_space(tools)

        return self.compute_theory_vector_real_space(tools)

    def compute_cells(
        self,
        ells: npt.NDArray[np.int64],
        scale0: float,
        scale1: float,
        tools: ModelingTools,
        tracers0: Sequence[Tracer],
        tracers1: Sequence[Tracer],
    ) -> npt.NDArray[np.float64]:
        """Compute the power spectrum for the given ells and tracers."""
        self.cells = {}
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
                        tuple(ells.tolist()),
                        p_of_k_a=pk,
                    )
                    * scale0
                    * scale1
                )
        self.cells[TRACER_NAMES_TOTAL] = np.array(sum(self.cells.values()))
        theory_vector = self.cells[TRACER_NAMES_TOTAL]
        return theory_vector

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
            raise NotImplementedError("Halo model power spectra not supported yet")
        else:
            raise ValueError(f"No power spectrum for {pk_name} can be found.")
        return pk
