"""Two point statistic support."""

from __future__ import annotations
import itertools
import warnings
from typing import Annotated, Sequence

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.nl_pt
import sacc.windows
from pydantic import (
    BaseModel,
    ConfigDict,
    BeforeValidator,
    PrivateAttr,
    Field,
)

import firecrown.generators.two_point as gen
from firecrown.likelihood.source import Source, Tracer
from firecrown.likelihood.weak_lensing import WeakLensingFactory, WeakLensing
from firecrown.likelihood.number_counts import NumberCountsFactory, NumberCounts
from firecrown.likelihood.statistic import Statistic
from firecrown.metadata_types import (
    GALAXY_LENS_TYPES,
    GALAXY_SOURCE_TYPES,
    InferredGalaxyZDist,
    Measurement,
    TracerNames,
    TwoPointCorrelationSpace,
    TwoPointHarmonic,
    TwoPointReal,
    TypeSource,
)

from firecrown.metadata_functions import (
    TwoPointHarmonicIndex,
    TwoPointRealIndex,
    extract_window_function,
    measurements_from_index,
    make_correlation_space,
)
from firecrown.data_types import DataVector, TheoryVector, TwoPointMeasurement

from firecrown.modeling_tools import ModelingTools
from firecrown.models.two_point import (
    TwoPointTheory,
    calculate_pk,
    ApplyInterpolationWhen,
)
from firecrown.updatable import UpdatableCollection
from firecrown.utils import (
    cached_angular_cl,
    make_log_interpolator,
    ClIntegrationOptions,
)
import firecrown.metadata_types as mdt


# only supported types are here, anything else will throw
# a value error


def calculate_angular_cl(
    ells: npt.NDArray[np.int64],
    pk_name: str,
    scale0: float,
    scale1: float,
    tools: ModelingTools,
    tracer0: Tracer,
    tracer1: Tracer,
    int_options: ClIntegrationOptions | None = None,
):
    """Calculate the angular mulitpole moments.

    :param ells: The angular wavenumbers at which to compute the power spectrum.
    :param pk_name: The name of the power spectrum to return.
    :param scale0: The scale factor for the first tracer.
    :param scale1: The scale factor for the second tracer.
    :param tools: The modeling tools to use.
    :param tracer0: The first tracer to use.
    :param tracer1: The second tracer to use.
    :return: The angular mulitpole moments.
    """
    pk = calculate_pk(pk_name, tools, tracer0, tracer1)
    cosmo_in = tools.get_ccl_cosmology()
    result = (
        cached_angular_cl(
            tools.get_ccl_cosmology(),
            (tracer0.ccl_tracer, tracer1.ccl_tracer),
            tuple(ells.ravel().tolist()),
            p_of_k_a=pk,
            p_of_k_a_lin=cosmo_in.get_linear_power(),
            int_options=int_options,
        )
        * scale0
        * scale1
    )
    return result


# pylint: disable=too-many-public-methods
class TwoPoint(Statistic):
    """A statistic that represents the correlation between two measurements.

    If the same source is used twice in the same TwoPoint object, this produces
    an autocorrelation.

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

    """

    @property
    def sacc_data_type(self) -> str:
        """Backwards compatibility for sacc_data_type."""
        return self.theory.sacc_data_type

    @property
    def source0(self) -> Source:
        """Backwards compatibility for source0."""
        return self.theory.source0

    @property
    def source1(self) -> Source:
        """Backwards compatibility for source1."""
        return self.theory.source1

    @property
    def window(self) -> None | npt.NDArray[np.float64]:
        """Backwards compatibility for window."""
        return self.theory.window

    @property
    def sacc_tracers(self) -> None | TracerNames:
        """Backwards compatibility for sacc_tracers."""
        return self.theory.sacc_tracers

    @property
    def ells(self) -> None | npt.NDArray[np.int64]:
        """Backwards compatibility for ells."""
        return self.theory.ells

    @property
    def thetas(self) -> None | npt.NDArray[np.float64]:
        """Backwards compatibility for thetas."""
        return self.theory.thetas

    @property
    def ells_for_xi(self) -> None | npt.NDArray[np.int64]:
        """Backwards compatibility for ells_for_xi."""
        return self.theory.ells_for_xi

    @property
    def cells(self):
        """Backwards compatibility for cells."""
        return self.theory.cells

    def __init__(
        self,
        sacc_data_type: str,
        source0: Source,
        source1: Source,
        *,
        interp_ells_gen: gen.LogLinearElls = gen.LogLinearElls(),
        ell_or_theta: None | gen.EllOrThetaConfig = None,
        tracers: None | TracerNames = None,
        int_options: ClIntegrationOptions | None = None,
        apply_interp: ApplyInterpolationWhen = ApplyInterpolationWhen.DEFAULT,
    ) -> None:
        super().__init__()
        self.theory = TwoPointTheory(
            sacc_data_type=sacc_data_type,
            sources=(source0, source1),
            interp_ells_gen=interp_ells_gen,
            ell_or_theta=ell_or_theta,
            tracers=tracers,
            int_options=int_options,
            apply_interp=apply_interp,
        )
        self._data: None | DataVector = None

    @classmethod
    def from_metadata_index(
        cls,
        metadata_indices: Sequence[TwoPointHarmonicIndex | TwoPointRealIndex],
        tp_factory: TwoPointFactory,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of TwoPoint statistics.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointCellsIndex or TwoPointXiThetaIndex metadata index objects. The
        purpose of this constructor is to create a TwoPoint statistic from metadata
        index, which requires a follow-up call to `read` to read the data and metadata
        from the SACC object.

        :param metadata_index: The metadata index objects to initialize the TwoPoint
            statistics.
        :param tp_factory: The TwoPointFactory to use.

        :return: An UpdatableCollection of TwoPoint statistics.
        """
        two_point_list = [
            cls(
                sacc_data_type=metadata_index["data_type"],
                source0=use_source_factory_metadata_index(n1, a, tp_factory),
                source1=use_source_factory_metadata_index(n2, b, tp_factory),
                int_options=tp_factory.int_options,
            )
            for metadata_index in metadata_indices
            for n1, a, n2, b in [measurements_from_index(metadata_index)]
        ]
        return UpdatableCollection(two_point_list)

    @classmethod
    def _from_metadata_single(
        cls, metadata: TwoPointHarmonic | TwoPointReal, tp_factory: TwoPointFactory
    ) -> TwoPoint:
        """Create a single TwoPoint statistic from metadata.

        This constructor creates a single TwoPoint statistic from a TwoPointHarmonic or
        TwoPointReal metadata object. It requires the sources to be initialized before
        calling this constructor. The metadata object is used to initialize the TwoPoint
        statistic. No further calls to `read` are needed.
        """
        match metadata:
            case TwoPointHarmonic():
                two_point = cls._from_metadata_single_base(metadata, tp_factory)
                two_point.theory.ells = metadata.ells
                two_point.theory.window = metadata.window
            case TwoPointReal():
                two_point = cls._from_metadata_single_base(metadata, tp_factory)
                two_point.theory.thetas = metadata.thetas
                two_point.theory.window = None
            case _:
                raise ValueError(f"Metadata of type {type(metadata)} is not supported!")
        two_point.ready = True
        return two_point

    @classmethod
    def _from_metadata_single_base(
        cls, metadata: TwoPointHarmonic | TwoPointReal, tp_factory: TwoPointFactory
    ):
        """Create a single TwoPoint statistic from metadata.

        Base method for creating a single TwoPoint statistic from metadata.

        :param metadata: The metadata object to initialize the TwoPoint statistic.
        :param wl_factory: The weak lensing factory to use.
        :param nc_factory: The number counts factory to use.

        :return: A TwoPoint statistic.
        """
        source0 = use_source_factory(
            metadata.XY.x, metadata.XY.x_measurement, tp_factory
        )
        source1 = use_source_factory(
            metadata.XY.y, metadata.XY.y_measurement, tp_factory
        )
        two_point = cls(
            metadata.get_sacc_name(),
            source0,
            source1,
            tracers=metadata.XY.get_tracer_names(),
            int_options=tp_factory.int_options,
        )
        return two_point

    @classmethod
    def from_metadata(
        cls,
        metadata_seq: Sequence[TwoPointHarmonic | TwoPointReal],
        tp_factory: TwoPointFactory,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of TwoPoint statistics from metadata.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointHarmonic or TwoPointReal metadata objects. The metadata objects
        are used to initialize the TwoPoint statistics. The sources are initialized
        using the factories provided.

        Note that TwoPoint created with this constructor are ready to be used, but
        contain no data.

        :param metadata_seq: The metadata objects to initialize the TwoPoint statistics.
        :param wl_factory: The weak lensing factory to use.
        :param nc_factory: The number counts factory to use.

        :return: An UpdatableCollection of TwoPoint statistics.
        """
        two_point_list = [
            cls._from_metadata_single(metadata, tp_factory) for metadata in metadata_seq
        ]

        return UpdatableCollection(two_point_list)

    @classmethod
    def create_two_point(
        cls, measurement: TwoPointMeasurement, tp_factory: TwoPointFactory
    ) -> TwoPoint:
        """Create a single TwoPoint statistic from a measurement.

        :param measurement: The measurement object to initialize the TwoPoint statistic.
        """
        two_point = cls._from_metadata_single(measurement.metadata, tp_factory)
        two_point.sacc_indices = measurement.indices
        two_point.set_data_vector(DataVector.create(measurement.data))
        two_point.ready = True
        return two_point

    @classmethod
    def from_measurement(
        cls,
        measurements: Sequence[TwoPointMeasurement],
        tp_factory: TwoPointFactory,
    ) -> UpdatableCollection[TwoPoint]:
        """Create an UpdatableCollection of TwoPoint statistics from measurements.

        This constructor creates an UpdatableCollection of TwoPoint statistics from a
        list of TwoPointMeasurement objects. The measurements are used to initialize the
        TwoPoint statistics. The sources are initialized using the factories provided.

        Note that TwoPoint created with this constructor are ready to be used and
        contain data.

        :param measurements: The measurements objects to initialize the TwoPoint
            statistics.
        :param wl_factory: The weak lensing factory to use.
        :param nc_factory: The number counts factory to use.

        :return: An UpdatableCollection of TwoPoint statistics.
        """
        two_point_list: list[TwoPoint] = [
            cls.create_two_point(m, tp_factory) for m in measurements
        ]
        return UpdatableCollection(two_point_list)

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file.

        :param sacc_data: The data in the sacc format.
        """
        self.theory.initialize_sources(sacc_data)

        if self.theory.ccl_kind == "cl":
            self.read_harmonic_space(sacc_data)
        else:
            self.read_real_space(sacc_data)

        super().read(sacc_data)

    def read_real_space(self, sacc_data: sacc.Sacc):
        """Read the data for this statistic from the SACC file."""
        assert self.theory.sacc_tracers is not None
        thetas_xis_indices = read_reals(self.theory, sacc_data)
        # We do not support window functions for real space statistics
        if thetas_xis_indices is not None:
            thetas, xis, sacc_indices = thetas_xis_indices
            if self.theory.ell_or_theta_config is not None:
                # If we have data from our construction, and also have data in the
                # SACC object, emit a warning and use the information read from the
                # SACC object.
                warnings.warn(
                    f"Tracers '{self.theory.sacc_tracers}' have 2pt data and you have "
                    f"specified `theta` in the configuration. `theta` is being "
                    f"ignored!",
                    stacklevel=2,
                )
        else:
            if self.theory.ell_or_theta_config is None:
                # The SACC file has no data points, just a tracer, in this case we
                # are building the statistic from scratch. In this case the user
                # must have set the dictionary ell_or_theta, containing the
                # minimum, maximum and number of bins to generate the ell values.
                raise RuntimeError(
                    f"Tracers '{self.theory.sacc_tracers}' for data type "
                    f"'{self.theory.sacc_data_type}' "
                    "have no 2pt data in the SACC file and no input theta values "
                    "were given!"
                )
            thetas, xis = gen.generate_reals(self.theory.ell_or_theta_config)
            sacc_indices = None
        self.theory.thetas = thetas
        self.sacc_indices = sacc_indices
        self._data = DataVector.create(xis)

    def read_harmonic_space(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic from the SACC file."""
        assert self.theory.sacc_tracers is not None
        ells_cells_indices = read_ell_cells(self.theory, sacc_data)
        Cells, ells, sacc_indices, window = self.read_harmonic_spectrum_data(
            ells_cells_indices, sacc_data
        )

        self.theory.ells = ells
        self.theory.window = window
        self.sacc_indices = sacc_indices
        self._data = DataVector.create(Cells)

    def read_harmonic_spectrum_data(
        self,
        ells_cells_indices: (
            None
            | tuple[
                npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]
            ]
        ),
        sacc_data: sacc.Sacc,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.float64] | None,
    ]:
        """Read all the data for this statistic from the SACC file.

        :param ells_cells_indices: The ells, the cells and the indices of the
            data in the SACC file.
        :param sacc_data: The data in the sacc format.
        :return: The ells, the cells and the indices, and window function if
            there is one.
        """
        if ells_cells_indices is not None:
            ells, Cells, sacc_indices = ells_cells_indices
            if self.theory.ell_or_theta_config is not None:
                # If we have data from our construction, and also have data in the
                # SACC object, emit a warning and use the information read from the
                # SACC object.
                warnings.warn(
                    f"Tracers '{self.theory.sacc_tracers}' have 2pt data and you have "
                    f"specified `ell` in the configuration. `ell` is being ignored!",
                    stacklevel=2,
                )
            replacement_ells: None | npt.NDArray[np.int64]
            window: None | npt.NDArray[np.float64]
            replacement_ells, window = extract_window_function(sacc_data, sacc_indices)
            if window is not None:
                # When using a window function, we do not calculate all Cl's.
                # For this reason we have a default set of ells that we use
                # to compute Cl's, and we have a set of ells used for
                # interpolation.
                assert replacement_ells is not None
                ells = replacement_ells
        else:
            if self.theory.ell_or_theta_config is None:
                # The SACC file has no data points, just a tracer, in this case we
                # are building the statistic from scratch. In this case the user
                # must have set the dictionary ell_or_theta, containing the
                # minimum, maximum and number of bins to generate the ell values.
                raise RuntimeError(
                    f"Tracers '{self.theory.sacc_tracers}' for data type "
                    f"'{self.theory.sacc_data_type}' "
                    "have no 2pt data in the SACC file and no input ell values "
                    "were given!"
                )
            ells, Cells = gen.generate_ells_cells(self.theory.ell_or_theta_config)
            sacc_indices = None

            # When generating the ells and Cells we do not have a window function
            window = None
        return Cells, ells, sacc_indices, window

    def get_data_vector(self) -> DataVector:
        """Return this statistic's data vector."""
        assert self._data is not None
        return self._data

    def set_data_vector(self, value: DataVector) -> None:
        """Set this statistic's data vector."""
        assert value is not None
        self._data = value

    def compute_theory_vector_real_space(self, tools: ModelingTools) -> TheoryVector:
        """Compute a two-point statistic in real space.

        This method computes the two-point statistic in real space. It first computes
        the Cl's in harmonic space and then translates them to real space using CCL.
        """
        assert self.theory.ccl_kind != "cl"
        assert self.theory.thetas is not None
        assert self.theory.ells_for_xi is not None

        tracers0, scale0, tracers1, scale1 = self.theory.get_tracers_and_scales(tools)

        # Compute the angular power spectrum (C_ell) at the multipoles specified in
        # ells_for_xi. CCL will later interpolate between these values as needed.
        if self.theory.apply_interp & ApplyInterpolationWhen.REAL:
            ells = self.theory.ells_for_xi
        else:
            ells = self.theory.interp_ells_gen.generate_all()

        cells_for_xi = self.compute_cells(
            ells, scale0, scale1, tools, tracers0, tracers1, interpolate=False
        )

        # Compute the real-space correlation function xi(theta). CCL uses the input
        # ells_for_xi and corresponding cells_for_xi, interpolates as needed, and
        # performs the Hankel transform to obtain xi at the specified angles.
        theory_vector = pyccl.correlation(
            tools.get_ccl_cosmology(),
            ell=ells,
            C_ell=cells_for_xi,
            theta=self.theory.thetas / 60,
            type=self.theory.ccl_kind,
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
        assert self.theory.ccl_kind == "cl"
        assert self.theory.ells is not None

        tracers0, scale0, tracers1, scale1 = self.theory.get_tracers_and_scales(tools)
        if self.theory.window is not None:
            # We are using a window function. This means we will have effective
            # ells, and effective Cells at those effective ells.
            cells = self.compute_cells(
                self.theory.ells,
                scale0,
                scale1,
                tools,
                tracers0,
                tracers1,
                interpolate=ApplyInterpolationWhen.HARMONIC_WINDOW
                in self.theory.apply_interp,
            )

            # Here we left multiply the computed Cl's by the window function to get the
            # final Cl's.
            theory_vector = np.einsum("lb, l -> b", self.theory.window, cells)
            # We also compute the mean ell value associated with each bin.
            self.theory.mean_ells = np.einsum(
                "lb, l -> b", self.theory.window, self.theory.ells
            )

            assert self._data is not None
            return TheoryVector.create(theory_vector)

        # If we get here, we are working in harmonic space without a window function.
        assert self.theory.ells is not None
        theory_vector = self.compute_cells(
            self.theory.ells,
            scale0,
            scale1,
            tools,
            tracers0,
            tracers1,
            interpolate=ApplyInterpolationWhen.HARMONIC in self.theory.apply_interp,
        )

        return TheoryVector.create(theory_vector)

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a two-point statistic from sources."""
        if self.theory.ccl_kind == "cl":
            return self.compute_theory_vector_harmonic_space(tools)

        return self.compute_theory_vector_real_space(tools)

    def _compute_cells_all_orders(
        self,
        ells: npt.NDArray[np.int64],
        scale0: float,
        scale1: float,
        tools: ModelingTools,
        tracers0: Sequence[Tracer],
        tracers1: Sequence[Tracer],
    ) -> npt.NDArray[np.float64]:
        """Compute the power spectrum for the given ells and tracers."""
        self.theory.cells = {}
        if tracers0 == tracers1:
            assert scale0 == scale1
        # We should consider how to avoid doing the same calculation twice,
        # if possible.
        for tracer0, tracer1 in itertools.product(tracers0, tracers1):
            pk_name = f"{tracer0.field}:{tracer1.field}"
            tn = TracerNames(tracer0.tracer_name, tracer1.tracer_name)
            result = calculate_angular_cl(
                ells,
                pk_name,
                scale0,
                scale1,
                tools,
                tracer0,
                tracer1,
                self.theory.int_options,
            )
            self.theory.cells[tn] = result
        self.theory.cells[mdt.TRACER_NAMES_TOTAL] = np.array(
            sum(self.theory.cells.values())
        )
        theory_vector = self.theory.cells[mdt.TRACER_NAMES_TOTAL]
        return theory_vector

    def _compute_cells_interpolated(
        self,
        ells: npt.NDArray[np.int64],
        ells_for_interpolation: npt.NDArray[np.int64],
        scale0: float,
        scale1: float,
        tools: ModelingTools,
        tracers0: Sequence[Tracer],
        tracers1: Sequence[Tracer],
    ) -> npt.NDArray[np.float64]:
        """Compute the interpolated power spectrum for the given ells and tracers.

        :param ells: The angular wavenumbers at which to compute the power spectrum.
        :param ells_for_interpolation: The angular wavenumbers at which the power
            spectrum is computed for interpolation.
        :param scale0: The scale factor for the first tracer.
        :param scale1: The scale factor for the second tracer.
        :param tools: The modeling tools to use.
        :param tracers0: The first tracers to use.
        :param tracers1: The second tracers to use.

        Compute the power spectrum for the given ells and tracers and interpolate
        the result to the ells provided.

        :return: The interpolated power spectrum.
        """
        computed_cells = self.compute_cells(
            ells_for_interpolation, scale0, scale1, tools, tracers0, tracers1
        )
        cell_interpolator = make_log_interpolator(
            ells_for_interpolation, computed_cells
        )
        cell_interpolated = np.zeros(len(ells))
        # We should not interpolate ell 0 and 1
        ells_larger_than_1 = ells > 1
        cell_interpolated[ells_larger_than_1] = cell_interpolator(
            ells[ells_larger_than_1]
        )
        return cell_interpolated

    def compute_cells(
        self,
        ells: npt.NDArray[np.int64],
        scale0: float,
        scale1: float,
        tools: ModelingTools,
        tracers0: Sequence[Tracer],
        tracers1: Sequence[Tracer],
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Compute the power spectrum for the given ells and tracers.

        This method computes the power spectrum for the given ells and tracers. If
        interpolate is True, it will interpolate the power spectrum to the ells
        provided.
        """
        if interpolate:
            # ells_for_interpolation are true ells (and thus integral).
            # These are the values at which we will have CCL calculate the "exact"
            # C_ells: these form our interpolation table.
            ells_for_interpolation = self.theory.generate_ells_for_interpolation()

            # The call below will calculate the "exact" C_ells (using CCL). Using these
            # "exact" C_ells it will then interpolate to determine C_ells at the
            # required ell values.
            return self._compute_cells_interpolated(
                ells,
                ells_for_interpolation,
                scale0,
                scale1,
                tools,
                tracers0,
                tracers1,
            )
        # No interpolation, all multipoles are computed exactly
        return self._compute_cells_all_orders(
            ells, scale0, scale1, tools, tracers0, tracers1
        )


def read_reals(
    theory: TwoPointTheory,
    sacc_data: sacc.Sacc,
) -> (
    None
    | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
):
    """Read and return theta and xi.

    :param theory: The theory, carrying data type and tracers.
    :param sacc_data: The SACC data object to be read.
    :return: The theta and xi values.
    """
    thetas, xis = sacc_data.get_theta_xi(
        theory.sacc_data_type, *theory.sacc_tracers, return_cov=False
    )
    # As version 0.13 of sacc, the method get_real returns the
    # theta values and the xi values in arrays of the same length.
    assert len(thetas) == len(xis)

    common_length = len(thetas)
    if common_length == 0:
        return None
    sacc_indices = np.atleast_1d(
        sacc_data.indices(theory.sacc_data_type, theory.sacc_tracers)
    )
    assert sacc_indices is not None  # Needed for mypy
    assert len(sacc_indices) == common_length
    return thetas, xis, sacc_indices


def read_ell_cells(
    theory: TwoPointTheory,
    sacc_data: sacc.Sacc,
) -> (
    None | tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]
):
    """Read and return ell and Cell.

    :param theory: The theory, carrying data type and tracers.
    :param sacc_data: The SACC data object to be read.
    :return: The ell and Cell values.
    """
    tracers = theory.sacc_tracers
    ells, cells = sacc_data.get_ell_cl(
        theory.sacc_data_type, *tracers, return_cov=False
    )
    # As version 0.13 of sacc, the method get_ell_cl returns the
    # ell values and the Cl values in arrays of the same length.
    assert len(ells) == len(cells)
    common_length = len(ells)
    if common_length == 0:
        return None
    sacc_indices = np.atleast_1d(sacc_data.indices(theory.sacc_data_type, tracers))
    assert sacc_indices is not None  # Needed for mypy
    assert len(sacc_indices) == common_length
    return ells, cells, sacc_indices


class ggTwoPointFactory(BaseModel):
    """Factory class for WeakLensing objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_space: Annotated[
        TwoPointCorrelationSpace,
        BeforeValidator(make_correlation_space),
        Field(description="The two-point correlation space."),
    ]
    weak_lensing_factories: list[WeakLensingFactory] = Field(default_factory=list)
    number_counts_factories: list[NumberCountsFactory] = Field(default_factory=list)
    int_options: ClIntegrationOptions | None = None

    _wl_factory_map: dict[TypeSource, WeakLensingFactory] = PrivateAttr()
    _nc_factory_map: dict[TypeSource, NumberCountsFactory] = PrivateAttr()

    def model_post_init(self, _, /) -> None:
        """Initialize the WeakLensingFactory object."""
        self._wl_factory_map: dict[TypeSource, WeakLensingFactory] = {}
        self._nc_factory_map: dict[TypeSource, NumberCountsFactory] = {}

        for wl_factory in self.weak_lensing_factories:
            if wl_factory.type_source in self._wl_factory_map:
                raise ValueError(
                    f"Duplicate WeakLensingFactory found for "
                    f"type_source {wl_factory.type_source}."
                )
            self._wl_factory_map[wl_factory.type_source] = wl_factory

        for nc_factory in self.number_counts_factories:
            if nc_factory.type_source in self._nc_factory_map:
                raise ValueError(
                    f"Duplicate NumberCountsFactory found for "
                    f"type_source {nc_factory.type_source}."
                )
            self._nc_factory_map[nc_factory.type_source] = nc_factory

    def get_factory(
        self, measurement: Measurement, type_source: TypeSource = TypeSource.DEFAULT
    ) -> WeakLensingFactory | NumberCountsFactory:
        """Get the Factory for the given Measurement and TypeSource."""
        match measurement:
            case measurement if measurement in GALAXY_SOURCE_TYPES:
                if type_source not in self._wl_factory_map:
                    raise ValueError(
                        f"No WeakLensingFactory found for type_source {type_source}."
                    )
                return self._wl_factory_map[type_source]
            case measurement if measurement in GALAXY_LENS_TYPES:
                if type_source not in self._nc_factory_map:
                    raise ValueError(
                        f"No NumberCountsFactory found for type_source {type_source}."
                    )
                return self._nc_factory_map[type_source]
            case _:
                raise (
                    ValueError(
                        f"Factory not found for measurement {measurement} "
                        f"is not supported."
                    )
                )

    def from_measurement(
        self, tpms: list[TwoPointMeasurement]
    ) -> UpdatableCollection[TwoPoint]:
        """Create a TwoPoint object from a list of TwoPointMeasurement."""
        return TwoPoint.from_measurement(measurements=tpms, tp_factory=self)

    def from_metadata(
        self, metadata_seq: list[TwoPointHarmonic | TwoPointReal]
    ) -> UpdatableCollection[TwoPoint]:
        """Create a TwoPoint object from a list of TwoPointHarmonic or TwoPointReal."""
        return TwoPoint.from_metadata(metadata_seq=metadata_seq, tp_factory=self)


def use_source_factory(
    inferred_galaxy_zdist: InferredGalaxyZDist,
    measurement: Measurement,
    tp_factory: TwoPointFactory,
) -> WeakLensing | NumberCounts:
    """Apply the factory to the inferred galaxy redshift distribution."""
    if measurement not in inferred_galaxy_zdist.measurements:
        raise ValueError(
            f"Measurement {measurement} not found in inferred galaxy redshift "
            f"distribution {inferred_galaxy_zdist.bin_name}!"
        )

    source_factory = tp_factory.get_factory(
        measurement, inferred_galaxy_zdist.type_source
    )
    source = source_factory.create(inferred_galaxy_zdist)
    return source


def use_source_factory_metadata_index(
    sacc_tracer: str,
    measurement: Measurement,
    tp_factory: TwoPointFactory,
) -> WeakLensing | NumberCounts:
    """Apply the factory to create a source using metadata only.

    This method is used when the galaxy redshift distribution is not available. It
    defaults to using the factory associated with the default TypeSource, since SACC
    does not encode TypeSource information.
    """
    source_factory = tp_factory.get_factory(measurement)
    source = source_factory.create_from_metadata_only(sacc_tracer)
    return source
