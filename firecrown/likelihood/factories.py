"""
Factory functions for creating likelihoods from SACC files.

This module provides factory functions to create likelihood objects by combining a SACC
file and a set of statistic factories. Users can define their own custom statistic
factories for advanced use cases or rely on the generic factory functions provided here
for simpler scenarios.

For straightforward contexts where all data in the SACC file is utilized, the generic
factories simplify the process. The user only needs to supply the SACC file and specify
which statistic factories to use, and the likelihood factory will handle the creation of
the likelihood object, assembling the necessary components automatically.

These functions are particularly useful when the full set of statistics present in a
SACC file is being used without the need for complex customization.
"""

from typing import Annotated
from enum import Enum, auto
from pathlib import Path

from typing_extensions import assert_never
import yaml
from pydantic import BaseModel, ConfigDict, BeforeValidator, Field, field_serializer

import sacc
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.likelihood.two_point import TwoPoint
from firecrown.data_functions import (
    extract_all_real_data,
    extract_all_harmonic_data,
    check_two_point_consistence_real,
    check_two_point_consistence_harmonic,
    TwoPointBinFilterCollection,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory
from firecrown.utils import YAMLSerializable


class TwoPointCorrelationSpace(YAMLSerializable, str, Enum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.lower()

    REAL = auto()
    HARMONIC = auto()


def _validate_correlation_space(value: TwoPointCorrelationSpace | str):
    if not isinstance(value, TwoPointCorrelationSpace) and isinstance(value, str):
        try:
            return TwoPointCorrelationSpace(
                value.lower()
            )  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for TwoPointCorrelationSpace: {value}"
            ) from exc
    return value


class TwoPointFactory(BaseModel):
    """Factory class for WeakLensing objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_space: Annotated[
        TwoPointCorrelationSpace,
        BeforeValidator(_validate_correlation_space),
        Field(description="The two-point correlation space."),
    ]
    weak_lensing_factory: WeakLensingFactory
    number_counts_factory: NumberCountsFactory

    def model_post_init(self, _, /) -> None:
        """Initialize the WeakLensingFactory object."""

    @field_serializer("correlation_space")
    @classmethod
    def serialize_correlation_space(cls, value: TwoPointCorrelationSpace) -> str:
        """Serialize the amplitude parameter."""
        return value.name


class DataSourceSacc(BaseModel):
    """Model for the data source in a likelihood configuration."""

    sacc_data_file: str
    filters: TwoPointBinFilterCollection | None = None
    _path: Path | None = None

    def set_path(self, path: Path) -> None:
        """Set the path for the data source."""
        self._path = path

    def get_filepath(self) -> Path:
        """Return the filename of the data source.

        Raises a FileNotFoundError if the file does not exist.
        :return: The filename
        """
        sacc_data_path = Path(self.sacc_data_file)
        # If sacc_data_file is absolute, use it directly
        if sacc_data_path.is_absolute() and sacc_data_path.exists():
            return Path(self.sacc_data_file)
        # If path is set, use it to find the file
        if self._path is not None:
            full_sacc_data_path = self._path / sacc_data_path
            if full_sacc_data_path.exists():
                return full_sacc_data_path
        # If path is not set, use the current directory
        elif sacc_data_path.exists():
            return sacc_data_path
        # If the file does not exist, raise an error
        raise FileNotFoundError(f"File {sacc_data_path} does not exist")

    def get_sacc_data(self) -> sacc.Sacc:
        """Load the SACC data file."""
        filename = self.get_filepath()
        return sacc.Sacc.load_fits(filename)


def ensure_path(file: str | Path) -> Path:
    """Ensure the file path is a Path object."""
    match file:
        case str():
            return Path(file)
        case Path():
            return file
        case _ as unreachable:
            assert_never(unreachable)


class TwoPointExperiment(BaseModel):
    """Model for the two-point experiment in a likelihood configuration."""

    two_point_factory: TwoPointFactory
    data_source: DataSourceSacc
    ccl_factory: CCLFactory | None = None

    def model_post_init(self, _, /) -> None:
        """Initialize the TwoPointExperiment object."""
        if self.ccl_factory is None:
            self.ccl_factory = CCLFactory()

    @classmethod
    def load_from_yaml(cls, file: str | Path) -> "TwoPointExperiment":
        """Load a TwoPointExperiment object from a YAML file."""
        filepath = ensure_path(file)

        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            tpe = cls.model_validate(config, strict=True)

        # Record the file directory
        tpe.data_source.set_path(filepath.parent)
        return tpe

    def make_likelihood(self) -> Likelihood:
        """Create a likelihood object for two-point statistics from a SACC file."""
        # Load the SACC file
        sacc_data = self.data_source.get_sacc_data()

        likelihood: None | Likelihood = None
        match self.two_point_factory.correlation_space:
            case TwoPointCorrelationSpace.REAL:
                likelihood = _build_two_point_likelihood_real(
                    sacc_data,
                    self.two_point_factory.weak_lensing_factory,
                    self.two_point_factory.number_counts_factory,
                    filters=self.data_source.filters,
                )
            case TwoPointCorrelationSpace.HARMONIC:
                likelihood = _build_two_point_likelihood_harmonic(
                    sacc_data,
                    self.two_point_factory.weak_lensing_factory,
                    self.two_point_factory.number_counts_factory,
                    filters=self.data_source.filters,
                )
            case _ as unreachable:
                assert_never(unreachable)
        assert likelihood is not None
        return likelihood


def build_two_point_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """
    Build a likelihood object for two-point statistics from a SACC file.

    This function creates a likelihood object for two-point statistics using a SACC file
    and a set of statistic factories. The user must provide the SACC file and specify
    which statistic factories to use. The likelihood object is created by combining the
    SACC file with the specified statistic factories.

    :param build_parameters: A NamedParameters object containing the following
        parameters:
        - sacc_file: The SACC file containing the data.
        - statistic_factories: A YAML file containing the statistic factories to use.
    """
    likelihood_config_file = build_parameters.get_string("likelihood_config")
    exp = TwoPointExperiment.load_from_yaml(likelihood_config_file)
    modeling_tools = ModelingTools(ccl_factory=exp.ccl_factory)

    likelihood = exp.make_likelihood()

    return likelihood, modeling_tools


def _build_two_point_likelihood_harmonic(
    sacc_data: sacc.Sacc,
    wl_factory: WeakLensingFactory,
    nc_factory: NumberCountsFactory,
    filters: TwoPointBinFilterCollection | None = None,
):
    """
    Build a likelihood object for two-point statistics in harmonic space.

    This function creates a likelihood object for two-point statistics in harmonic space
    using a SACC file and a set of statistic factories. The user must provide the SACC
    file and specify which statistic factories to use. The likelihood object is created
    by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param wl_factory: The weak lensing statistic factory.
    :param nc_factory: The number counts statistic factory.

    :return: A likelihood object for two-point statistics in harmonic space.
    """
    tpms = extract_all_harmonic_data(sacc_data)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in harmonic space found in the SACC file."
        )
    check_two_point_consistence_harmonic(tpms)
    if filters is not None:
        tpms = filters(tpms)

    two_points = TwoPoint.from_measurement(
        tpms, wl_factory=wl_factory, nc_factory=nc_factory
    )

    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood


def _build_two_point_likelihood_real(
    sacc_data: sacc.Sacc,
    wl_factory: WeakLensingFactory,
    nc_factory: NumberCountsFactory,
    filters: TwoPointBinFilterCollection | None = None,
):
    """
    Build a likelihood object for two-point statistics in real space.

    This function creates a likelihood object for two-point statistics in real space
    using a SACC file and a set of statistic factories. The user must provide the SACC
    file and specify which statistic factories to use. The likelihood object is created
    by combining the SACC file with the specified statistic factories.

    :param sacc_data: The SACC file containing the data.
    :param wl_factory: The weak lensing statistic factory.
    :param nc_factory: The number counts statistic factory.

    :return: A likelihood object for two-point statistics in real space.
    """
    tpms = extract_all_real_data(sacc_data)
    if len(tpms) == 0:
        raise ValueError(
            "No two-point measurements in real space found in the SACC file."
        )
    check_two_point_consistence_real(tpms)
    if filters is not None:
        tpms = filters(tpms)

    two_points = TwoPoint.from_measurement(
        tpms, wl_factory=wl_factory, nc_factory=nc_factory
    )

    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood
