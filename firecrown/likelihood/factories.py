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

import yaml
from pydantic import BaseModel, ConfigDict, BeforeValidator

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
)
from firecrown.modeling_tools import ModelingTools
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


def _validate_correlation_space(value):
    if isinstance(value, str):
        try:
            return TwoPointCorrelationSpace(value)  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for TwoPointCorrelationSpace: {value}"
            ) from exc
    return value


class TwoPointFactory(BaseModel):
    """Factory class for WeakLensing objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_space: Annotated[
        TwoPointCorrelationSpace, BeforeValidator(_validate_correlation_space)
    ]
    weak_lensing_factory: WeakLensingFactory
    number_counts_factory: NumberCountsFactory

    def model_post_init(self, __context) -> None:
        """Initialize the WeakLensingFactory object."""


class DataSourceSacc(BaseModel):
    """Model for the data source in a likelihood configuration."""

    sacc_data_file: str

    def model_post_init(self, __context) -> None:
        """Initialize the DataSourceSacc object."""
        sacc_data_file = Path(self.sacc_data_file)
        if not sacc_data_file.exists():
            raise FileNotFoundError(f"File {sacc_data_file} does not exist")

    def get_sacc_data(self) -> sacc.Sacc:
        """Load the SACC data file."""
        return sacc.Sacc.load_fits(self.sacc_data_file)


class TwoPointExperiment(BaseModel):
    """Model for the two-point experiment in a likelihood configuration."""

    two_point_factory: TwoPointFactory
    data_source: DataSourceSacc

    def model_post_init(self, __context) -> None:
        """Initialize the TwoPointExperiment object."""


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

    with open(likelihood_config_file, "r", encoding="utf-8") as f:
        likelihood_config = yaml.safe_load(f)

    if likelihood_config is None:
        raise ValueError("No likelihood config found.")

    exp = TwoPointExperiment.model_validate(likelihood_config, strict=True)
    modeling_tools = ModelingTools()

    # Load the SACC file
    sacc_data = exp.data_source.get_sacc_data()

    match exp.two_point_factory.correlation_space:
        case TwoPointCorrelationSpace.REAL:
            likelihood = _build_two_point_likelihood_real(
                sacc_data,
                exp.two_point_factory.weak_lensing_factory,
                exp.two_point_factory.number_counts_factory,
            )
        case TwoPointCorrelationSpace.HARMONIC:
            likelihood = _build_two_point_likelihood_harmonic(
                sacc_data,
                exp.two_point_factory.weak_lensing_factory,
                exp.two_point_factory.number_counts_factory,
            )

    return likelihood, modeling_tools


def _build_two_point_likelihood_harmonic(
    sacc_data: sacc.Sacc,
    wl_factory: WeakLensingFactory,
    nc_factory: NumberCountsFactory,
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

    two_points = TwoPoint.from_measurement(
        tpms, wl_factory=wl_factory, nc_factory=nc_factory
    )

    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood


def _build_two_point_likelihood_real(
    sacc_data: sacc.Sacc,
    wl_factory: WeakLensingFactory,
    nc_factory: NumberCountsFactory,
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

    two_points = TwoPoint.from_measurement(
        tpms, wl_factory=wl_factory, nc_factory=nc_factory
    )

    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood
