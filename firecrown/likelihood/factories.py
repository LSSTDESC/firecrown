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

import yaml

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


def build_two_point_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """
    Build a likelihood object for two-point statistics from a SACC file.

    This function creates a likelihood object for two-point statistics using a SACC file
    and a set of statistic factories. The user must provide the SACC file and specify
    which statistic factories to use. The likelihood object is created by combining the
    SACC file with the specified statistic factories.

    :param build_parameters: A NamedParameters object containing the following parameters:
        - sacc_file: The SACC file containing the data.
        - statistic_factories: A YAML file containing the statistic factories to use.
    """
    sacc_file = build_parameters.get_string("sacc_file")
    statistic_factories = build_parameters.get_string("statistic_factories")
    harmonic = build_parameters.get_bool("harmonic", default_value=False)
    real = build_parameters.get_bool("real", default_value=False)

    if harmonic and real:
        raise ValueError(
            "Cannot use both harmonic and real space simultaneously at "
            "the same analysis"
        )
    if not (harmonic or real):
        raise ValueError("Must use either harmonic or real space at the same analysis")

    if sacc_file is None:
        raise ValueError("No SACC file provided")

    if statistic_factories is None:
        raise ValueError("No statistic factories provided")

    with open(statistic_factories, "r", encoding="utf-8") as f:
        statistic_factories_yaml = yaml.safe_load(f)

    if statistic_factories_yaml is None:
        raise ValueError("Error loading statistic factories from YAML")

    if "number_count_factory" not in statistic_factories_yaml:
        raise ValueError(
            "Number count factory not found in statistic factories "
            "[number_count_factory]"
        )
    if "weak_lensing_factory" not in statistic_factories_yaml:
        raise ValueError(
            "Weak lensing factory not found in statistic factories "
            "[weak_lensing_factory]"
        )

    wl_factory = WeakLensingFactory.model_validate(
        statistic_factories_yaml["weak_lensing_factory"], strict=True
    )
    nc_factory = NumberCountsFactory.model_validate(
        statistic_factories_yaml["number_count_factory"], strict=True
    )

    modeling_tools = ModelingTools()

    # Load the SACC file
    sacc_data = sacc.Sacc.load_fits(sacc_file)

    if harmonic:
        return (
            _build_two_point_likelihood_harmonic(sacc_data, wl_factory, nc_factory),
            modeling_tools,
        )

    return (
        _build_two_point_likelihood_real(sacc_data, wl_factory, nc_factory),
        modeling_tools,
    )


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
    check_two_point_consistence_real(tpms)

    two_points = TwoPoint.from_measurement(
        tpms, wl_factory=wl_factory, nc_factory=nc_factory
    )

    likelihood = ConstGaussian.create_ready(two_points, sacc_data.covariance.dense)

    return likelihood
