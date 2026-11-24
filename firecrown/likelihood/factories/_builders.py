"""Factory methods for creating Likelihood objects from configuration."""

from firecrown.likelihood.factories._models import (
    ClusterExperiment,
    TwoPointExperiment,
)
from firecrown.likelihood_base import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools


def build_cluster_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Build a likelihood object for binned cluster statistics from a SACC file.

    This function creates a likelihood object for binned cluster number counts,
    mean masses and lensing profiles using a SACC file and a set of binned cluster
    factories described in a YAML file (see :class:`ClusterExperiment`).

    :param build_parameters: A NamedParameters object containing the following
        parameters:
        - likelihood_config: A YAML file describing the cluster experiment.
    """
    likelihood_config_file = build_parameters.get_string("likelihood_config")
    exp = ClusterExperiment.load_from_yaml(likelihood_config_file)
    modeling_tools = ModelingTools(ccl_factory=exp.ccl_factory)

    likelihood = exp.make_likelihood()

    return likelihood, modeling_tools


def build_two_point_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Build a likelihood object for two-point statistics from a SACC file.

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
