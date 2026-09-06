"""NumCosmo likelihood factory API."""

from firecrown.connector.numcosmo._data import NumCosmoData, NumCosmoGaussCov
from firecrown.connector.numcosmo._mapping import MappingNumCosmo
from firecrown.likelihood import (
    ConstGaussian,
    Likelihood,
    NamedParameters,
    load_likelihood,
)


class NumCosmoFactory:
    """NumCosmo likelihood class.

    This class provide the necessary factory methods
    to create NumCosmo+firecrown likelihoods.
    """

    def __init__(
        self,
        likelihood_source: str,
        build_parameters: NamedParameters,
        mapping: MappingNumCosmo | None,
        model_list: list[str],
    ) -> None:
        """Initialize a NumCosmoFactory.

        :param likelihood_source: the filename for the likelihood factory function
        :param build_parameters: the build parameters
        :param mapping: the mapping
        :param model_list: the model list
        """
        likelihood, tools = load_likelihood(likelihood_source, build_parameters)

        self.data: NumCosmoGaussCov | NumCosmoData
        self.mapping: MappingNumCosmo | None = mapping
        if isinstance(likelihood, ConstGaussian):
            self.data = NumCosmoGaussCov.new_from_likelihood(
                likelihood,
                model_list,
                tools,
                mapping,
                likelihood_source,
                build_parameters,
            )
        else:
            self.data = NumCosmoData.new_from_likelihood(
                likelihood,
                model_list,
                tools,
                mapping,
                likelihood_source,
                build_parameters,
            )

    def get_data(self) -> NumCosmoGaussCov | NumCosmoData:
        """This method return the appropriate Ncm.Data class to be used by NumCosmo.

        :returns: the data used by NumCosmo
        """
        return self.data

    def get_mapping(self) -> MappingNumCosmo | None:
        """This method return the current MappingNumCosmo.

        :returns: the current mapping.
        """
        return self.mapping

    def get_firecrown_likelihood(self) -> Likelihood:
        """This method returns the Firecrown Likelihood.

        :returns: the likelihood
        """
        return self.data.likelihood
