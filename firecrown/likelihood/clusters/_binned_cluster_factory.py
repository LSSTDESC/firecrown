"""Factory for the binned cluster statistics of several cluster samples."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

import sacc

from firecrown.likelihood.clusters._binned_cluster import BinnedCluster
from firecrown.likelihood.clusters.abundance import (
    SACC_DATA_TYPES as ABUNDANCE_DATA_TYPES,
)
from firecrown.likelihood.clusters.abundance import BinnedClusterNumberCountsFactory
from firecrown.likelihood.clusters.lensing import SACC_DATA_TYPES as LENSING_DATA_TYPES
from firecrown.likelihood.clusters.lensing import BinnedClusterShearProfileFactory


class BinnedClusterFactory(BaseModel):
    """Factory for binned cluster statistics, as TwoPointFactory is for two-point.

    It holds at most one abundance factory and one lensing factory per cluster
    sample, identified by its SACC survey tracer. When a sample has both, they
    must use the same :class:`ClusterModelFactory`, so that its statistics
    describe the same clusters.

    Example YAML, with an anchor to write the model once::

        abundance_factories:
          - survey_name: my_survey
            model: &my_survey_model
              mass_distribution:
                type: MurataUnbinnedFactory
                pivot_log_mass: 14.625862906
                pivot_redshift: 0.6
        lensing_factories:
          - survey_name: my_survey
            observable: deltasigma
            cluster_concentration: 4.0
            model: *my_survey_model
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    abundance_factories: list[BinnedClusterNumberCountsFactory] = Field(
        default_factory=list
    )
    lensing_factories: list[BinnedClusterShearProfileFactory] = Field(
        default_factory=list
    )

    _abundance_map: dict[str, BinnedClusterNumberCountsFactory] = PrivateAttr()
    _lensing_map: dict[str, BinnedClusterShearProfileFactory] = PrivateAttr()

    @model_validator(mode="after")
    def _check_factories(self) -> Self:
        """Check the factories are unique per sample and share its model."""
        if not self.abundance_factories and not self.lensing_factories:
            raise ValueError(
                "BinnedClusterFactory needs at least one abundance or lensing factory."
            )
        for kind, names in (
            ("abundance", [f.survey_name for f in self.abundance_factories]),
            ("lensing", [f.survey_name for f in self.lensing_factories]),
        ):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"Duplicate {kind} factories for surveys {duplicates}."
                )
        models = {f.survey_name: f.model for f in self.abundance_factories}
        for factory in self.lensing_factories:
            model = models.get(factory.survey_name)
            if model is not None and model != factory.model:
                raise ValueError(
                    "The abundance and lensing factories of survey "
                    f"'{factory.survey_name}' must use the same model."
                )
        return self

    def model_post_init(self, _: object, /) -> None:
        """Index the factories by survey name."""
        self._abundance_map = {f.survey_name: f for f in self.abundance_factories}
        self._lensing_map = {f.survey_name: f for f in self.lensing_factories}

    def get_factory(
        self, data_type: str, survey_name: str
    ) -> BinnedClusterNumberCountsFactory | BinnedClusterShearProfileFactory:
        """Return the factory predicting a SACC data type for a survey.

        :param data_type: the SACC data type of the measurement
        :param survey_name: the name of the survey tracer
        :returns: the abundance or lensing factory of the survey
        """
        if data_type in ABUNDANCE_DATA_TYPES.values():
            factory_map: dict[
                str, BinnedClusterNumberCountsFactory | BinnedClusterShearProfileFactory
            ] = dict(self._abundance_map)
        elif data_type in LENSING_DATA_TYPES.values():
            factory_map = dict(self._lensing_map)
        else:
            raise ValueError(f"No cluster factory predicts the data type {data_type}.")
        if survey_name not in factory_map:
            raise ValueError(f"No factory for {data_type} of survey '{survey_name}'.")
        return factory_map[survey_name]

    def from_sacc(self, sacc_data: sacc.Sacc) -> list[BinnedCluster]:
        """Create the statistics of every factory and read their SACC data.

        :param sacc_data: the SACC data holding the measurements
        :returns: the abundance statistics followed by the lensing statistics
        """
        statistics: list[BinnedCluster] = [
            factory.from_sacc(sacc_data) for factory in self.abundance_factories
        ]
        statistics.extend(
            factory.from_sacc(sacc_data) for factory in self.lensing_factories
        )
        return statistics
