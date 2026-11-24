"""Factory for the binned cluster number counts and mean mass statistic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import sacc
from crow import ClusterAbundance
from crow.properties import ClusterProperty
from pydantic import BaseModel, ConfigDict, Field

from firecrown.likelihood.clusters._factories import (
    ClusterModelFactory,
    _placeholder_cosmology,
    observables_in_sacc,
)
from firecrown.likelihood.clusters.abundance._statistic import (
    BinnedClusterNumberCounts,
)

AbundanceObservable = Literal["counts", "mass"]
"""Observables of :class:`BinnedClusterNumberCountsFactory`."""

_PROPERTIES: dict[str, ClusterProperty] = {
    "counts": ClusterProperty.COUNTS,
    "mass": ClusterProperty.MASS,
}

# pylint: disable=no-member
SACC_DATA_TYPES: dict[str, str] = {
    "counts": sacc.standard_types.cluster_counts,
    "mass": sacc.standard_types.cluster_mean_log_mass,
}
# pylint: enable=no-member
"""SACC data type of each observable."""


class BinnedClusterNumberCountsFactory(BaseModel):
    """Factory for the binned number counts and mean mass of one cluster sample.

    The statistic predicts the counts (``counts``) and the mean log10 mass
    (``mass``) in the redshift and richness bins of the SACC survey tracer
    ``survey_name``, with a :class:`crow.ClusterAbundance` theory built from
    ``model``.

    Example YAML::

        survey_name: my_survey
        observables: [counts, mass]
        model:
          mass_distribution:
            type: MurataBinnedFactory
            pivot_log_mass: 14.625862906
            pivot_redshift: 0.6
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    survey_name: Annotated[
        str, Field(description="Name of the SACC survey tracer of the sample.")
    ]
    model: ClusterModelFactory
    observables: Annotated[
        Annotated[list[AbundanceObservable], Field(min_length=1)] | None,
        Field(
            description=(
                "Observables to predict. When omitted they are inferred from the "
                "SACC data of the survey."
            )
        ),
    ] = None

    def observables_in_sacc(self, sacc_data: sacc.Sacc) -> list[str]:
        """Return the observables the SACC data holds for the survey.

        :param sacc_data: the SACC data
        :returns: the names of the observables found
        """
        return observables_in_sacc(sacc_data, self.survey_name, SACC_DATA_TYPES)

    def create(
        self, observables: Sequence[str] | None = None
    ) -> BinnedClusterNumberCounts:
        """Create the statistic, which still has to read its SACC data.

        :param observables: the observables to predict; defaults to
            ``observables``, which must then be set
        :returns: the statistic, with a new crow recipe
        """
        names = self.observables if observables is None else observables
        if not names:
            raise ValueError(
                f"No observables to predict for survey '{self.survey_name}'; set "
                "'observables' or create the statistic from SACC data."
            )
        properties = ClusterProperty.NONE
        for name in names:
            if name not in _PROPERTIES:
                raise ValueError(
                    f"Unknown abundance observable '{name}'; "
                    f"expected one of {list(_PROPERTIES)}."
                )
            properties |= _PROPERTIES[name]
        # pylint: disable=no-member
        cluster_theory = ClusterAbundance(
            cosmo=_placeholder_cosmology(),
            halo_mass_function=self.model.halo_mass_function.create(),
        )
        return BinnedClusterNumberCounts(
            properties,
            self.survey_name,
            self.model.create_recipe(cluster_theory),
            parameter_prefix=self.model.parameter_prefix,
        )

    def from_sacc(self, sacc_data: sacc.Sacc) -> BinnedClusterNumberCounts:
        """Create the statistic and read its SACC data.

        :param sacc_data: the SACC data holding the measurements
        :returns: the statistic, ready for use
        """
        names = self.observables or self.observables_in_sacc(sacc_data)
        if not names:
            raise ValueError(
                "The SACC data contains no cluster counts or mean masses for "
                f"survey '{self.survey_name}'."
            )
        statistic = self.create(names)
        statistic.read(sacc_data)
        return statistic
