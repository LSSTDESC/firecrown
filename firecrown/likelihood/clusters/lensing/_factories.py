"""Factory for the binned cluster shear profile statistic."""

from __future__ import annotations

import warnings
from typing import Annotated, Literal

import sacc
from crow import ClusterShearProfile
from crow.properties import ClusterProperty
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from firecrown.likelihood.clusters._factories import (
    ClusterModelFactory,
    GridBinnedClusterRecipeFactory,
    _placeholder_cosmology,
    observables_in_sacc,
)
from firecrown.likelihood.clusters.lensing._statistic import (
    BinnedClusterShearProfile,
)

LensingObservable = Literal["deltasigma", "shear"]
"""Observables of :class:`BinnedClusterShearProfileFactory`."""

_PROPERTIES: dict[str, ClusterProperty] = {
    "deltasigma": ClusterProperty.DELTASIGMA,
    "shear": ClusterProperty.SHEAR,
}

# pylint: disable=no-member
SACC_DATA_TYPES: dict[str, str] = {
    "deltasigma": sacc.standard_types.cluster_delta_sigma,
    "shear": sacc.standard_types.cluster_shear,
}
# pylint: enable=no-member
"""SACC data type of each observable."""


class LensingEfficiencyOptions(BaseModel):
    """Options for the mean geometric lensing efficiency used in reduced shear.

    These map onto :meth:`crow.ClusterShearProfile.set_beta_parameters` and
    :meth:`crow.ClusterShearProfile.set_beta_s_interp`. They only matter when
    the reduced tangential shear (``shear``) is predicted; ``deltasigma``
    predictions do not depend on them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    z_inf: Annotated[
        float, Field(description="Redshift representing the source plane at infinity.")
    ] = 10.0
    z_max: Annotated[
        float,
        Field(description="Upper limit of the source redshift integration."),
    ] = 10.0
    delta_z_cut: Annotated[
        float,
        Field(
            description=(
                "Offset added to the cluster redshift to obtain the lower limit "
                "of the source redshift integration when z_min is not given."
            )
        ),
    ] = 0.1
    z_min: Annotated[
        float | None,
        Field(description="Fixed lower limit of the source redshift integration."),
    ] = None
    approx: Annotated[
        Literal["order1", "order2"],
        Field(description="Order of the reduced shear approximation."),
    ] = "order1"
    interpolate: Annotated[
        bool,
        Field(
            description=(
                "Tabulate the lensing efficiency on interpolation_nodes redshifts "
                "spanning the recipe's true_z_interval instead of integrating at "
                "every evaluation. crow builds the table once, when the recipe is "
                "created, with a placeholder cosmology and does not rebuild it "
                "when the cosmology is updated; leave this off whenever "
                "cosmological parameters vary."
            )
        ),
    ] = False
    interpolation_nodes: Annotated[
        int, Field(gt=1, description="Number of redshift nodes of the table.")
    ] = 10


class BinnedClusterShearProfileFactory(BaseModel):
    """Factory for the binned stacked lensing profile of one cluster sample.

    The statistic predicts the excess surface mass density (``deltasigma``) or
    the reduced tangential shear (``shear``) in the redshift, richness and
    radius bins of the SACC survey tracer ``survey_name``, with a
    :class:`crow.ClusterShearProfile` theory built from ``model``. It
    registers the sampler parameter ``cluster_theory_cluster_concentration``.

    Example YAML::

        survey_name: my_survey
        observable: deltasigma
        cluster_concentration: 4.0
        model:
          mass_distribution:
            type: MurataUnbinnedFactory
            pivot_log_mass: 14.625862906
            pivot_redshift: 0.6
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    survey_name: Annotated[
        str, Field(description="Name of the SACC survey tracer of the sample.")
    ]
    model: ClusterModelFactory
    observable: Annotated[
        LensingObservable | None,
        Field(
            description=(
                "Observable to predict. When omitted it is inferred from the SACC "
                "data of the survey."
            )
        ),
    ] = None
    cluster_concentration: Annotated[
        float | None,
        Field(
            description=(
                "Default of the sampler parameter "
                "cluster_theory_cluster_concentration. None, or any value of 0 "
                "or below, selects the Bhattacharya et al. (2013) "
                "concentration-mass relation, which crow's grid recipe cannot "
                "evaluate for lensing profiles."
            )
        ),
    ] = None
    two_halo_term: Annotated[
        bool,
        Field(
            description=(
                "Add the two-halo contribution. Only available for deltasigma."
            )
        ),
    ] = False
    boost_factor: Annotated[
        bool, Field(description="Apply the NFW boost-factor correction.")
    ] = False
    lensing_efficiency: LensingEfficiencyOptions = Field(
        default_factory=LensingEfficiencyOptions
    )

    @model_validator(mode="after")
    def _check_two_halo_term(self) -> Self:
        """Reject the two-halo term for the reduced shear."""
        self._check_observable(self.observable)
        return self

    def _check_observable(self, observable: str | None) -> None:
        """Check that an observable can be predicted with these options."""
        if observable is not None and observable not in _PROPERTIES:
            raise ValueError(
                f"Unknown lensing observable '{observable}'; "
                f"expected one of {list(_PROPERTIES)}."
            )
        if observable == "shear" and self.two_halo_term:
            raise ValueError(
                "crow does not support the two-halo term for the reduced shear; "
                "disable two_halo_term or predict 'deltasigma'."
            )

    @property
    def uses_concentration_mass_relation(self) -> bool:
        """Whether crow falls back to its concentration-mass relation.

        crow uses the Bhattacharya et al. (2013) relation whenever the
        concentration parameter is not positive.
        """
        return self.cluster_concentration is None or self.cluster_concentration <= 0

    def observables_in_sacc(self, sacc_data: sacc.Sacc) -> list[str]:
        """Return the observables the SACC data holds for the survey.

        :param sacc_data: the SACC data
        :returns: the names of the observables found
        """
        return observables_in_sacc(sacc_data, self.survey_name, SACC_DATA_TYPES)

    def create(self, observable: str | None = None) -> BinnedClusterShearProfile:
        """Create the statistic, which still has to read its SACC data.

        :param observable: the observable to predict; defaults to
            ``observable``, which must then be set
        :returns: the statistic, with a new crow recipe
        """
        # pylint: disable=no-member
        name = self.observable if observable is None else observable
        if name is None:
            raise ValueError(
                f"No observable to predict for survey '{self.survey_name}'; set "
                "'observable' or create the statistic from SACC data."
            )
        self._check_observable(name)
        if (
            isinstance(self.model.recipe, GridBinnedClusterRecipeFactory)
            and self.uses_concentration_mass_relation
        ):
            warnings.warn(
                "GridBinnedClusterRecipeFactory evaluates lensing profiles on a "
                "mass grid, which crow's concentration-mass relation does not "
                "support; the evaluation fails unless the sampler sets "
                "cluster_theory_cluster_concentration to a positive value. Set "
                "cluster_concentration or use ExactBinnedClusterRecipeFactory.",
                stacklevel=2,
            )
        is_delta_sigma = name == "deltasigma"
        options = self.lensing_efficiency
        cluster_theory = ClusterShearProfile(
            cosmo=_placeholder_cosmology(),
            halo_mass_function=self.model.halo_mass_function.create(),
            cluster_concentration=self.cluster_concentration,
            is_delta_sigma=is_delta_sigma,
            use_beta_s_interp=options.interpolate and not is_delta_sigma,
            two_halo_term=self.two_halo_term,
            boost_factor=self.boost_factor,
        )
        if not is_delta_sigma:
            cluster_theory.set_beta_parameters(
                z_inf=options.z_inf,
                zmax=options.z_max,
                delta_z_cut=options.delta_z_cut,
                zmin=options.z_min,
                approx=options.approx,
            )
        recipe = self.model.create_recipe(cluster_theory)
        if not is_delta_sigma and options.interpolate:
            z_min, z_max = recipe.true_z_interval
            cluster_theory.set_beta_s_interp(z_min, z_max, options.interpolation_nodes)
        return BinnedClusterShearProfile(
            _PROPERTIES[name],
            self.survey_name,
            recipe,
            parameter_prefix=self.model.parameter_prefix,
        )

    def from_sacc(self, sacc_data: sacc.Sacc) -> BinnedClusterShearProfile:
        """Create the statistic and read its SACC data.

        :param sacc_data: the SACC data holding the measurements
        :returns: the statistic, ready for use
        """
        name: str | None = self.observable
        if name is None:
            found = self.observables_in_sacc(sacc_data)
            if not found:
                raise ValueError(
                    "The SACC data contains no cluster lensing profiles for "
                    f"survey '{self.survey_name}'."
                )
            if len(found) > 1:
                raise ValueError(
                    f"Survey '{self.survey_name}' has both deltasigma and shear "
                    "measurements; set 'observable' to choose one."
                )
            name = found[0]
        statistic = self.create(name)
        statistic.read(sacc_data)
        return statistic
