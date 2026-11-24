"""Factories for the crow model shared by the binned cluster statistics.

A :class:`ClusterModelFactory` describes the crow model of one cluster sample:
the halo mass function, the mass-proxy relation, the redshift kernel, optional
completeness and purity models, and the integration recipe. The number counts
and lensing statistics of a sample each build their own crow recipe from the
same model, so they share its sampler parameters.

The crow objects are created with a placeholder cosmology; the statistics
inject the cosmology from :class:`~firecrown.modeling_tools.ModelingTools` at
every update.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

import pyccl
import sacc
from crow import ClusterAbundance, kernel, mass_proxy
from crow.cluster_modules.completeness_models import CompletenessAguena16
from crow.cluster_modules.purity_models import PurityAguena16, PurityAguena16LnProxy
from crow.integrator.numcosmo_integrator import (
    NumCosmoIntegralMethod,
    NumCosmoIntegrator,
)
from crow.recipes.binned_exact import ExactBinnedClusterRecipe
from crow.recipes.binned_grid import GridBinnedClusterRecipe
from crow.recipes.binned_parent import BinnedClusterRecipe
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self


def _validate_interval(value: list[float]) -> list[float]:
    """Check that an interval is ``[lower, upper]`` with ``lower < upper``."""
    if len(value) != 2 or not value[0] < value[1]:
        raise ValueError(
            f"An interval must be [lower, upper] with lower < upper; got {value}."
        )
    return value


Interval = Annotated[list[float], AfterValidator(_validate_interval)]


def _placeholder_cosmology() -> pyccl.Cosmology:
    """Return the cosmology used to construct crow objects.

    crow objects require a cosmology at construction. The statistics replace it
    with the cosmology from :class:`~firecrown.modeling_tools.ModelingTools` at
    every update, so this value does not enter the predictions, with one
    exception: the lensing efficiency interpolation table (see
    :class:`~firecrown.likelihood.clusters.lensing.LensingEfficiencyOptions`).
    """
    return pyccl.CosmologyVanillaLCDM()


def observables_in_sacc(
    sacc_data: sacc.Sacc, survey_name: str, data_types: Mapping[str, str]
) -> list[str]:
    """Return the observables with data for a survey in a SACC object.

    An observable is present when its SACC data type has at least one data
    point whose first tracer is ``survey_name``.

    :param sacc_data: the SACC data
    :param survey_name: name of the survey tracer
    :param data_types: SACC data type of each observable name
    :returns: the names of the observables found, in the order of ``data_types``
    """
    return [
        name
        for name, data_type in data_types.items()
        if any(
            tracers[0] == survey_name
            for tracers in sacc_data.get_tracer_combinations(data_type=data_type)
        )
    ]


class HaloMassFunctionFactory(BaseModel):
    """Factory for pyccl halo mass functions.

    The mass function is looked up by name with
    :meth:`pyccl.halos.MassFunc.from_name` (for example ``Tinker08``,
    ``Tinker10``, ``Bocquet16`` or ``Despali16``) and evaluated in the given
    mass definition (for example ``200c``, ``200m``, ``500c`` or ``vir``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[
        str, Field(description="Name registered with pyccl.halos.MassFunc.from_name.")
    ] = "Tinker08"
    mass_def: Annotated[
        str, Field(description="Mass definition understood by pyccl.halos.MassDef.")
    ] = "200c"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        try:
            pyccl.halos.MassFunc.from_name(value)
        except KeyError as exc:
            raise ValueError(f"Unknown halo mass function '{value}'.") from exc
        return value

    @field_validator("mass_def")
    @classmethod
    def _validate_mass_def(cls, value: str) -> str:
        try:
            pyccl.halos.MassDef.from_name(value)
        except ValueError as exc:
            raise ValueError(f"Invalid mass definition '{value}': {exc}") from exc
        return value

    def create(self) -> pyccl.halos.MassFunc:
        """Create the halo mass function object."""
        return pyccl.halos.MassFunc.from_name(self.name)(mass_def=self.mass_def)


class MurataBinnedFactory(BaseModel):
    """Factory for the binned Murata et al. (2019) mass-richness relation.

    The relation is integrated analytically over each richness bin; it is the
    variant required by :class:`ExactBinnedClusterRecipeFactory`. It registers
    the sampler parameters ``mass_distribution_mu0``, ``mu1``, ``mu2``,
    ``sigma0``, ``sigma1`` and ``sigma2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MurataBinnedFactory"],
        Field(description="The type of the mass distribution."),
    ] = "MurataBinnedFactory"
    pivot_log_mass: Annotated[
        float, Field(description="Pivot log10 halo mass of the relation.")
    ]
    pivot_redshift: Annotated[float, Field(description="Pivot redshift.")]

    def create(self) -> mass_proxy.MurataBinned:
        """Create a new :class:`crow.mass_proxy.MurataBinned`."""
        return mass_proxy.MurataBinned(self.pivot_log_mass, self.pivot_redshift)


class MurataUnbinnedFactory(BaseModel):
    """Factory for the unbinned Murata et al. (2019) mass-richness relation.

    The relation is evaluated on a richness grid; it is the variant required by
    :class:`GridBinnedClusterRecipeFactory`. It registers the same sampler
    parameters as :class:`MurataBinnedFactory`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["MurataUnbinnedFactory"],
        Field(description="The type of the mass distribution."),
    ] = "MurataUnbinnedFactory"
    pivot_log_mass: Annotated[
        float, Field(description="Pivot log10 halo mass of the relation.")
    ]
    pivot_redshift: Annotated[float, Field(description="Pivot redshift.")]

    def create(self) -> mass_proxy.MurataUnbinned:
        """Create a new :class:`crow.mass_proxy.MurataUnbinned`."""
        return mass_proxy.MurataUnbinned(self.pivot_log_mass, self.pivot_redshift)


MassDistributionFactory = Annotated[
    MurataBinnedFactory | MurataUnbinnedFactory,
    Field(discriminator="type"),
]


class SpectroscopicRedshiftFactory(BaseModel):
    """Factory for the spectroscopic (exact) cluster redshift kernel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["SpectroscopicRedshiftFactory"],
        Field(description="The type of the redshift distribution."),
    ] = "SpectroscopicRedshiftFactory"

    def create(self) -> kernel.SpectroscopicRedshift:
        """Create a new :class:`crow.kernel.SpectroscopicRedshift`."""
        return kernel.SpectroscopicRedshift()


class CompletenessAguena16Factory(BaseModel):
    """Factory for the Aguena & Lima (2016) completeness model.

    It registers the sampler parameters ``completeness_a_n``, ``b_n``,
    ``a_logm_piv`` and ``b_logm_piv``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["CompletenessAguena16Factory"],
        Field(description="The type of the completeness model."),
    ] = "CompletenessAguena16Factory"

    def create(self) -> CompletenessAguena16:
        """Create a new :class:`crow.completeness_models.CompletenessAguena16`."""
        return CompletenessAguena16()


class PurityAguena16Factory(BaseModel):
    """Factory for the Aguena & Lima (2016) purity model in richness.

    It registers the sampler parameters ``purity_a_n``, ``b_n``, ``a_logm_piv``
    and ``b_logm_piv``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PurityAguena16Factory"],
        Field(description="The type of the purity model."),
    ] = "PurityAguena16Factory"

    def create(self) -> PurityAguena16:
        """Create a new :class:`crow.purity_models.PurityAguena16`."""
        return PurityAguena16()


class PurityAguena16LnProxyFactory(BaseModel):
    """Factory for the Aguena & Lima (2016) purity model in log richness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Annotated[
        Literal["PurityAguena16LnProxyFactory"],
        Field(description="The type of the purity model."),
    ] = "PurityAguena16LnProxyFactory"

    def create(self) -> PurityAguena16LnProxy:
        """Create a new :class:`crow.purity_models.PurityAguena16LnProxy`."""
        return PurityAguena16LnProxy()


PurityFactory = Annotated[
    PurityAguena16Factory | PurityAguena16LnProxyFactory,
    Field(discriminator="type"),
]


class _BinnedClusterRecipeFactoryBase(BaseModel):
    """Integration limits shared by the binned cluster recipes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mass_interval: Annotated[
        Interval,
        Field(description="log10 halo mass range integrated over, [lower, upper]."),
    ] = Field(default_factory=lambda: [11.0, 17.0])
    true_z_interval: Annotated[
        Interval,
        Field(description="True redshift range integrated over, [lower, upper]."),
    ] = Field(default_factory=lambda: [0.0, 5.0])

    def _intervals(self) -> tuple[tuple[float, float], tuple[float, float]]:
        mass_interval = (self.mass_interval[0], self.mass_interval[1])
        true_z_interval = (self.true_z_interval[0], self.true_z_interval[1])
        return mass_interval, true_z_interval


class ExactBinnedClusterRecipeFactory(_BinnedClusterRecipeFactoryBase):
    """Factory for :class:`crow.recipes.binned_exact.ExactBinnedClusterRecipe`.

    The recipe integrates over mass, redshift and mass proxy with NumCosmo's
    adaptive quadrature; it requires the optional ``numcosmo`` dependency and
    a :class:`MurataBinnedFactory` mass distribution.
    """

    type: Annotated[
        Literal["ExactBinnedClusterRecipeFactory"],
        Field(description="The type of the cluster recipe."),
    ] = "ExactBinnedClusterRecipeFactory"
    integral_method: Annotated[
        Literal["p", "p_v", "h", "h_v"] | None,
        Field(
            description=(
                "NumCosmo cubature: p- or h-adaptive, optionally vectorized "
                "(_v). None keeps crow's default, p-adaptive cubature, which "
                "converges to a wrong value for some deltasigma bins; h_v "
                "does not."
            )
        ),
    ] = None

    def create(
        self,
        *,
        cluster_theory: ClusterAbundance,
        redshift_distribution: kernel.SpectroscopicRedshift,
        mass_distribution: mass_proxy.MurataBinned | mass_proxy.MurataUnbinned,
        completeness: CompletenessAguena16 | None,
        purity: PurityAguena16 | PurityAguena16LnProxy | None,
    ) -> ExactBinnedClusterRecipe:
        """Create a new :class:`crow.recipes.binned_exact.ExactBinnedClusterRecipe`."""
        mass_interval, true_z_interval = self._intervals()
        recipe = ExactBinnedClusterRecipe(
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            purity=purity,
            mass_interval=mass_interval,
            true_z_interval=true_z_interval,
        )
        if self.integral_method is not None:
            recipe.integrator = NumCosmoIntegrator(
                method=NumCosmoIntegralMethod[self.integral_method.upper()]
            )
        return recipe


class GridBinnedClusterRecipeFactory(_BinnedClusterRecipeFactoryBase):
    """Factory for :class:`crow.recipes.binned_grid.GridBinnedClusterRecipe`.

    The recipe tabulates the integrand on fixed mass, redshift and mass-proxy
    grids and integrates with Simpson's rule. It requires a
    :class:`MurataUnbinnedFactory` mass distribution. The fixed grids make the
    predictions smooth in the parameters, which suits finite-difference
    derivatives.
    """

    type: Annotated[
        Literal["GridBinnedClusterRecipeFactory"],
        Field(description="The type of the cluster recipe."),
    ] = "GridBinnedClusterRecipeFactory"
    proxy_grid_size: Annotated[
        int, Field(gt=1, description="Number of mass-proxy grid points.")
    ] = 30
    redshift_grid_size: Annotated[
        int, Field(gt=1, description="Number of redshift grid points.")
    ] = 30
    mass_grid_size: Annotated[
        int, Field(gt=1, description="Number of log10 mass grid points.")
    ] = 30

    def create(
        self,
        *,
        cluster_theory: ClusterAbundance,
        redshift_distribution: kernel.SpectroscopicRedshift,
        mass_distribution: mass_proxy.MurataBinned | mass_proxy.MurataUnbinned,
        completeness: CompletenessAguena16 | None,
        purity: PurityAguena16 | PurityAguena16LnProxy | None,
    ) -> GridBinnedClusterRecipe:
        """Create a new :class:`crow.recipes.binned_grid.GridBinnedClusterRecipe`."""
        mass_interval, true_z_interval = self._intervals()
        return GridBinnedClusterRecipe(
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            purity=purity,
            mass_interval=mass_interval,
            true_z_interval=true_z_interval,
            proxy_grid_size=self.proxy_grid_size,
            redshift_grid_size=self.redshift_grid_size,
            mass_grid_size=self.mass_grid_size,
        )


ClusterRecipeFactory = Annotated[
    ExactBinnedClusterRecipeFactory | GridBinnedClusterRecipeFactory,
    Field(discriminator="type"),
]

_REQUIRED_MASS_DISTRIBUTION: dict[type, type] = {
    ExactBinnedClusterRecipeFactory: MurataBinnedFactory,
    GridBinnedClusterRecipeFactory: MurataUnbinnedFactory,
}


class ClusterModelFactory(BaseModel):
    """The crow model of one cluster sample, shared by all of its statistics.

    Each statistic of the sample builds its own crow recipe from this model
    with :meth:`create_recipe`, so all of them register the same sampler
    parameters (for example ``mass_distribution_mu0``) and receive the same
    values.

    Example YAML::

        mass_distribution:
          type: MurataBinnedFactory
          pivot_log_mass: 14.625862906
          pivot_redshift: 0.6
        recipe:
          type: ExactBinnedClusterRecipeFactory
          mass_interval: [12.0, 17.0]
          true_z_interval: [0.1, 2.0]
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    halo_mass_function: HaloMassFunctionFactory = Field(
        default_factory=HaloMassFunctionFactory
    )
    mass_distribution: MassDistributionFactory
    redshift_distribution: SpectroscopicRedshiftFactory = Field(
        default_factory=SpectroscopicRedshiftFactory
    )
    completeness: CompletenessAguena16Factory | None = None
    purity: PurityFactory | None = None
    recipe: ClusterRecipeFactory = Field(default_factory=GridBinnedClusterRecipeFactory)
    parameter_prefix: Annotated[
        str | None,
        Field(
            description=(
                "Prefix for the sampler parameters of this sample, e.g. "
                "'redmapper' gives 'redmapper_mass_distribution_mu0'. Without a "
                "prefix, samples share parameter names."
            )
        ),
    ] = None

    @model_validator(mode="after")
    def _check_mass_distribution(self) -> Self:
        """Check that the recipe supports the mass distribution."""
        required = _REQUIRED_MASS_DISTRIBUTION[type(self.recipe)]
        if not isinstance(self.mass_distribution, required):
            raise ValueError(
                f"{type(self.recipe).__name__} requires a {required.__name__} "
                f"mass distribution, not {type(self.mass_distribution).__name__}."
            )
        return self

    def create_recipe(self, cluster_theory: ClusterAbundance) -> BinnedClusterRecipe:
        """Create a crow recipe for a cluster theory, with new crow objects.

        :param cluster_theory: the cluster theory of the statistic
        :returns: the crow recipe
        """
        # pylint: disable=no-member
        return self.recipe.create(
            cluster_theory=cluster_theory,
            redshift_distribution=self.redshift_distribution.create(),
            mass_distribution=self.mass_distribution.create(),
            completeness=(
                None if self.completeness is None else self.completeness.create()
            ),
            purity=None if self.purity is None else self.purity.create(),
        )
