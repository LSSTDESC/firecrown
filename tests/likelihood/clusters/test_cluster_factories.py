"""Tests for the binned galaxy cluster factories."""

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyccl
import pytest
import sacc
from crow import ClusterAbundance, ClusterShearProfile, mass_proxy
from crow.integrator.numcosmo_integrator import NumCosmoIntegralMethod
from crow.properties import ClusterProperty
from crow.recipes.binned_exact import ExactBinnedClusterRecipe
from crow.recipes.binned_grid import GridBinnedClusterRecipe

from firecrown.data_functions import TwoPointBinFilterCollection
from firecrown.likelihood import ConstGaussian, NamedParameters, TwoPointFactory
from firecrown.likelihood.clusters import (
    BinnedCluster,
    BinnedClusterFactory,
    BinnedClusterNumberCounts,
    BinnedClusterNumberCountsFactory,
    BinnedClusterShearProfile,
    BinnedClusterShearProfileFactory,
    ClusterModelFactory,
    CompletenessAguena16Factory,
    ExactBinnedClusterRecipeFactory,
    GridBinnedClusterRecipeFactory,
    HaloMassFunctionFactory,
    LensingEfficiencyOptions,
    MurataBinnedFactory,
    MurataUnbinnedFactory,
    PurityAguena16Factory,
    PurityAguena16LnProxyFactory,
)
from firecrown.likelihood.clusters._factories import observables_in_sacc
from firecrown.likelihood.clusters.abundance import SACC_DATA_TYPES
from firecrown.likelihood.factories import (
    ClusterExperiment,
    DataSourceSacc,
    TwoPointExperiment,
    build_cluster_likelihood,
)
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.metadata_types import TwoPointCorrelationSpace
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import get_default_params_map
from firecrown.utils import base_model_from_yaml, base_model_to_yaml

PIVOT_LOG_MASS = 14.625862906
PIVOT_REDSHIFT = 0.6
MURATA_PARAMETERS = ("mu0", "mu1", "mu2", "sigma0", "sigma1", "sigma2")
MURATA_NAMES = {f"mass_distribution_{p}" for p in MURATA_PARAMETERS}

CLUSTER_FACTORY_YAML = """
abundance_factories:
  - survey_name: my_survey
    observables: [counts]
    model: &my_survey_model
      halo_mass_function: {name: Tinker08, mass_def: 200c}
      mass_distribution:
        type: MurataUnbinnedFactory
        pivot_log_mass: 14.625862906
        pivot_redshift: 0.6
      recipe:
        type: GridBinnedClusterRecipeFactory
        mass_interval: [13.0, 17.0]
        true_z_interval: [0.1, 2.0]
lensing_factories:
  - survey_name: my_survey
    observable: deltasigma
    cluster_concentration: 4.0
    model: *my_survey_model
"""

CLUSTER_EXPERIMENT_YAML = """
data_source:
  sacc_data_file: clusters.fits
cluster_factory:
  abundance_factories:
    - survey_name: my_survey
      observables: [counts, mass]
      model:
        mass_distribution:
          type: MurataBinnedFactory
          pivot_log_mass: 14.625862906
          pivot_redshift: 0.6
        recipe:
          type: ExactBinnedClusterRecipeFactory
          mass_interval: [13.0, 17.0]
          true_z_interval: [0.0, 2.0]
"""


def exact_model(**overrides: Any) -> ClusterModelFactory:
    """Model with the exact recipe and the binned Murata relation."""
    kwargs: dict[str, Any] = {
        "mass_distribution": MurataBinnedFactory(
            pivot_log_mass=PIVOT_LOG_MASS, pivot_redshift=PIVOT_REDSHIFT
        ),
        "recipe": ExactBinnedClusterRecipeFactory(
            mass_interval=[13.0, 17.0], true_z_interval=[0.0, 2.0]
        ),
    }
    kwargs.update(overrides)
    return ClusterModelFactory(**kwargs)


def grid_model(**overrides: Any) -> ClusterModelFactory:
    """Model with a small grid recipe and the unbinned Murata relation."""
    kwargs: dict[str, Any] = {
        "mass_distribution": MurataUnbinnedFactory(
            pivot_log_mass=PIVOT_LOG_MASS, pivot_redshift=PIVOT_REDSHIFT
        ),
        "recipe": GridBinnedClusterRecipeFactory(
            mass_interval=[13.0, 17.0],
            true_z_interval=[0.1, 2.0],
            redshift_grid_size=20,
            mass_grid_size=50,
            proxy_grid_size=20,
        ),
    }
    kwargs.update(overrides)
    return ClusterModelFactory(**kwargs)


def abundance(**overrides: Any) -> BinnedClusterNumberCountsFactory:
    """Counts and mean mass factory of my_survey with the exact recipe."""
    kwargs: dict[str, Any] = {
        "survey_name": "my_survey",
        "observables": ["counts", "mass"],
        "model": exact_model(),
    }
    kwargs.update(overrides)
    return BinnedClusterNumberCountsFactory(**kwargs)


def lensing(**overrides: Any) -> BinnedClusterShearProfileFactory:
    """Excess surface density factory of my_survey with the grid recipe."""
    kwargs: dict[str, Any] = {
        "survey_name": "my_survey",
        "observable": "deltasigma",
        "cluster_concentration": 4.0,
        "model": grid_model(),
    }
    kwargs.update(overrides)
    return BinnedClusterShearProfileFactory(**kwargs)


def parameter_names(statistic: BinnedCluster) -> set[str]:
    """Return the sampler parameter names of a statistic."""
    return set(statistic.required_parameters().get_params_names())


def with_covariance(sacc_data: sacc.Sacc) -> sacc.Sacc:
    """Return a copy of the SACC data with an identity covariance."""
    result = sacc_data.copy()
    result.add_covariance(np.eye(len(result.mean)))
    return result


def evaluate(likelihood: ConstGaussian, tools: ModelingTools) -> float:
    """Update likelihood and tools with default parameters and return chi-squared."""
    params = get_default_params_map(tools, likelihood)
    tools.update(params)
    tools.prepare()
    likelihood.update(params)
    return likelihood.compute_chisq(tools)


def cluster_experiment(
    cluster_sacc_data: sacc.Sacc, tmp_path: Path, factory: BinnedClusterFactory
) -> ClusterExperiment:
    """Save the SACC data with a covariance and return an experiment reading it."""
    sacc_path = tmp_path / "clusters.fits"
    with_covariance(cluster_sacc_data).save_fits(str(sacc_path), overwrite=True)
    return ClusterExperiment(
        cluster_factory=factory,
        data_source=DataSourceSacc(sacc_data_file=str(sacc_path)),
    )


# Shared crow components


def test_halo_mass_function_factory_defaults() -> None:
    """The default mass function is Tinker08 in 200c."""
    hmf = HaloMassFunctionFactory().create()
    assert isinstance(hmf, pyccl.halos.MassFuncTinker08)
    assert hmf.mass_def.Delta == 200
    assert hmf.mass_def.rho_type == "critical"


def test_halo_mass_function_factory_other_model() -> None:
    """Other mass functions and definitions are looked up by name."""
    hmf = HaloMassFunctionFactory(name="Tinker10", mass_def="200m").create()
    assert isinstance(hmf, pyccl.halos.MassFuncTinker10)
    assert hmf.mass_def.rho_type == "matter"


def test_halo_mass_function_factory_invalid() -> None:
    """Unknown mass functions and definitions are rejected."""
    with pytest.raises(ValueError, match="Unknown halo mass function"):
        HaloMassFunctionFactory(name="NotAMassFunction")
    with pytest.raises(ValueError, match="Invalid mass definition"):
        HaloMassFunctionFactory(mass_def="200x")


def test_interval_validation() -> None:
    """Intervals must be [lower, upper] with lower < upper."""
    with pytest.raises(ValueError, match="lower < upper"):
        ExactBinnedClusterRecipeFactory(mass_interval=[17.0, 13.0])
    with pytest.raises(ValueError, match="lower < upper"):
        GridBinnedClusterRecipeFactory(true_z_interval=[0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "integral_method,expected",
    [(None, NumCosmoIntegralMethod.P_V), ("h_v", NumCosmoIntegralMethod.H_V)],
)
def test_exact_recipe_integral_method(
    integral_method: Literal["h_v"] | None, expected: NumCosmoIntegralMethod
) -> None:
    """The exact recipe uses the requested cubature, or crow's default."""
    model = exact_model(
        recipe=ExactBinnedClusterRecipeFactory(integral_method=integral_method)
    )
    recipe = abundance(model=model).create().cluster_recipe
    assert isinstance(recipe, ExactBinnedClusterRecipe)
    assert recipe.integrator.method == expected


def test_model_requires_matching_mass_distribution() -> None:
    """Each recipe requires its variant of the Murata relation."""
    with pytest.raises(ValueError, match="requires a MurataBinnedFactory"):
        exact_model(mass_distribution=grid_model().mass_distribution)
    with pytest.raises(ValueError, match="requires a MurataUnbinnedFactory"):
        grid_model(mass_distribution=exact_model().mass_distribution)


def test_model_create_recipe() -> None:
    """Every recipe is new and holds the model's crow objects."""
    model = exact_model()
    theory = ClusterAbundance(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=HaloMassFunctionFactory().create(),
    )
    recipe = model.create_recipe(theory)
    assert isinstance(recipe, ExactBinnedClusterRecipe)
    assert recipe.cluster_theory is theory
    assert isinstance(recipe.mass_distribution, mass_proxy.MurataBinned)
    assert recipe.mass_interval == (13.0, 17.0)
    assert recipe.true_z_interval == (0.0, 2.0)
    assert recipe.completeness is None
    assert recipe.purity is None
    assert model.create_recipe(theory) is not recipe


def test_observables_in_sacc(cluster_sacc_data: sacc.Sacc) -> None:
    """Observables are found per survey from their SACC data types."""
    assert observables_in_sacc(cluster_sacc_data, "my_survey", SACC_DATA_TYPES) == [
        "counts",
        "mass",
    ]
    assert not observables_in_sacc(cluster_sacc_data, "absent", SACC_DATA_TYPES)


# Abundance factory


def test_abundance_create() -> None:
    """The abundance statistic uses a ClusterAbundance theory."""
    bnc = abundance().create()
    assert isinstance(bnc, BinnedClusterNumberCounts)
    assert bnc.cluster_properties == ClusterProperty.COUNTS | ClusterProperty.MASS
    assert bnc.survey_name == "my_survey"
    theory = bnc.cluster_recipe.cluster_theory
    assert isinstance(theory, ClusterAbundance)
    assert not isinstance(theory, ClusterShearProfile)
    assert parameter_names(bnc) == MURATA_NAMES


def test_abundance_observables() -> None:
    """Observables come from the factory or the call; unknown ones fail."""
    factory = abundance(observables=None)
    with pytest.raises(ValueError, match="No observables to predict"):
        factory.create()
    assert factory.create(["counts"]).cluster_properties == ClusterProperty.COUNTS
    with pytest.raises(ValueError, match="Unknown abundance observable"):
        factory.create(["richness"])
    with pytest.raises(ValueError):
        abundance(observables=[])
    with pytest.raises(ValueError):
        abundance(observables=["deltasigma"])


@pytest.mark.parametrize(
    "purity", [PurityAguena16Factory(), PurityAguena16LnProxyFactory()]
)
def test_abundance_selection_models(
    purity: PurityAguena16Factory | PurityAguena16LnProxyFactory,
) -> None:
    """Completeness and purity register their sampler parameters."""
    model = exact_model(completeness=CompletenessAguena16Factory(), purity=purity)
    names = parameter_names(abundance(model=model).create())
    assert {"completeness_a_n", "purity_a_n"} <= names


def test_parameter_prefix_separates_surveys() -> None:
    """A parameter prefix gives each sample its own parameters."""
    rm = abundance(model=exact_model(parameter_prefix="rm")).create()
    assert parameter_names(rm) == {f"rm_{name}" for name in MURATA_NAMES}


def test_abundance_from_sacc(cluster_sacc_data: sacc.Sacc) -> None:
    """The abundance statistic reads its data, inferring the observables."""
    bnc = abundance().from_sacc(cluster_sacc_data)
    assert bnc.ready
    assert bnc.sky_area == 4000
    assert len(bnc.get_data_vector()) == 4
    inferred = abundance(observables=None).from_sacc(cluster_sacc_data)
    assert inferred.cluster_properties == ClusterProperty.COUNTS | ClusterProperty.MASS
    with pytest.raises(ValueError, match="no cluster counts or mean masses"):
        abundance(survey_name="absent", observables=None).from_sacc(cluster_sacc_data)


# Lensing factory


def test_lensing_create_delta_sigma() -> None:
    """The lensing statistic uses a ClusterShearProfile theory."""
    bnd = lensing().create()
    assert isinstance(bnd, BinnedClusterShearProfile)
    assert bnd.cluster_properties == ClusterProperty.DELTASIGMA
    assert isinstance(bnd.cluster_recipe, GridBinnedClusterRecipe)
    theory = bnd.cluster_recipe.cluster_theory
    assert isinstance(theory, ClusterShearProfile)
    assert theory.is_delta_sigma
    assert isinstance(bnd.cluster_recipe.mass_distribution, mass_proxy.MurataUnbinned)
    assert parameter_names(bnd) == MURATA_NAMES | {
        "cluster_theory_cluster_concentration"
    }


def test_lensing_create_reduced_shear() -> None:
    """The reduced shear uses the lensing efficiency options."""
    factory = lensing(
        observable="shear",
        lensing_efficiency=LensingEfficiencyOptions(
            z_max=5.0, approx="order2", interpolate=True
        ),
    )
    theory = factory.create().cluster_recipe.cluster_theory
    assert isinstance(theory, ClusterShearProfile)
    assert not theory.is_delta_sigma
    assert theory.use_beta_s_interp
    assert theory.approx == "order2"


def test_lensing_observable_checks() -> None:
    """The observable must be set and compatible with the options."""
    with pytest.raises(ValueError, match="two-halo term"):
        lensing(observable="shear", two_halo_term=True)
    factory = lensing(observable=None, two_halo_term=True)
    with pytest.raises(ValueError, match="two-halo term"):
        factory.create("shear")
    with pytest.raises(ValueError, match="Unknown lensing observable"):
        factory.create("kappa")
    with pytest.raises(ValueError, match="No observable to predict"):
        factory.create()


def test_grid_recipe_warns_for_concentration_mass_relation() -> None:
    """The grid recipe cannot use crow's concentration-mass relation."""
    factory = lensing(cluster_concentration=None)
    assert factory.uses_concentration_mass_relation
    with pytest.warns(UserWarning, match="concentration-mass relation does not"):
        factory.create()


def test_exact_recipe_allows_concentration_mass_relation() -> None:
    """The exact recipe uses the concentration-mass relation without warning."""
    factory = lensing(cluster_concentration=None, model=exact_model())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bnd = factory.create()
    assert isinstance(bnd.cluster_recipe, ExactBinnedClusterRecipe)
    assert not [w for w in caught if "relation does not" in str(w.message)]


def test_lensing_from_sacc(cluster_sacc_data: sacc.Sacc) -> None:
    """The lensing statistic reads its data; the observable must be unique."""
    bnd = lensing().from_sacc(cluster_sacc_data)
    assert bnd.ready
    assert len(bnd.get_data_vector()) == 2
    with pytest.raises(ValueError, match="set 'observable' to choose one"):
        lensing(observable=None).from_sacc(cluster_sacc_data)
    shear_only = cluster_sacc_data.copy()
    # pylint: disable-next=no-member
    shear_only.remove_selection(data_type=sacc.standard_types.cluster_delta_sigma)
    inferred = lensing(observable=None).from_sacc(shear_only)
    assert inferred.cluster_properties == ClusterProperty.SHEAR
    with pytest.raises(ValueError, match="no cluster lensing profiles"):
        lensing(survey_name="absent", observable=None).from_sacc(cluster_sacc_data)


# BinnedClusterFactory


def test_binned_cluster_factory_validation() -> None:
    """Factories must exist, be unique per survey and share the sample model."""
    with pytest.raises(ValueError, match="at least one abundance or lensing"):
        BinnedClusterFactory()
    with pytest.raises(ValueError, match="Duplicate abundance factories"):
        BinnedClusterFactory(abundance_factories=[abundance(), abundance()])
    with pytest.raises(ValueError, match="must use the same model"):
        BinnedClusterFactory(
            abundance_factories=[abundance()], lensing_factories=[lensing()]
        )
    BinnedClusterFactory(
        abundance_factories=[abundance(model=grid_model())],
        lensing_factories=[lensing()],
    )


def test_binned_cluster_factory_get_factory() -> None:
    """Factories are found by SACC data type and survey."""
    counts, profile = abundance(model=grid_model()), lensing()
    factory = BinnedClusterFactory(
        abundance_factories=[counts], lensing_factories=[profile]
    )
    # pylint: disable=no-member
    assert factory.get_factory(sacc.standard_types.cluster_counts, "my_survey") == (
        counts
    )
    assert factory.get_factory(sacc.standard_types.cluster_shear, "my_survey") == (
        profile
    )
    with pytest.raises(ValueError, match="No factory for"):
        factory.get_factory(sacc.standard_types.cluster_counts, "absent")
    # pylint: enable=no-member
    with pytest.raises(ValueError, match="No cluster factory predicts"):
        factory.get_factory("galaxy_shear_cl_ee", "my_survey")


def test_binned_cluster_factory_from_sacc(cluster_sacc_data: sacc.Sacc) -> None:
    """Counts and lensing of one sample share the model parameters."""
    factory = BinnedClusterFactory(
        abundance_factories=[abundance(observables=["counts"], model=grid_model())],
        lensing_factories=[lensing()],
    )
    bnc, bnd = factory.from_sacc(cluster_sacc_data)
    assert isinstance(bnc, BinnedClusterNumberCounts)
    assert isinstance(bnd, BinnedClusterShearProfile)
    assert bnc.cluster_recipe is not bnd.cluster_recipe
    assert parameter_names(bnc) == MURATA_NAMES


def test_binned_cluster_factory_two_surveys(cluster_sacc_data: sacc.Sacc) -> None:
    """Two samples with prefixes get independent parameters."""
    factory = BinnedClusterFactory(
        abundance_factories=[
            abundance(model=exact_model(parameter_prefix="a")),
            abundance(
                survey_name="not_my_survey", model=exact_model(parameter_prefix="b")
            ),
        ]
    )
    first, second = factory.from_sacc(cluster_sacc_data)
    assert second.survey_name == "not_my_survey"
    assert parameter_names(first).isdisjoint(parameter_names(second))


def test_binned_cluster_factory_yaml() -> None:
    """The factory reads YAML with a shared model anchor and round-trips."""
    factory = base_model_from_yaml(BinnedClusterFactory, CLUSTER_FACTORY_YAML)
    assert factory.abundance_factories[0].model == factory.lensing_factories[0].model
    restored = base_model_from_yaml(BinnedClusterFactory, base_model_to_yaml(factory))
    assert restored == factory


# Experiments and likelihood builders


def test_cluster_experiment_counts(
    cluster_sacc_data: sacc.Sacc, tmp_path: Path
) -> None:
    """A counts and mean mass likelihood evaluates to a finite chi-squared."""
    experiment = cluster_experiment(
        cluster_sacc_data,
        tmp_path,
        BinnedClusterFactory(abundance_factories=[abundance()]),
    )
    likelihood = experiment.make_likelihood()
    assert isinstance(likelihood, ConstGaussian)
    assert len(likelihood.get_data_vector()) == 4
    tools = ModelingTools(ccl_factory=experiment.ccl_factory)
    assert np.isfinite(evaluate(likelihood, tools))


def test_cluster_experiment_counts_and_delta_sigma(
    cluster_sacc_data: sacc.Sacc, tmp_path: Path
) -> None:
    """Counts and lensing statistics with shared parameters evaluate together."""
    factory = base_model_from_yaml(BinnedClusterFactory, CLUSTER_FACTORY_YAML)
    experiment = cluster_experiment(cluster_sacc_data, tmp_path, factory)
    likelihood = experiment.make_likelihood()
    assert isinstance(likelihood, ConstGaussian)
    assert len(likelihood.get_data_vector()) == 4
    assert set(likelihood.required_parameters().get_params_names()) == (
        MURATA_NAMES | {"cluster_theory_cluster_concentration"}
    )
    tools = ModelingTools(ccl_factory=experiment.ccl_factory)
    assert np.isfinite(evaluate(likelihood, tools))


def test_cluster_experiment_reduced_shear(
    cluster_sacc_data: sacc.Sacc, tmp_path: Path
) -> None:
    """A reduced shear likelihood evaluates to a finite chi-squared."""
    profile = lensing(
        observable="shear",
        lensing_efficiency=LensingEfficiencyOptions(z_max=5.0, interpolate=True),
    )
    experiment = cluster_experiment(
        cluster_sacc_data, tmp_path, BinnedClusterFactory(lensing_factories=[profile])
    )
    likelihood = experiment.make_likelihood()
    assert isinstance(likelihood, ConstGaussian)
    assert len(likelihood.get_data_vector()) == 2
    tools = ModelingTools(ccl_factory=experiment.ccl_factory)
    assert np.isfinite(evaluate(likelihood, tools))


def test_cluster_experiment_errors(
    cluster_sacc_data: sacc.Sacc, tmp_path: Path
) -> None:
    """The experiment needs a covariance and rejects two-point filters."""
    sacc_path = tmp_path / "no_covariance.fits"
    cluster_sacc_data.save_fits(str(sacc_path), overwrite=True)
    factory = BinnedClusterFactory(abundance_factories=[abundance()])
    experiment = ClusterExperiment(
        cluster_factory=factory,
        data_source=DataSourceSacc(sacc_data_file=str(sacc_path)),
    )
    with pytest.raises(ValueError, match="no covariance"):
        experiment.make_likelihood()
    filtered = ClusterExperiment(
        cluster_factory=factory,
        data_source=DataSourceSacc(
            sacc_data_file=str(sacc_path),
            filters=TwoPointBinFilterCollection(
                require_filter_for_all=False, allow_empty=True, filters=[]
            ),
        ),
    )
    with pytest.raises(ValueError, match="filters cannot be applied"):
        filtered.make_likelihood()


def test_build_cluster_likelihood(cluster_sacc_data: sacc.Sacc, tmp_path: Path) -> None:
    """The builder reads a cluster experiment from YAML."""
    with_covariance(cluster_sacc_data).save_fits(
        str(tmp_path / "clusters.fits"), overwrite=True
    )
    experiment_file = tmp_path / "experiment.yaml"
    experiment_file.write_text(CLUSTER_EXPERIMENT_YAML, encoding="utf-8")
    likelihood, tools = build_cluster_likelihood(
        NamedParameters({"likelihood_config": str(experiment_file)})
    )
    assert isinstance(likelihood, ConstGaussian)
    assert isinstance(tools, ModelingTools)
    assert set(likelihood.required_parameters().get_params_names()) == MURATA_NAMES


def test_two_point_experiment_with_clusters(
    sacc_galaxy_cells: tuple[sacc.Sacc, dict[str, Any], dict[str, Any]],
    cluster_sacc_data: sacc.Sacc,
    tmp_path: Path,
) -> None:
    """Cluster statistics join the two-point statistics of one likelihood."""
    two_point_data, _, _ = sacc_galaxy_cells
    joint = two_point_data.copy()
    joint.covariance = None
    for tracer in cluster_sacc_data.tracers.values():
        joint.add_tracer_object(tracer)
    for point in cluster_sacc_data.data:
        joint.add_data_point(point.data_type, point.tracers, point.value, **point.tags)
    joint.add_covariance(np.eye(len(joint.mean)))
    sacc_path = tmp_path / "joint.fits"
    joint.save_fits(str(sacc_path), overwrite=True)

    experiment = TwoPointExperiment(
        two_point_factory=TwoPointFactory(
            correlation_space=TwoPointCorrelationSpace.HARMONIC,
            weak_lensing_factories=[
                WeakLensingFactory(per_bin_systematics=[], global_systematics=[])
            ],
            number_counts_factories=[
                NumberCountsFactory(per_bin_systematics=[], global_systematics=[])
            ],
        ),
        data_source=DataSourceSacc(sacc_data_file=str(sacc_path)),
        cluster_factory=BinnedClusterFactory(abundance_factories=[abundance()]),
    )
    likelihood = experiment.make_likelihood()
    assert isinstance(likelihood, ConstGaussian)
    assert len(likelihood.get_data_vector()) == len(two_point_data.mean) + 4
    assert MURATA_NAMES <= set(likelihood.required_parameters().get_params_names())
