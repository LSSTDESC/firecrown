"""NumCosmo configuration file generation utilities.

Provides functions to create NumCosmo YAML configuration files
for cosmological parameter estimation with Firecrown likelihoods.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

from pathlib import Path
import os
import dataclasses

from numcosmo_py import Ncm, Nc
import numcosmo_py.external.cosmosis as nc_cosmosis
from numcosmo_py.helper import register_model_class
from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory
from firecrown.likelihood.likelihood import NamedParameters
from ._types import Model, Frameworks, ConfigGenerator, FrameworkCosmology


def _create_mapping(
    distance_max_z: float, reltol: float, required_cosmology: FrameworkCosmology
) -> nc_cosmosis.MappingNumCosmo | None:
    """Create NumCosmo mapping configuration."""
    match required_cosmology:
        case FrameworkCosmology.NONE:
            return nc_cosmosis.MappingNumCosmo()
        case FrameworkCosmology.BACKGROUND:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.NONE,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.NONE,
                distance_max_z=distance_max_z,
                reltol=reltol,
            )
        case FrameworkCosmology.LINEAR:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.CLASS,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.NONE,
                distance_max_z=distance_max_z,
                reltol=reltol,
            )
        case FrameworkCosmology.NONLINEAR:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.CLASS,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.HALOFIT,
                distance_max_z=distance_max_z,
                reltol=reltol,
            )
        case _:
            raise ValueError(f"Unsupported cosmology: {required_cosmology}")


def _create_factory(
    factory_path: Path,
    factory_filename: str,
    build_parameters: NamedParameters,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    model_list: list[str],
) -> NumCosmoFactory:
    """Create NumCosmo factory instance."""
    previous_dir = os.getcwd()
    os.chdir(factory_path.parent)

    numcosmo_factory = NumCosmoFactory(
        factory_filename,
        build_parameters,
        mapping=mapping,
        model_list=model_list,
    )

    os.chdir(previous_dir)
    return numcosmo_factory


def _setup_model_set() -> Ncm.MSet:
    """Create and configure model set."""
    mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter

    cosmo = Nc.HICosmoDECpl()
    cosmo.omega_x2omega_k()
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)
    mset.set(cosmo)

    return mset


def _setup_dataset(numcosmo_factory: NumCosmoFactory) -> Ncm.Dataset:
    """Create and configure dataset."""
    dataset = Ncm.Dataset.new()  # pylint: disable=no-value-for-parameter
    firecrown_data = numcosmo_factory.get_data()
    if isinstance(firecrown_data, Ncm.DataGaussCov):
        firecrown_data.set_size(0)
    dataset.append_data(firecrown_data)
    return dataset


def create_config(
    factory_path: Path,
    build_parameters: NamedParameters,
    model_list: list[str],
    use_absolute_path: bool = False,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
    distance_max_z: float = 4.0,
    reltol: float = 1e-7,
) -> Ncm.ObjDictStr:
    """Create standard NumCosmo experiment configuration.

    :param factory_path: Path to factory file
    :param build_parameters: Likelihood build parameters
    :param model_list: List of model names
    :param use_absolute_path: Use absolute paths
    :param use_cosmology: Include CLASS computation
    :param distance_max_z: Maximum redshift for distance calculations
    :param reltol: Relative tolerance for calculations
    :return: NumCosmo experiment object
    """
    experiment = Ncm.ObjDictStr()
    factory_filename = (
        factory_path.absolute().as_posix() if use_absolute_path else factory_path.name
    )

    mapping = _create_mapping(distance_max_z, reltol, required_cosmology)
    numcosmo_factory = _create_factory(
        factory_path, factory_filename, build_parameters, mapping, model_list
    )

    mset = _setup_model_set()
    dataset = _setup_dataset(numcosmo_factory)

    likelihood = Ncm.Likelihood.new(dataset)
    experiment.add("likelihood", likelihood)
    experiment.add("model-set", mset)

    return experiment


def add_models(
    config: Ncm.ObjDictStr,
    models: list[Model],
) -> Ncm.ObjDictStr:
    """Add model parameters to NumCosmo configuration.

    :param config: NumCosmo experiment object (modified in-place)
    :param models: List of models with parameters
    :return: Model builders object
    """
    mset = config.get("model-set")
    assert isinstance(mset, Ncm.MSet)
    model_builders = Ncm.ObjDictStr.new()  # pylint: disable=no-value-for-parameter

    for model in models:
        model_builder = Ncm.ModelBuilder.new(Ncm.Model, model.name, model.description)
        for parameter in model.parameters:
            model_builder.add_sparam(
                parameter.symbol,
                parameter.name,
                parameter.lower_bound,
                parameter.upper_bound,
                parameter.scale,
                parameter.abstol,
                parameter.default_value,
                Ncm.ParamType.FREE if parameter.free else Ncm.ParamType.FIXED,
            )

        model_builders.add(model.name, model_builder)
        FirecrownModel = register_model_class(model_builder)
        mset.set(FirecrownModel())

    return model_builders


@dataclasses.dataclass
class NumCosmoConfigGenerator(ConfigGenerator):
    """Generates NumCosmo YAML configuration files.

    Creates two files:
    - numcosmo_{prefix}.yaml: Experiment configuration
    - numcosmo_{prefix}.builders.yaml: Model builders
    """

    framework = Frameworks.NUMCOSMO

    def write_config(self) -> None:
        """Write NumCosmo configuration."""
        assert self.factory_path is not None
        assert self.build_parameters is not None

        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        model_list = [model.name for model in self.models]

        config = create_config(
            factory_path=self.factory_path,
            build_parameters=self.build_parameters,
            model_list=model_list,
            use_absolute_path=self.use_absolute_path,
            required_cosmology=self.required_cosmology,
        )
        model_builders = add_models(config, self.models)

        numcosmo_yaml = self.output_path / f"numcosmo_{self.prefix}.yaml"
        builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

        ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
        ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
        ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())
