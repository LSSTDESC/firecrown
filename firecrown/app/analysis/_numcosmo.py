"""NumCosmo configuration file generator.

Generates NumCosmo YAML files for cosmological parameter estimation with Firecrown.
Produces experiment configuration and model builder files using NumCosmo's
serialization.

NumCosmo uses CLASS for cosmology computation and supports functional priors (e.g., σ8
as derived parameter). Configuration must be written in a fresh subprocess to avoid
GType conflicts. Parameters are scaled (h → H0 × 100) and A_s is stored as ln(10¹⁰ ×
A_s).

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import multiprocessing as mp
import re
from typing import assert_never, Any
from pathlib import Path
import os
import dataclasses
import numpy as np
from numcosmo_py import Ncm, Nc
import numcosmo_py.external.cosmosis as nc_cosmosis
from numcosmo_py.helper import register_model_class
from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory
from firecrown.likelihood import NamedParameters
from ._types import (
    Model,
    Parameter,
    Frameworks,
    ConfigGenerator,
    FrameworkCosmology,
    CCLCosmologySpec,
    Prior,
    PriorGaussian,
    PriorUniform,
    get_path_str,
)

# Map CCL parameter names to NumCosmo parameter names
NAME_MAP: dict[str, str] = {
    "Omega_c": "Omegac",
    "Omega_b": "Omegab",
    "Omega_k": "Omegak",
    "h": "H0",  # h * 100
    "w0": "w0",
    "wa": "w1",
    "n_s": "n_SA",
    "T_CMB": "Tgamma0",
    "Neff": "ENnu",
    "m_nu": "mnu_0",
}


@dataclasses.dataclass
class ConfigOptions:
    """Configuration options for NumCosmo experiment setup.

    Bundles all configuration parameters needed to set up a NumCosmo experiment,
    simplifying function signatures throughout the module.

    :ivar output_path: Output directory for YAML files and temporary data
    :ivar factory_source: Firecrown factory YAML path or module identifier
    :ivar build_parameters: Parameters passed to likelihood factory
    :ivar models: List of systematic/nuisance models to configure
    :ivar cosmo_spec: Cosmology specification with parameters and priors
    :ivar use_absolute_path: Whether to use absolute paths in configuration files
    :ivar required_cosmology: Level of cosmology computation
        (NONE/BACKGROUND/LINEAR/NONLINEAR)
    :ivar prefix: Filename prefix for generated YAML files
    :ivar distance_max_z: Maximum redshift for distance integrals (default: 4.0)
    :ivar reltol: Relative tolerance for numerical calculations (default: 1e-7)
    """

    output_path: Path
    factory_source: str | Path
    build_parameters: NamedParameters
    models: list[Model]
    cosmo_spec: CCLCosmologySpec
    use_absolute_path: bool
    required_cosmology: FrameworkCosmology
    prefix: str
    distance_max_z: float = 4.0
    reltol: float = 1e-7


def _create_mapping(options: ConfigOptions) -> nc_cosmosis.MappingNumCosmo | None:
    """Create NumCosmo cosmology power spectrum mapping.

    Configures NumCosmo computation level (background, linear power spectrum, or
    nonlinear with HaloFit) based on required cosmology. Returns None if no cosmology
    computation is needed.

    :param options: Configuration options containing required_cosmology,
        distance_max_z, and reltol
    :return: Configured NumCosmo mapping or None if computation not needed
    """
    match options.required_cosmology:
        case FrameworkCosmology.NONE:
            return None
        case FrameworkCosmology.BACKGROUND:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.NONE,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.NONE,
                distance_max_z=options.distance_max_z,
                reltol=options.reltol,
            )
        case FrameworkCosmology.LINEAR:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.CLASS,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.NONE,
                distance_max_z=options.distance_max_z,
                reltol=options.reltol,
            )
        case FrameworkCosmology.NONLINEAR:
            return nc_cosmosis.create_numcosmo_mapping(
                matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.CLASS,
                nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.HALOFIT,
                distance_max_z=options.distance_max_z,
                reltol=options.reltol,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _create_factory(
    output_path: Path,
    factory_source_str: str,
    build_parameters: NamedParameters,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    model_list: list[str],
) -> NumCosmoFactory:
    """Create NumCosmo factory instance for likelihood computation.

    Initializes the Firecrown likelihood factory in the output directory context.
    Temporarily changes working directory for factory initialization, then restores the
    original directory to avoid side effects.

    :param output_path: Output directory for factory initialization
    :param factory_source_str: Factory source YAML path or module identifier string
    :param build_parameters: Parameters passed to likelihood factory builder
    :param mapping: NumCosmo CLASS cosmology mapping (or None for no cosmology)
    :param model_list: List of model names to register with factory
    :return: Initialized NumCosmo factory instance
    """
    cwd = os.getcwd()
    try:
        os.chdir(output_path)
        return NumCosmoFactory(
            factory_source_str, build_parameters, mapping=mapping, model_list=model_list
        )
    finally:
        os.chdir(cwd)


def _add_prior(
    priors: list[Ncm.Prior],
    param_name: str,
    prior: Prior | None,
    param_scale: dict[str, float] | None = None,
) -> None:
    """Add prior constraint to priors list if specified.

    Creates appropriate NumCosmo prior object (Gaussian or uniform) and appends it to
    the priors list. Does nothing if prior is None. Applies optional scaling to prior
    bounds and values.

    :param priors: List of prior objects (modified in-place)
    :param param_name: Full parameter name with namespace (e.g., 'NcHICosmo:H0')
    :param prior: Prior specification (Gaussian or uniform)
    :param param_scale: Optional scale factors indexed by parameter name
    """
    if prior is None:
        return
    param_scale = param_scale or {}
    scale = param_scale.get(param_name, 1.0)

    match prior:
        case PriorGaussian():
            priors.append(
                Ncm.PriorGaussParam.new_name(
                    param_name, prior.mean * scale, prior.sigma * scale
                )
            )
        case PriorUniform():
            priors.append(
                Ncm.PriorFlatParam.new_name(
                    param_name, prior.lower * scale, prior.upper * scale, 1.0e-3
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _param_to_nc_dict(
    param: Parameter, param_scale: dict[str, float] | None = None
) -> dict[str, Any]:
    """Convert Firecrown parameter to NumCosmo parameter descriptor dictionary.

    Transforms parameter specification to NumCosmo format with field name mapping
    (e.g., 'lower_bound' → 'lower-bound') and applies scaling. Excludes derived fields
    like 'name', 'symbol', and 'prior'.

    :param param: Parameter specification with bounds, default, and flags
    :param param_scale: Optional scale factors indexed by parameter name
    :return: Parameter configuration dictionary for NumCosmo
    """
    param_scale = param_scale or {}
    scale = param_scale.get(param.name, 1.0)
    rename = {
        "lower_bound": "lower-bound",
        "upper_bound": "upper-bound",
        "default_value": "value",
        "free": "fit",
    }
    nc_param = {
        rename.get(key, key): value * scale if isinstance(value, float) else value
        for key, value in param.model_dump().items()
        if key not in ["name", "symbol", "prior"]
    }
    return nc_param


def _set_standard_params(
    options: ConfigOptions,
    mset: Ncm.MSet,
    cosmo: Nc.HICosmoDECpl,
    prim: Nc.HIPrimPowerLaw,
    reion: Nc.HIReionCamb,
    priors: list[Ncm.Prior],
) -> None:
    """Set standard cosmological parameters with priors in NumCosmo model set.

    Configures standard LCDM and extended parameters (Omega_c, Omega_b, h, n_s, Neff,
    T_CMB, etc.) with their values and optional prior constraints. Applies scaling
    where needed (h → H0 x 100). Skips special amplitude and neutrino mass parameters
    (A_s, sigma8, m_nu) which are handled separately.

    :param options: Configuration options containing cosmology specification
    :param mset: NumCosmo model set (modified in-place)
    :param cosmo: Cosmology model (HICosmoDECpl)
    :param prim: Primordial power spectrum model (HIPrimPowerLaw)
    :param reion: Reionization model (HIReionCamb)
    :param priors: Prior list (modified in-place)
    :raises ValueError: If parameter name is not recognized in NAME_MAP
    """
    targets: list[Ncm.Model] = [cosmo, prim, reion]
    skip_list = ["A_s", "sigma8", "m_nu"]
    param_scale = {"h": 100.0}
    for param in options.cosmo_spec.parameters:
        if param.name in skip_list:
            continue
        nc_name = NAME_MAP[param.name]
        for model in targets:
            if nc_name in model.param_names():
                model[nc_name] = param.default_value
                model.param_set_desc(nc_name, _param_to_nc_dict(param, param_scale))
                if param.prior is not None:
                    _add_prior(
                        priors,
                        f"{mset.get_ns_by_id(model.id())}:{nc_name}",
                        param.prior,
                        param_scale,
                    )
                break
        else:
            raise ValueError(f"Unknown parameter {param.name}")


def _set_amplitude_A_s(
    options: ConfigOptions,
    mset: Ncm.MSet,
    prim: Nc.HIPrimPowerLaw,
    priors: list[Ncm.Prior],
) -> None:
    """Set A_s amplitude parameter with logarithmic transformation and priors.

    NumCosmo stores A_s as ln(10^10 x A_s) for numerical stability. Prior bounds are
    transformed accordingly using logarithmic scaling. Does nothing if A_s is not
    specified in the cosmology.

    :param options: Configuration options containing cosmology specification
    :param mset: NumCosmo model set
    :param prim: Primordial power spectrum model (modified in-place)
    :param priors: Prior list (modified in-place if A_s prior exists)
    """
    if "A_s" not in options.cosmo_spec:
        return

    A_s_param = options.cosmo_spec["A_s"]
    A_s = A_s_param.default_value
    prim["ln10e10ASA"] = np.log(1.0e10 * A_s)
    if A_s_param.prior is None:
        return

    param_name = f"{mset.get_ns_by_id(prim.id())}:ln10e10ASA"
    match A_s_param.prior:
        case PriorGaussian():
            priors.append(
                Ncm.PriorGaussParam.new_name(
                    param_name,
                    np.log(1.0e10 * A_s_param.prior.mean),
                    A_s_param.prior.sigma / A_s,
                )
            )
        case PriorUniform():
            priors.append(
                Ncm.PriorFlatParam.new_name(
                    param_name,
                    np.log(1.0e10 * A_s_param.prior.lower),
                    np.log(1.0e10 * A_s_param.prior.upper),
                    1.0e-3,
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _set_amplitude_sigma8(
    options: ConfigOptions,
    cosmo: Nc.HICosmoDECpl,
    prim: Nc.HIPrimPowerLaw,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    priors: list[Ncm.Prior],
) -> None:
    """Set sigma8 amplitude parameter as functional prior.

    Computes A_s from target sigma8 value by computing sigma8(A_s=1) and scaling
    accordingly. Sets up functional prior constraint that applies directly to the
    derived sigma8 quantity. This allows sampling in A_s parameter space while
    constraining the final sigma8 value. Does nothing if sigma8 not specified or if
    mapping unavailable.

    :param options: Configuration options containing cosmology specification and
        tolerances
    :param cosmo: Cosmology model (modified in-place for A_s computation)
    :param prim: Primordial power spectrum model (modified in-place)
    :param mapping: NumCosmo mapping with power spectrum (required if sigma8 present)
    :param priors: Prior list (modified in-place if sigma8 prior exists)
    :raises ValueError: If sigma8 specified but mapping or p_ml unavailable
    """
    if "sigma8" not in options.cosmo_spec:
        return

    if mapping is None or mapping.p_ml is None:
        raise ValueError("Mapping must have p_ml set for sigma8")

    sigma8_param = options.cosmo_spec["sigma8"]
    sigma8 = sigma8_param.default_value

    A_s = np.exp(prim["ln10e10ASA"]) * 1.0e-10
    fact = (
        sigma8
        / mapping.p_ml.sigma_tophat_R(cosmo, options.reltol, 0.0, 8.0 / cosmo.h())
    ) ** 2
    prim.param_set_by_name("ln10e10ASA", np.log(1.0e10 * A_s * fact))

    if sigma8_param.prior is None:
        return

    psf = Ncm.PowspecFilter.new(mapping.p_ml, Ncm.PowspecFilterType.TOPHAT)
    sigma8_func = Ncm.MSetFuncList.new("NcHICosmo:sigma8", psf)

    match sigma8_param.prior:
        case PriorGaussian():
            priors.append(
                Ncm.PriorGaussFunc.new(
                    sigma8_func, sigma8_param.prior.mean, sigma8_param.prior.sigma, 0.0
                )
            )
        case PriorUniform():
            priors.append(
                Ncm.PriorFlatFunc.new(
                    sigma8_func,
                    sigma8_param.prior.lower,
                    sigma8_param.prior.upper,
                    1.0e-3,
                    0.0,
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _set_neutrino_masses(
    options: ConfigOptions,
    cosmo: Nc.HICosmoDECpl,
    mset: Ncm.MSet,
    priors: list[Ncm.Prior],
) -> None:
    """Set neutrino mass parameter with optional prior constraint.

    Configures massive neutrino mass m_nu in the cosmology model and adds any
    associated prior constraints. Does nothing if no massive neutrinos are present.

    :param options: Configuration options containing cosmology specification
    :param cosmo: Cosmology model (modified in-place)
    :param mset: NumCosmo model set (used for namespace resolution)
    :param priors: Prior list (modified in-place if m_nu prior exists)
    """
    if options.cosmo_spec.get_num_massive_neutrinos() == 0:
        return

    nc_name = NAME_MAP["m_nu"]
    param = options.cosmo_spec["m_nu"]
    assert nc_name in cosmo.param_names()
    cosmo[nc_name] = param.default_value
    cosmo.param_set_desc(nc_name, _param_to_nc_dict(param))
    if param.prior is not None:
        _add_prior(
            priors,
            f"{mset.get_ns_by_id(cosmo.id())}:{nc_name}",
            param.prior,
        )


def _setup_cosmology(
    options: ConfigOptions,
    mset: Ncm.MSet,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    priors: list[Ncm.Prior],
) -> None:
    """Set up cosmology models and parameters in NumCosmo model set.

    Creates and configures cosmology model (HICosmoDECpl), primordial power spectrum
    (HIPrimPowerLaw), and reionization (HIReionCamb) with all parameters and
    constraints. Handles standard parameters (Omega_c, Omega_b, etc.), amplitude
    parameters (A_s, sigma8), and neutrino masses. Does nothing if required_cosmology
    is NONE.

    :param options: Configuration options containing cosmology specification and
        settings
    :param mset: NumCosmo model set (modified in-place)
    :param mapping: NumCosmo CLASS mapping (required if computation needed)
    :param priors: Prior list (modified in-place)
    """
    if options.required_cosmology == FrameworkCosmology.NONE:
        return

    assert mapping is not None

    cosmo = Nc.HICosmoDECpl(
        massnu_length=options.cosmo_spec.get_num_massive_neutrinos()
    )
    cosmo.omega_x2omega_k()
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)
    mset.set(cosmo)

    _set_standard_params(options, mset, cosmo, prim, reion, priors)
    _set_amplitude_A_s(options, mset, prim, priors)
    _set_amplitude_sigma8(options, cosmo, prim, mapping, priors)
    _set_neutrino_masses(options, cosmo, mset, priors)


def _to_pascal(s: str) -> str:
    """Convert string to PascalCase for NumCosmo model names.

    Splits on non-alphanumeric characters and capitalizes first letter of each part
    while preserving case of remaining characters.

    Examples:
        - 'intrinsic_alignment' → 'IntrinsicAlignment'
        - 'ia_model_v1' → 'IaModelV1'
        - 'shear_bias_multiplicative' → 'ShearBiasMultiplicative'

    :param s: Input string (typically model or parameter name)
    :return: PascalCase formatted string
    """
    parts = re.split(r"[^A-Za-z0-9]+", s)
    return "".join(p[0].upper() + p[1:] for p in parts if p)


def _setup_models(
    options: ConfigOptions,
    mset: Ncm.MSet,
    priors: list[Ncm.Prior],
) -> Ncm.ObjDictStr:
    """Add systematic/nuisance models to NumCosmo model set.

    Creates NumCosmo ModelBuilder objects for each model, registers them dynamically,
    instantiates them, and adds to the model set. Also configures any parameter priors
    for the models.

    :param options: Configuration options containing models list
    :param mset: NumCosmo model set (modified in-place)
    :param priors: Prior list (modified in-place for model parameter priors)
    :return: ObjDictStr mapping model names to builder objects
    """
    model_builders = Ncm.ObjDictStr.new()  # pylint: disable=no-value-for-parameter

    for model in options.models:
        model_name = _to_pascal(model.name)
        model_builder = Ncm.ModelBuilder.new(Ncm.Model, model_name, model.description)
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
        model_builders.add(model_name, model_builder)
        FirecrownModel = register_model_class(model_builder)
        fc_model = FirecrownModel()
        mset.set(fc_model)
        for parameter in model.parameters:
            if parameter.prior is not None:
                _add_prior(
                    priors,
                    f"{mset.get_ns_by_id(fc_model.id())}:{parameter.name}",
                    parameter.prior,
                )

    return model_builders


def _create_dataset(numcosmo_factory: NumCosmoFactory) -> Ncm.Dataset:
    """Create NumCosmo dataset from Firecrown factory data.

    Wraps the likelihood data object from the Firecrown factory in a NumCosmo dataset
    container for use in likelihood calculations.

    :param numcosmo_factory: Firecrown factory with likelihood data
    :return: NumCosmo dataset wrapper
    """
    dataset = Ncm.Dataset.new()  # pylint: disable=no-value-for-parameter
    firecrown_data = numcosmo_factory.get_data()
    if isinstance(firecrown_data, Ncm.DataGaussCov):
        firecrown_data.set_size(0)
    dataset.append_data(firecrown_data)
    return dataset


def _create_likelihood(dataset: Ncm.Dataset, priors: list[Ncm.Prior]) -> Ncm.Likelihood:
    """Create NumCosmo likelihood combining data and prior constraints.

    Wraps dataset in a likelihood object and adds all prior constraints for use in
    parameter estimation.

    :param dataset: NumCosmo dataset with observations
    :param priors: Prior constraints on parameters
    :return: NumCosmo likelihood combining data and priors
    """
    likelihood = Ncm.Likelihood.new(dataset)
    for prior in priors:
        likelihood.priors_add(prior)
    return likelihood


def _setup_experiment(options: ConfigOptions) -> tuple[Ncm.ObjDictStr, Ncm.ObjDictStr]:
    """Set up complete NumCosmo experiment with cosmology, models, and likelihood.

    Creates factory, model set with cosmology and systematic models, dataset, and
    likelihood. Returns serializable objects for saving to YAML.

    All configuration parameters are provided via the ConfigOptions dataclass, which
    bundles them for cleaner function signatures.

    :param options: Configuration options with all necessary parameters
    :return: Tuple of (experiment config dict, model builders dict)
    """
    mapping = _create_mapping(options)
    factory = _create_factory(
        options.output_path,
        get_path_str(options.factory_source, options.use_absolute_path),
        options.build_parameters,
        mapping,
        [_to_pascal(model.name) for model in options.models],
    )
    priors: list[Ncm.Prior] = []
    mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
    _setup_cosmology(options, mset, mapping, priors)
    model_builders = _setup_models(options, mset, priors)

    likelihood = _create_likelihood(_create_dataset(factory), priors)
    experiment = Ncm.ObjDictStr()
    experiment.add("likelihood", likelihood)
    experiment.add("model-set", mset)

    return experiment, model_builders


def _write_config_worker(options: ConfigOptions) -> int:
    """Worker function executed in fresh subprocess for YAML serialization.

    Runs in isolated process to avoid GType conflicts with NumCosmo's type system.
    Creates configuration and writes YAML files using NumCosmo's serialization API.

    Configuration is encapsulated in ConfigOptions to allow passing through
    multiprocessing context without serialization issues.

    :param options: Configuration options with all necessary parameters
    :return: Exit code (0 for success)
    """
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    experiment, model_builders = _setup_experiment(options)

    numcosmo_yaml = options.output_path / f"numcosmo_{options.prefix}.yaml"
    builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
    ser.dict_str_to_yaml_file(experiment, numcosmo_yaml.as_posix())
    ser.dict_str_to_yaml_file(model_builders, builders_file.as_posix())

    return 0


@dataclasses.dataclass
class NumCosmoConfigGenerator(ConfigGenerator):
    """NumCosmo configuration generator.

    Generates NumCosmo YAML files for parameter estimation:
    - numcosmo_{prefix}.yaml: Experiment configuration with cosmology,
      dataset, likelihood, and priors
    - numcosmo_{prefix}.builders.yaml: Model builder definitions for
      systematic/nuisance parameters

    Configuration is written in a fresh subprocess to avoid GType conflicts.
    """

    framework = Frameworks.NUMCOSMO

    def write_config(self) -> None:
        """Write NumCosmo configuration in a fresh subprocess (safe for GType)."""
        assert self.factory_source is not None
        assert self.build_parameters is not None
        config_options = ConfigOptions(
            self.output_path,
            self.factory_source,
            self.build_parameters,
            self.models,
            self.cosmo_spec,
            self.use_absolute_path,
            self.required_cosmology,
            self.prefix,
        )
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_write_config_worker, args=(config_options,))
        proc.start()
        proc.join(timeout=300.0)

        if proc.exitcode != 0:
            raise RuntimeError(
                f"write_config() subprocess failed with exit code {proc.exitcode}"
            )
