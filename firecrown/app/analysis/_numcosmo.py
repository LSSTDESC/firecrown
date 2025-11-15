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
from typing import assert_never
from pathlib import Path
import os
import dataclasses
import numpy as np
from numcosmo_py import Ncm, Nc
import numcosmo_py.external.cosmosis as nc_cosmosis
from numcosmo_py.helper import register_model_class
from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory
from firecrown.likelihood.likelihood import NamedParameters
from ._types import (
    Model,
    Frameworks,
    ConfigGenerator,
    FrameworkCosmology,
    CCLCosmologyAnalysisSpec,
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
    "sum_nu_masses": "mnu_0",
}


def _create_mapping(
    distance_max_z: float, reltol: float, required_cosmology: FrameworkCosmology
) -> nc_cosmosis.MappingNumCosmo | None:
    """Create NumCosmo cosmology mapping.

    Configures CLASS computation level based on required cosmology.
    Returns None if no cosmology computation needed.

    :param distance_max_z: Maximum redshift for distance calculations
    :param reltol: Relative tolerance for numerical computations
    :param required_cosmology: Level of cosmology computation
    :return: NumCosmo mapping or None
    """
    match required_cosmology:
        case FrameworkCosmology.NONE:
            return None
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
        case _ as unreachable:
            assert_never(unreachable)


def _create_factory(
    output_path: Path,
    factory_source_str: str,
    build_parameters: NamedParameters,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    model_list: list[str],
) -> NumCosmoFactory:
    """Create NumCosmo factory instance.

    Temporarily changes to output_path directory for factory initialization,
    then restores original working directory.

    :param output_path: Output directory
    :param factory_source_str: Factory source path or module string
    :param build_parameters: Parameters for likelihood factory
    :param mapping: NumCosmo cosmology mapping
    :param model_list: List of model names
    :return: Initialized NumCosmo factory
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
    scale: float = 1.0,
) -> None:
    """Add prior to list if specified.

    Creates appropriate NumCosmo prior object (Gaussian or flat/uniform)
    and appends to priors list. Does nothing if prior is None.

    :param priors: List of priors (modified in-place)
    :param param_name: Full parameter name with namespace (e.g., 'NcHICosmo:H0')
    :param prior: Prior specification
    :param scale: Scale factor applied to prior bounds
    """
    if prior is None:
        return

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


def _set_standard_params(
    mset: Ncm.MSet,
    cosmo: Nc.HICosmoDECpl,
    prim: Nc.HIPrimPowerLaw,
    ccl_cosmo: dict,
    cosmo_spec: CCLCosmologyAnalysisSpec,
    priors: list[Ncm.Prior],
) -> None:
    """Set standard cosmological parameters in model set.

    Configures parameters like Omega_c, Omega_b, h, n_s, etc. with their
    values and priors. Applies scaling where needed (e.g., h → H0 × 100).

    :param mset: NumCosmo model set
    :param cosmo: Cosmology model
    :param prim: Primordial power spectrum model
    :param ccl_cosmo: CCL cosmology dictionary
    :param cosmo_spec: Cosmology specification with priors
    :param priors: List of priors (modified in-place)
    """
    param_map: dict[str, tuple[Ncm.Model, float, Prior | None]] = {
        "Omega_c": (cosmo, 1.0, cosmo_spec.priors.Omega_c),
        "Omega_b": (cosmo, 1.0, cosmo_spec.priors.Omega_b),
        "Omega_k": (cosmo, 1.0, cosmo_spec.priors.Omega_k),
        "h": (cosmo, 100.0, cosmo_spec.priors.h),
        "w0": (cosmo, 1.0, cosmo_spec.priors.w0),
        "wa": (cosmo, 1.0, cosmo_spec.priors.wa),
        "n_s": (prim, 1.0, cosmo_spec.priors.n_s),
        "T_CMB": (cosmo, 1.0, None),
        "Neff": (cosmo, 1.0, cosmo_spec.priors.Neff),
    }

    for ccl_name, (model, scale, prior) in param_map.items():
        if ccl_cosmo[ccl_name] is not None:
            model[NAME_MAP[ccl_name]] = ccl_cosmo[ccl_name] * scale
            _add_prior(
                priors,
                f"{mset.get_ns_by_id(model.id())}:{NAME_MAP[ccl_name]}",
                prior,
                scale,
            )

    if ccl_cosmo["sum_nu_masses"] > 0.0:
        cosmo[NAME_MAP["sum_nu_masses"]] = ccl_cosmo["sum_nu_masses"]
        _add_prior(
            priors,
            f"{mset.get_ns_by_id(cosmo.id())}:{NAME_MAP['sum_nu_masses']}",
            cosmo_spec.priors.m_nu,
        )


def _set_amplitude_A_s(
    mset: Ncm.MSet,
    prim: Nc.HIPrimPowerLaw,
    A_s: float,
    prior: Prior | None,
    priors: list[Ncm.Prior],
) -> None:
    """Set A_s amplitude parameter with logarithmic transformation.

    NumCosmo stores A_s as ln10e10ASA = ln(10¹⁰ × A_s) for numerical stability.
    Prior bounds are transformed accordingly.

    :param mset: NumCosmo model set
    :param prim: Primordial power spectrum model
    :param A_s: A_s value
    :param prior: Prior specification
    :param priors: List of priors (modified in-place)
    """
    prim["ln10e10ASA"] = np.log(1.0e10 * A_s)
    if prior is None:
        return

    param_name = f"{mset.get_ns_by_id(prim.id())}:ln10e10ASA"
    match prior:
        case PriorGaussian():
            priors.append(
                Ncm.PriorGaussParam.new_name(
                    param_name,
                    np.log(1.0e10 * prior.mean),
                    prior.sigma / A_s,
                )
            )
        case PriorUniform():
            priors.append(
                Ncm.PriorFlatParam.new_name(
                    param_name,
                    np.log(1.0e10 * prior.lower),
                    np.log(1.0e10 * prior.upper),
                    1.0e-3,
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _set_amplitude_sigma8(
    cosmo: Nc.HICosmoDECpl,
    prim: Nc.HIPrimPowerLaw,
    mapping: nc_cosmosis.MappingNumCosmo,
    sigma8: float,
    h: float,
    prior: Prior | None,
    priors: list[Ncm.Prior],
    reltol: float,
) -> None:
    """Set sigma8 amplitude parameter as functional prior.

    Computes A_s from sigma8 target value and sets up functional prior
    that constrains sigma8 directly. This allows sampling in A_s space
    while applying prior to derived sigma8.

    :param cosmo: Cosmology model
    :param prim: Primordial power spectrum model
    :param mapping: NumCosmo mapping with power spectrum
    :param sigma8: Target sigma8 value
    :param h: Hubble parameter value
    :param prior: Prior specification for sigma8
    :param priors: List of priors (modified in-place)
    :param reltol: Relative tolerance for sigma8 calculation
    """
    if mapping.p_ml is None:
        raise ValueError("Mapping must have p_ml set for sigma8")

    A_s = np.exp(prim["ln10e10ASA"]) * 1.0e-10
    fact = (sigma8 / mapping.p_ml.sigma_tophat_R(cosmo, reltol, 0.0, 8.0 / h)) ** 2
    prim.param_set_by_name("ln10e10ASA", np.log(1.0e10 * A_s * fact))

    if prior is None:
        return

    psf = Ncm.PowspecFilter.new(mapping.p_ml, Ncm.PowspecFilterType.TOPHAT)
    sigma8_func = Ncm.MSetFuncList.new("NcHICosmo:sigma8", psf)

    match prior:
        case PriorGaussian():
            priors.append(
                Ncm.PriorGaussFunc.new(sigma8_func, prior.mean, prior.sigma, 0.0)
            )
        case PriorUniform():
            priors.append(
                Ncm.PriorFlatFunc.new(
                    sigma8_func, prior.lower, prior.upper, 1.0e-3, 0.0
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _setup_model_set(
    required_cosmology: FrameworkCosmology,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    cosmo_spec: CCLCosmologyAnalysisSpec,
    reltol: float,
) -> tuple[Ncm.MSet, list[Ncm.Prior]]:
    """Create and configure NumCosmo model set with cosmology.

    Sets up cosmology model (HICosmoDECpl), primordial power spectrum (HIPrimPowerLaw),
    and reionization (HIReionCamb). Configures all parameters and priors.

    :param required_cosmology: Level of cosmology computation
    :param mapping: NumCosmo cosmology mapping
    :param cosmo_spec: Cosmology specification
    :param reltol: Relative tolerance for calculations
    :return: Tuple of (model set, list of priors)
    """
    mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
    if required_cosmology == FrameworkCosmology.NONE:
        return mset, []

    assert mapping is not None
    ccl_cosmo = cosmo_spec.cosmology.to_ccl_cosmology()

    cosmo = Nc.HICosmoDECpl(massnu_length=1 if ccl_cosmo["sum_nu_masses"] else 0)
    cosmo.omega_x2omega_k()
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)
    mset.set(cosmo)

    priors: list[Ncm.Prior] = []
    _set_standard_params(mset, cosmo, prim, ccl_cosmo, cosmo_spec, priors)

    if ccl_cosmo["A_s"] is not None and not np.isnan(ccl_cosmo["A_s"]):
        _set_amplitude_A_s(mset, prim, ccl_cosmo["A_s"], cosmo_spec.priors.A_s, priors)

    if ccl_cosmo["sigma8"] is not None and not np.isnan(ccl_cosmo["sigma8"]):
        _set_amplitude_sigma8(
            cosmo,
            prim,
            mapping,
            ccl_cosmo["sigma8"],
            ccl_cosmo["h"],
            cosmo_spec.priors.sigma8,
            priors,
            reltol,
        )

    return mset, priors


def _create_dataset(numcosmo_factory: NumCosmoFactory) -> Ncm.Dataset:
    """Create NumCosmo dataset from factory.

    Wraps Firecrown data object in NumCosmo dataset container.

    :param numcosmo_factory: NumCosmo factory instance
    :return: NumCosmo dataset
    """
    dataset = Ncm.Dataset.new()  # pylint: disable=no-value-for-parameter
    firecrown_data = numcosmo_factory.get_data()
    if isinstance(firecrown_data, Ncm.DataGaussCov):
        firecrown_data.set_size(0)
    dataset.append_data(firecrown_data)
    return dataset


def _create_likelihood(dataset: Ncm.Dataset, priors: list[Ncm.Prior]) -> Ncm.Likelihood:
    """Create NumCosmo likelihood with priors.

    Combines dataset with prior constraints into complete likelihood.

    :param dataset: NumCosmo dataset
    :param priors: List of prior objects
    :return: NumCosmo likelihood
    """
    likelihood = Ncm.Likelihood.new(dataset)
    for prior in priors:
        likelihood.priors_add(prior)
    return likelihood


def create_config(
    output_path: Path,
    factory_source: str | Path,
    build_parameters: NamedParameters,
    model_list: list[str],
    cosmo_spec: CCLCosmologyAnalysisSpec,
    *,
    use_absolute_path: bool = False,
    required_cosmology: FrameworkCosmology = FrameworkCosmology.NONLINEAR,
    distance_max_z: float = 4.0,
    reltol: float = 1e-7,
) -> Ncm.ObjDictStr:
    """Create NumCosmo experiment configuration.

    Builds complete experiment with cosmology mapping, model set, dataset,
    and likelihood. Returns serializable object dictionary.

    :param output_path: Output directory
    :param factory_source: Factory source path or module string
    :param build_parameters: Parameters for likelihood factory
    :param model_list: List of model names
    :param cosmo_spec: Cosmology specification
    :param use_absolute_path: Use absolute paths in configuration
    :param required_cosmology: Level of cosmology computation
    :param distance_max_z: Maximum redshift for distance calculations
    :param reltol: Relative tolerance for numerical computations
    :return: NumCosmo experiment object dictionary
    """
    mapping = _create_mapping(distance_max_z, reltol, required_cosmology)
    factory = _create_factory(
        output_path,
        get_path_str(factory_source, use_absolute_path),
        build_parameters,
        mapping,
        model_list,
    )
    mset, priors = _setup_model_set(required_cosmology, mapping, cosmo_spec, reltol)
    likelihood = _create_likelihood(_create_dataset(factory), priors)

    experiment = Ncm.ObjDictStr()
    experiment.add("likelihood", likelihood)
    experiment.add("model-set", mset)
    return experiment


def add_models(
    config: Ncm.ObjDictStr,
    models: list[Model],
) -> Ncm.ObjDictStr:
    """Add systematic/nuisance models to NumCosmo configuration.

    Creates model builders for each model, registers them dynamically,
    and adds instances to the model set.

    :param config: NumCosmo experiment object (modified in-place)
    :param models: List of models with parameters
    :return: Model builders object dictionary
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


def _write_config_worker(
    args: tuple[
        Path,
        str,
        NamedParameters,
        list[Model],
        CCLCosmologyAnalysisSpec,
        bool,
        FrameworkCosmology,
        str,
    ],
) -> int:
    """Worker function executed in fresh subprocess.

    Runs in isolated process to avoid GType conflicts. Creates configuration
    and writes YAML files using NumCosmo serialization.

    :param args: Tuple of configuration parameters
    :return: Exit code (0 for success)
    """
    (
        output_path,
        factory_source,
        build_parameters,
        models,
        cosmo_spec,
        use_absolute_path,
        required_cosmology,
        prefix,
    ) = args

    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    model_list = [m.name for m in models]
    config = create_config(
        output_path=output_path,
        factory_source=factory_source,
        build_parameters=build_parameters,
        cosmo_spec=cosmo_spec,
        model_list=model_list,
        use_absolute_path=use_absolute_path,
        required_cosmology=required_cosmology,
    )
    model_builders = add_models(config, models)

    numcosmo_yaml = output_path / f"numcosmo_{prefix}.yaml"
    builders_file = numcosmo_yaml.with_suffix(".builders.yaml")

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
    ser.dict_str_to_yaml_file(config, numcosmo_yaml.as_posix())
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
        ctx = mp.get_context("spawn")
        args = (
            self.output_path,
            self.factory_source,
            self.build_parameters,
            self.models,
            self.cosmo_spec,
            self.use_absolute_path,
            self.required_cosmology,
            self.prefix,
        )

        proc = ctx.Process(target=_write_config_worker, args=(args,))
        proc.start()
        proc.join(timeout=300.0)

        if proc.exitcode != 0:
            raise RuntimeError(
                f"write_config() subprocess failed with exit code {proc.exitcode}"
            )
