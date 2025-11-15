"""NumCosmo configuration file generation utilities.

Provides functions to create NumCosmo YAML configuration files
for cosmological parameter estimation with Firecrown likelihoods.

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

# Map CCL parameter names to (NumCosmo name, model, scale, prior_name)
NAME_MAP: dict[str, str] = {
    "Omega_c": "Omegac",
    "Omega_b": "Omegab",
    "Omega_k": "Omegak",
    "h": "H0",
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
    """Create NumCosmo mapping configuration."""
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
    """Create NumCosmo factory instance."""
    previous_dir = os.getcwd()
    os.chdir(output_path)

    numcosmo_factory = NumCosmoFactory(
        factory_source_str,
        build_parameters,
        mapping=mapping,
        model_list=model_list,
    )

    os.chdir(previous_dir)
    return numcosmo_factory


def _add_prior(
    priors: list[Ncm.Prior],
    param_name: str,
    prior: PriorGaussian | PriorUniform | None,
    scale: float = 1.0,
) -> None:
    """Add prior to list if specified.

    :param priors: List of priors to append to
    :param param_name: Parameter name in NumCosmo
    :param prior: Prior specification
    :param scale: Scale factor to apply to prior bounds
    """
    if prior is None:
        return

    match prior:
        case PriorGaussian():
            prior_g = Ncm.PriorGaussParam.new_name(
                param_name, prior.mean * scale, prior.sigma * scale
            )
            priors.append(prior_g)
        case PriorUniform():
            prior_u = Ncm.PriorFlatParam.new_name(
                param_name, prior.lower * scale, prior.upper * scale, 1.0e-3
            )
            priors.append(prior_u)
        case _ as unreachable:
            assert_never(unreachable)


def _setup_model_set(
    required_cosmology: FrameworkCosmology,
    mapping: nc_cosmosis.MappingNumCosmo | None,
    cosmo_spec: CCLCosmologyAnalysisSpec,
) -> tuple[Ncm.MSet, list[Ncm.Prior]]:
    """Create and configure model set."""
    mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter
    if required_cosmology == FrameworkCosmology.NONE:
        return mset, []

    assert mapping is not None
    ccl_cosmo = cosmo_spec.cosmology.to_ccl_cosmology()
    massive_nu = bool(ccl_cosmo["sum_nu_masses"])

    cosmo = Nc.HICosmoDECpl(massnu_length=1 if massive_nu else 0)
    cosmo.omega_x2omega_k()
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter

    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)
    mset.set(cosmo)

    priors: list[Ncm.Prior] = []
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

    # Set cosmological parameters and their priors
    for ccl_name, (model, scale, prior) in param_map.items():
        nc_name = NAME_MAP[ccl_name]
        if ccl_cosmo[ccl_name] is not None:
            model[nc_name] = ccl_cosmo[ccl_name] * scale
            _add_prior(
                priors, f"{mset.get_ns_by_id(model.id())}:{nc_name}", prior, scale
            )

    # Handle massive neutrino parameters
    if ccl_cosmo["sum_nu_masses"] > 0.0:
        nc_name = NAME_MAP["sum_nu_masses"]
        cosmo[nc_name] = ccl_cosmo["sum_nu_masses"]
        _add_prior(
            priors, f"{mset.get_ns_by_id(cosmo.id())}:{nc_name}", cosmo_spec.priors.m_nu
        )

    # Handle amplitude parameters (A_s)
    if ccl_cosmo["A_s"] is not None and not np.isnan(ccl_cosmo["A_s"]):

        def convert(A_s):
            return np.log(1.0e10 * A_s)

        def convert_sigma(sigma_A_s):
            return sigma_A_s / ccl_cosmo["A_s"]

        prim["ln10e10ASA"] = convert(ccl_cosmo["A_s"])
        if cosmo_spec.priors.A_s is not None:
            match cosmo_spec.priors.A_s:
                case PriorGaussian():
                    priors.append(
                        Ncm.PriorGaussParam.new_name(
                            f"{mset.get_ns_by_id(prim.id())}:ln10e10ASA",
                            convert(cosmo_spec.priors.A_s.mean),
                            convert_sigma(cosmo_spec.priors.A_s.sigma),
                        )
                    )
                case PriorUniform():
                    priors.append(
                        Ncm.PriorFlatParam.new_name(
                            f"{mset.get_ns_by_id(prim.id())}:ln10e10ASA",
                            convert(cosmo_spec.priors.A_s.lower),
                            convert(cosmo_spec.priors.A_s.upper),
                            1.0e-3,
                        )
                    )
                case _ as unreachable:
                    assert_never(unreachable)

    if ccl_cosmo["sigma8"] is not None and not np.isnan(ccl_cosmo["sigma8"]):
        A_s = np.exp(prim["ln10e10ASA"]) * 1.0e-10
        if mapping.p_ml is None:
            raise ValueError("Mapping must have p_ml set for sigma8")
        fact = (
            ccl_cosmo["sigma8"]
            / mapping.p_ml.sigma_tophat_R(cosmo, 1.0e-5, 0.0, 8.0 / ccl_cosmo["h"])
        ) ** 2
        prim.param_set_by_name("ln10e10ASA", np.log(1.0e10 * A_s * fact))
        psf = Ncm.PowspecFilter.new(mapping.p_ml, Ncm.PowspecFilterType.TOPHAT)

        if cosmo_spec.priors.sigma8 is not None:
            sigma8_func = Ncm.MSetFuncList.new("NcHICosmo:sigma8", psf)
            match cosmo_spec.priors.sigma8:
                case PriorGaussian():
                    prior_g = Ncm.PriorGaussFunc.new(
                        sigma8_func,
                        cosmo_spec.priors.sigma8.mean,
                        cosmo_spec.priors.sigma8.sigma,
                        0.0,
                    )
                    priors.append(prior_g)
                case PriorUniform():
                    prior_u = Ncm.PriorFlatFunc.new(
                        sigma8_func,
                        cosmo_spec.priors.sigma8.lower,
                        cosmo_spec.priors.sigma8.upper,
                        1.0e-3,
                        0.0,
                    )
                    priors.append(prior_u)
                case _ as unreachable:
                    assert_never(unreachable)

    return mset, priors


def _setup_dataset(numcosmo_factory: NumCosmoFactory) -> Ncm.Dataset:
    """Create and configure dataset."""
    dataset = Ncm.Dataset.new()  # pylint: disable=no-value-for-parameter
    firecrown_data = numcosmo_factory.get_data()
    if isinstance(firecrown_data, Ncm.DataGaussCov):
        firecrown_data.set_size(0)
    dataset.append_data(firecrown_data)
    return dataset


def create_config(
    output_path: Path,
    factory_source: str | Path,
    build_parameters: NamedParameters,
    model_list: list[str],
    cosmo_spec: CCLCosmologyAnalysisSpec,
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
    factory_source_str = get_path_str(factory_source, use_absolute_path)

    mapping = _create_mapping(distance_max_z, reltol, required_cosmology)
    numcosmo_factory = _create_factory(
        output_path, factory_source_str, build_parameters, mapping, model_list
    )

    mset, priors = _setup_model_set(required_cosmology, mapping, cosmo_spec)
    dataset = _setup_dataset(numcosmo_factory)

    likelihood = Ncm.Likelihood.new(dataset)
    for prior in priors:
        likelihood.priors_add(prior)

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
    """Worker executed in a fresh process."""
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
        output_path,
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
    """Generates NumCosmo YAML configuration files.

    Creates two files:
    - numcosmo_{prefix}.yaml: Experiment configuration
    - numcosmo_{prefix}.builders.yaml: Model builders
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
