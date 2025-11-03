"""Helper functions for NumCosmo ini file generation.

This module provides utilities for generating NumCosmo configuration files
with proper comment formatting and standard sections.
"""

from pathlib import Path
import os

from numcosmo_py import Ncm, Nc
import numcosmo_py.external.cosmosis as nc_cosmosis
from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory
from firecrown.likelihood.likelihood import NamedParameters


def create_standard_numcosmo_config(
    factory_path: Path,
    sacc_path: Path,
    build_parameters: NamedParameters,
    model_list: list[str],
    use_absolute_path: bool = False,
    distance_max_z: float = 4.0,
    reltol: float = 1e-7,
) -> Ncm.ObjDictStr:
    """Create a standard NumCosmo configuration object.

    :param factory_path: Path to the factory file
    :param sacc_path: Path to the SACC data file
    :param output_path: Path to the output directory
    :return: Configured Ncm.ObjDictStr object
    """
    experiment = Ncm.ObjDictStr()
    if use_absolute_path:
        factory_filename = factory_path.absolute().as_posix()
        sacc_filename = sacc_path.absolute().as_posix()
    else:
        factory_filename = factory_path.name
        sacc_filename = sacc_path.name

    mapping = nc_cosmosis.create_numcosmo_mapping(
        matter_ps=nc_cosmosis.LinearMatterPowerSpectrum.CLASS,
        nonlin_matter_ps=nc_cosmosis.NonLinearMatterPowerSpectrum.HALOFIT,
        distance_max_z=distance_max_z,
        reltol=reltol,
    )

    build_parameters.set_from_basic_dict({"sacc_file": sacc_filename})

    previous_dir = os.getcwd()
    os.chdir(factory_path.parent)

    numcosmo_factory = NumCosmoFactory(
        factory_filename,
        build_parameters,
        mapping=mapping,
        model_list=model_list,
    )

    os.chdir(previous_dir)

    mset = Ncm.MSet.empty_new()  # pylint: disable=no-value-for-parameter

    cosmo = Nc.HICosmoDECpl()
    cosmo.omega_x2omega_k()
    prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
    reion = Nc.HIReionCamb.new()  # pylint: disable=no-value-for-parameter
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)
    mset.set(cosmo)

    dataset = Ncm.Dataset.new()  # pylint: disable=no-value-for-parameter
    likelihood = Ncm.Likelihood.new(dataset)
    firecrown_data = numcosmo_factory.get_data()
    if isinstance(firecrown_data, Ncm.DataGaussCov):
        firecrown_data.set_size(0)
    dataset.append_data(firecrown_data)

    experiment.add("likelihood", likelihood)
    experiment.add("model-set", mset)

    return experiment
