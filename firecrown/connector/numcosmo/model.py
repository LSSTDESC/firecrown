"""Helper function to create NumCosmo models."""

from dataclasses import dataclass, field
from typing import List
import sys

import yaml
from numcosmo_py import Ncm, GObject


@dataclass
class ScalarParameter(yaml.YAMLObject):  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold scalar parameter information."""

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper
    yaml_tag = "!ScalarParameter"

    symbol: str
    name: str
    lower_bound: float = -sys.float_info.max
    upper_bound: float = +sys.float_info.max
    scale: float = 1.0
    absolute_tolerance: float = 0.0
    default_value: float = 0.0
    fit_type: Ncm.ParamType = Ncm.ParamType.FREE


@dataclass
class VectorParameter(yaml.YAMLObject):  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold scalar parameter information."""

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper
    yaml_tag = "!VectorParameter"

    default_length: int
    symbol: str
    name: str
    lower_bound: float = -sys.float_info.max
    upper_bound: float = +sys.float_info.max
    scale: float = 1.0
    absolute_tolerance: float = 0.0
    default_value: float = 0.0
    fit_type: Ncm.ParamType = Ncm.ParamType.FREE


@dataclass
class NumCosmoModel(yaml.YAMLObject):
    """Dataclass to hold NumCosmo model information."""

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper
    yaml_tag = "!NumCosmoModel"

    name: str
    description: str
    scalar_params: List[ScalarParameter] = field(default_factory=list)
    vector_params: List[VectorParameter] = field(default_factory=list)


def param_type_representer(dumper, ftype):
    """Representer for NumCosmo parameter types."""
    return dumper.represent_scalar(
        "!NcmParamType", Ncm.ParamType(ftype).value_nick
    )


def param_type_constructor(loader, node):
    """Constructor for NumCosmo parameter types."""
    value = loader.construct_scalar(node)
    enum_item = Ncm.cfg_get_enum_by_id_name_nick(Ncm.ParamType, value)
    return Ncm.ParamType(enum_item.value)


yaml.add_constructor("!NcmParamType", param_type_constructor)
yaml.add_representer(Ncm.ParamType, param_type_representer)


def define_numcosmo_model(numcosmo_model: NumCosmoModel) -> Ncm.Model:
    """Define a NumCosmo model.

    :param numcosmo_model: NumCosmo model to define.

    :return: NumCosmo model class.
    """
    model_name = numcosmo_model.name
    model_description = numcosmo_model.description
    scalar_params = numcosmo_model.scalar_params
    vector_params = numcosmo_model.vector_params

    mb = Ncm.ModelBuilder.new(Ncm.Model, model_name, model_description)

    for sparam in scalar_params:
        mb.add_sparam(
            sparam.symbol,
            sparam.name,
            sparam.lower_bound,
            sparam.upper_bound,
            sparam.scale,
            sparam.absolute_tolerance,
            sparam.default_value,
            sparam.fit_type,
        )

    for vparam in vector_params:
        mb.add_vparam(
            vparam.default_length,
            vparam.symbol,
            vparam.name,
            vparam.lower_bound,
            vparam.upper_bound,
            vparam.scale,
            vparam.absolute_tolerance,
            vparam.default_value,
            vparam.fit_type,
        )

    numcosmo_model = mb.create()
    GObject.new(numcosmo_model)
    py_numcosmo_model = numcosmo_model.pytype  # type: ignore
    GObject.type_register(py_numcosmo_model)

    return py_numcosmo_model
