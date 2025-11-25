"""YAML serialization utilities for Firecrown."""

import yaml
from pydantic import BaseModel
from typing_extensions import Self

from yaml import Dumper, Loader


class YAMLSerializable:
    """Protocol for classes that can be serialized to and from YAML."""

    def to_yaml(self) -> str:
        """Return the YAML representation of the object."""
        return yaml.dump(self, Dumper=Dumper, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        """Load the object from YAML."""
        return yaml.load(yaml_str, Loader=Loader)


def base_model_from_yaml(cls: type, yaml_str: str):
    """Create a base model from a yaml string."""
    if not issubclass(cls, BaseModel):
        raise ValueError("cls must be a subclass of pydantic.BaseModel")

    try:
        return cls.model_validate(
            yaml.safe_load(yaml_str),
            strict=True,
        )
    except Exception as e:
        raise ValueError(
            f"Error creating {cls.__name__} from yaml. Parsing error message:\n{e}"
        ) from e


def base_model_to_yaml(model: BaseModel) -> str:
    """Convert a base model to a yaml string."""
    return yaml.dump(
        model.model_dump(), default_flow_style=None, sort_keys=False, width=80
    )
