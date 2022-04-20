from __future__ import annotations
from typing import List, Dict, Optional


def parameter_get_full_name(prefix: Optional[str], param: str) -> str:
    if prefix:
        return f"{prefix}_{param}"
    else:
        return param


class ParamsMap(Dict[str, float]):
    def get_from_prefix_param(self, prefix: Optional[str], param: str) -> float:
        fullname = parameter_get_full_name(prefix, param)

        if fullname in self.keys():
            return self[fullname]
        else:
            raise KeyError(f"Prefix `{prefix}`, key `{param}' not found.")


class RequiredParameters:
    def __init__(self, params_names: List[str]):
        self.params_names = params_names
        pass

    def __add__(self, rp: RequiredParameters):
        return RequiredParameters(self.params_names + rp.params_names)

    def get_params_names(self):
        return self.params_names
