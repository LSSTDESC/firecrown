from __future__ import annotations
from typing import List, Dict, Optional

from .core import Systematic

def get_from_prefix_param(systematic: Systematic, params: Dict[str, float],
                          prefix: Optional[str],
                          param: str) -> float:
    if prefix and f"{prefix}_{param}" in params.keys():
        return params[f"{prefix}_{param}"]
    elif param in params.keys():
        return params[param]
    else:
        typename = type(systematic).__name__
        raise KeyError(f"{typename} key `{param}' not found")
