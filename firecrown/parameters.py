from __future__ import annotations
from typing import List, Dict, Optional

def get_from_prefix_param(obj, params: Dict[str, float],
                          prefix: Optional[str],
                          param: str) -> float:
    if prefix and f"{prefix}_{param}" in params.keys():
        return params[f"{prefix}_{param}"]
    elif param in params.keys():
        return params[param]
    else:
        typename = type(obj).__name__
        raise KeyError(f"{typename} key `{param}' not found")
