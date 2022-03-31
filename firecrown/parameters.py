from __future__ import annotations
from typing import List, Dict, Optional

class ParamsMap(Dict[str, float]):
    pass
    
    def get_from_prefix_param(self, 
                              prefix: Optional[str],
                              param: str) -> float:
        if prefix and f"{prefix}_{param}" in self.keys():
            return self[f"{prefix}_{param}"]
        elif param in self.keys():
            return self[param]
        else:
            raise KeyError(f"Prefix `{prefix}`, key `{param}' not found.")
