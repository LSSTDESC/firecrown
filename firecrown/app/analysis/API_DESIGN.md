# API Design: Public vs Internal

## Overview

The `firecrown.app.analysis` module follows a clear public/internal API pattern to provide stability while allowing internal improvements.

## Naming Convention

### Internal Modules (Prefixed with `_`)

All implementation modules are prefixed with underscore:

```
_analysis_builder.py
_types.py
_config_generator.py
_cosmosis.py
_cobaya.py
_numcosmo.py
```

**Characteristics:**
- Implementation details
- May change without notice
- No backward compatibility guarantees
- Not part of public API

### Public API (Exported in `__init__.py`)

Only items in `__all__` are public:

```python
__all__ = [
    "AnalysisBuilder",   # Base class
    "Frameworks",        # Framework enum
    "Model",             # Model definition
    "Parameter",         # Parameter definition
    "get_generator",     # Advanced: generator factory
    "ConfigGenerator",   # Advanced: generator base class
]
```

**Characteristics:**
- Stable interface
- Backward compatibility guaranteed
- Documented and supported
- Safe to use in external code

## Documentation Pattern

Each internal module includes a notice:

```python
"""Module description.

This is an internal module. Use the public API from firecrown.app.analysis.
"""
```

The `__init__.py` includes comprehensive documentation:

```python
"""Analysis building infrastructure for Firecrown.

Public API (backward compatibility guaranteed):
    - AnalysisBuilder: Base class for building analyses
    - Frameworks: Enum of supported frameworks
    - Model: Model parameter definition
    - Parameter: Individual parameter definition

Internal modules (prefixed with _) are implementation details and may change
without notice. Only use the public API exported in __all__.
"""
```

## Usage Guidelines

### ✅ Correct Usage

```python
# Import from public API
from firecrown.app.analysis import AnalysisBuilder, Model, Parameter

class MyAnalysis(AnalysisBuilder):
    ...
```

### ❌ Incorrect Usage

```python
# Don't import from internal modules
from firecrown.app.analysis._analysis_builder import AnalysisBuilder  # BAD
from firecrown.app.analysis._types import Model  # BAD
```

## Benefits

1. **Stability**: Public API remains stable across versions
2. **Flexibility**: Internal implementation can be refactored freely
3. **Clarity**: Clear boundary between public and private
4. **Maintainability**: Easier to evolve the codebase
5. **Documentation**: Users know what's safe to use

## Version Compatibility

- **Public API**: Follows semantic versioning
  - Breaking changes only in major versions
  - Deprecation warnings before removal
  
- **Internal modules**: No guarantees
  - Can change in any version
  - No deprecation period required

## Examples in the Wild

This pattern is used by many Python projects:

- `_internal` in pip
- `_vendor` in requests
- `_compat` in many libraries
- Leading `_` for private modules is a Python convention
