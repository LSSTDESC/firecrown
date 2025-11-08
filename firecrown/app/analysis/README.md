# Analysis Building Infrastructure

This module provides the infrastructure for building complete Firecrown analysis setups.

## Public API (Backward Compatibility Guaranteed)

Import from `firecrown.app.analysis`:

```python
from firecrown.app.analysis import (
    AnalysisBuilder,  # Base class for building analyses
    Frameworks,       # Enum of supported frameworks
    Model,            # Model parameter definition
    Parameter,        # Individual parameter definition
    get_generator,    # Factory for config generators (advanced)
    ConfigGenerator,  # Base generator class (advanced)
)
```

Only the items listed in `__all__` are part of the public API and have backward compatibility guarantees.

## Internal Modules (No Backward Compatibility)

All modules prefixed with `_` are internal implementation details:

- `_analysis_builder.py` - AnalysisBuilder implementation
- `_types.py` - Type definitions
- `_config_generator.py` - Configuration generator strategy
- `_cosmosis.py` - CosmoSIS utilities
- `_cobaya.py` - Cobaya utilities
- `_numcosmo.py` - NumCosmo utilities

**Do not import directly from these modules.** They may change without notice.

## Usage

### Creating a New Analysis

Subclass `AnalysisBuilder` and implement the required methods:

```python
from firecrown.app.analysis import AnalysisBuilder, Model, Parameter

class MyAnalysis(AnalysisBuilder):
    def generate_sacc(self, output_path):
        # Generate or download SACC data
        ...
    
    def generate_factory(self, output_path, sacc):
        # Generate factory file
        ...
    
    def get_build_parameters(self, sacc_path):
        # Return build parameters
        ...
    
    def get_models(self):
        # Return model definitions
        ...
```

### Supported Frameworks

```python
from firecrown.app.analysis import Frameworks

# Available frameworks
Frameworks.COSMOSIS
Frameworks.COBAYA
Frameworks.NUMCOSMO
```

## Design Principles

1. **Public API Stability**: Items in `__all__` maintain backward compatibility
2. **Internal Flexibility**: `_` prefixed modules can change as needed
3. **Clear Boundaries**: Public API is minimal and well-documented
4. **Implementation Freedom**: Internal refactoring doesn't affect users
