# Firecrown Analysis Configuration Generators

This module provides automated generation of framework-specific configuration files for cosmological parameter estimation with Firecrown likelihoods.

## Overview

The analysis module generates complete, ready-to-run configurations for three statistical analysis frameworks:
- **CosmoSIS**: INI-based pipeline configuration
- **Cobaya**: YAML-based Bayesian analysis
- **NumCosmo**: YAML-based numerical cosmology framework

## Architecture

### Core Components

**`_types.py`**: Type definitions and base classes
- `ConfigGenerator`: Abstract base class for all framework generators
- `CCLCosmologyAnalysisSpec`: Cosmology specification with parameters and priors
- `Model`, `Parameter`: Definitions for sampling parameters
- `FrameworkCosmology`: Enum for cosmology computation requirements (none/background/linear/nonlinear)

**`_config_generator.py`**: Factory function
- `get_generator()`: Creates appropriate generator based on framework choice

**Framework-specific modules**:
- `_cosmosis.py`: CosmoSIS INI file generation
- `_cobaya.py`: Cobaya YAML file generation  
- `_numcosmo.py`: NumCosmo YAML file generation

### Design Pattern

All generators follow a builder pattern with phased construction:

```python
generator = get_generator(framework, output_path, prefix, use_absolute_path, require_cosmology)
generator.add_sacc(sacc_path)
generator.add_factory(factory_path)
generator.add_build_parameters(build_parameters)
generator.add_models(models)
generator.write_config()
```

## Framework-Specific Details

### CosmoSIS (`_cosmosis.py`)

**Generated Files**:
- `cosmosis_{prefix}.ini`: Pipeline configuration
- `cosmosis_{prefix}_values.ini`: Parameter values and bounds
- `cosmosis_{prefix}_priors.ini`: Prior specifications (if priors defined)

**Key Features**:
- Configures CAMB for cosmology computation (optional)
- Supports both σ₈ and A_s amplitude parameters
- Handles Gaussian priors by converting to uniform bounds (mean ± 3σ)
- Uses CosmoSIS Standard Library (CSL) modules

**Parameter Mapping** (CCL → CosmoSIS):
```python
NAME_MAP = {
    "Omega_c": "omega_c",
    "Omega_b": "omega_b",
    "h": "h0",
    "sigma8": "sigma_8",
    # ...
}
```

**Special Handling**:
- Adds fixed `tau = 0.08` (reionization optical depth) required by CAMB
- Formats floats with `.0` suffix for CosmoSIS compatibility
- Supports extended interpolation in INI files

### Cobaya (`_cobaya.py`)

**Generated Files**:
- `cobaya_{prefix}.yaml`: Complete analysis configuration

**Key Features**:
- Configures CAMB theory code (optional)
- Native support for both Gaussian and uniform priors
- Automatic parameter scaling (e.g., h → H0 × 100)
- Uses Cobaya's external likelihood interface

**Parameter Mapping** (CCL → Cobaya):
```python
NAME_MAP = {
    "Omega_c": "omch2",  # Scaled by h²
    "Omega_b": "ombh2",  # Scaled by h²
    "h": "H0",           # Scaled by 100
    # ...
}
```

**Prior Format**:
- Gaussian: `{"dist": "norm", "loc": mean, "scale": sigma}`
- Uniform: `{"min": lower, "max": upper}`

### NumCosmo (`_numcosmo.py`)

**Generated Files**:
- `numcosmo_{prefix}.yaml`: Experiment configuration
- `numcosmo_{prefix}.builders.yaml`: Model builders

**Key Features**:
- Configures CLASS for cosmology computation (optional)
- Supports functional priors (e.g., σ₈ as derived parameter)
- Handles A_s with logarithmic transformation
- Runs in subprocess to avoid GType conflicts

**Parameter Mapping** (CCL → NumCosmo):
```python
NAME_MAP = {
    "Omega_c": "Omegac",
    "h": "H0",           # Scaled by 100
    "n_s": "n_SA",
    "sum_nu_masses": "mnu_0",
    # ...
}
```

**Special Handling**:
- A_s stored as `ln10e10ASA = log(10¹⁰ × A_s)`
- σ₈ implemented as functional prior using power spectrum filter
- Priors use namespace prefixes: `{namespace}:{parameter}`
- Configuration written in fresh subprocess for GType safety

## Cosmology Requirements

The `FrameworkCosmology` enum controls cosmology computation:

- **`NONE`**: No cosmology computation (likelihood provides its own)
- **`BACKGROUND`**: Background cosmology only (distances, H(z))
- **`LINEAR`**: Linear matter power spectrum
- **`NONLINEAR`**: Nonlinear matter power spectrum (default)

**Framework Implementations**:

| Requirement | CosmoSIS | Cobaya | NumCosmo |
|-------------|----------|--------|----------|
| NONE | No CAMB | No theory | No mapping |
| BACKGROUND | CAMB background mode | CAMB | CLASS (no P(k)) |
| LINEAR | CAMB power mode | CAMB | CLASS linear |
| NONLINEAR | CAMB + halofit | CAMB + halofit | CLASS + HALOFIT |

## Parameter Scaling

Different frameworks use different conventions for cosmological parameters:

| CCL Parameter | CosmoSIS | Cobaya | NumCosmo | Scale Factor |
|---------------|----------|--------|----------|--------------|
| h | h0 | H0 | H0 | 1.0 / 100.0 / 100.0 |
| Omega_c | omega_c | omch2 | Omegac | 1.0 / h² / 1.0 |
| Omega_b | omega_b | ombh2 | Omegab | 1.0 / h² / 1.0 |
| A_s | A_s | As | ln10e10ASA | 1.0 / 1.0 / log(10¹⁰×) |

## Prior Handling

### CosmoSIS
- **Values file**: Uniform priors as `min start max`
- **Priors file**: Gaussian priors as `gaussian mean sigma`
- Gaussian priors in values file converted to uniform (mean ± 3σ)

### Cobaya
- **Gaussian**: `{"dist": "norm", "loc": μ, "scale": σ}`
- **Uniform**: `{"min": a, "max": b}`
- Priors applied with automatic scaling

### NumCosmo
- **Gaussian**: `Ncm.PriorGaussParam` with scaled mean and sigma
- **Uniform**: `Ncm.PriorFlatParam` with scaled bounds
- Functional priors for derived parameters (e.g., σ₈)

## Usage Example

```python
from firecrown.app.analysis import get_generator, Frameworks, FrameworkCosmology
from firecrown.app.analysis import CCLCosmologyAnalysisSpec

# Create cosmology specification
cosmo_spec = CCLCosmologyAnalysisSpec.vanilla_lcdm()

# Create generator
generator = get_generator(
    framework=Frameworks.COSMOSIS,
    output_path=Path("./output"),
    prefix="my_analysis",
    use_absolute_path=True,
    require_cosmology=FrameworkCosmology.NONLINEAR
)

# Configure generator
generator.cosmo_spec = cosmo_spec
generator.add_sacc(Path("data.sacc"))
generator.add_factory(Path("factory.py"))
generator.add_build_parameters(NamedParameters({"sacc_file": "data.sacc"}))
generator.add_models(models)

# Generate configuration files
generator.write_config()
```

## Implementation Notes

### CosmoSIS
- Uses `configparser` with extended interpolation
- Comments formatted with `;;` prefix
- Requires CSL_DIR environment variable
- Fixed parameters need `.0` suffix

### Cobaya
- Pure Python dictionaries converted to YAML
- Uses `LikelihoodConnector` external likelihood
- Supports `input_style="CAMB"` for theory integration

### NumCosmo
- Requires subprocess execution for GType safety
- Uses `Ncm.Serialize` for YAML output
- Model builders registered dynamically
- Supports complex prior types (functional, derived)

## Error Handling

All generators validate:
- Required fields are present before `write_config()`
- Cosmology specification is valid
- Parameter bounds are consistent
- Prior specifications match parameter types
- File paths are accessible

## Extension Points

To add a new framework:

1. Create `_newframework.py` module
2. Implement `NewFrameworkConfigGenerator(ConfigGenerator)`
3. Add framework to `Frameworks` enum
4. Update `get_generator()` factory function
5. Add parameter name mapping if needed
6. Implement framework-specific prior handling

## References

- CosmoSIS: https://cosmosis.readthedocs.io/
- Cobaya: https://cobaya.readthedocs.io/
- NumCosmo: https://numcosmo.github.io/
