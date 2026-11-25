# Firecrown Language

Terms used in Firecrown's tutorials, likelihood modeling, SACC data handling,
release compatibility, and scientific test expectations.

## 1. Scientific Concepts

### 1.1 Likelihood and Model Components

**Statistic**:
A scientific quantity represented in a likelihood by observed values and model predictions.

**Data vector**:
The ordered observed or supplied values for the selected statistic entries.

**Theory vector**:
The ordered model predictions corresponding one-for-one to a data vector.

**Source**:
A Firecrown model component that describes a projected observable and supplies one or more CCL tracers for a two-point statistic.
_Avoid_: source galaxy, galaxy sample

**CCL tracer**:
A `pyccl.Tracer` describing one contribution of a projected observable to an angular spectrum.
_Avoid_: Source, SACC tracer

**Systematic**:
An effect represented in a likelihood model because it changes the relation between the inferred observable and its idealized theoretical counterpart.
_Avoid_: modeling choice

**Modeling choice**:
A selected prescription or assumption used to predict an observable. A choice's placement in a `Source` or in `ModelingTools` does not by itself determine whether it is also a systematic.
_Avoid_: systematic

### 1.2 Galaxy Samples and Two-Point Measurements

**Tomographic bin**:
A named galaxy-bin descriptor containing a redshift grid, an inferred redshift distribution, and the measurement types available for that bin.
_Avoid_: inferred redshift distribution

**Inferred redshift distribution**:
The estimated distribution of true redshift for a selected population, sampled on a redshift grid; commonly represented as `dN/dz` or `n(z)`.
_Avoid_: tomographic bin

**Lens sample**:
A foreground galaxy population whose density or clustering provides the lensing or clustering field in a two-point analysis. The role does not by itself enforce redshift ordering.

**Source-galaxy sample**:
A background galaxy population whose shapes provide a weak-lensing shear measurement. The role does not by itself enforce redshift ordering or imply a Firecrown `Source`.
_Avoid_: Source

**Harmonic space**:
The representation of a two-point measurement by multipole coordinate `ell`.

**Real space**:
The representation of a two-point measurement by angular-separation coordinate `theta`.
_Avoid_: configuration space, angular-correlation space

**Scale cut**:
An inclusive rule that selects two-point measurements by the statistic's native coordinate: `ell` in harmonic space or `theta` in real space. For windowed bandpowers, an observation may also be retained or removed according to window support.
_Avoid_: physical-distance cut

### 1.3 SACC Data

**SACC structural metadata**:
Descriptors of SACC tracer identities, bin combinations, data types, layouts, coordinates, and indices.
_Avoid_: metadata for SACC measurements or calibration

**SACC measurement**:
An extracted measured value sequence together with its structural metadata and covariance indices.

**SACC calibration**:
Redshift-distribution inputs attached to SACC tracers, such as sampled `dN/dz` values.

**SACC covariance**:
The covariance matrix associated with SACC measurement entries.

## 2. Software Engineering Concepts

### 2.1 Public API and Release Compatibility

**Public API path**:
A Firecrown import path with no underscore-prefixed module component and a non-underscored exported name. In a public module, every non-underscored accessible name is public, including names imported from elsewhere; names listed in that module's `__all__` are explicitly public, and removing one from `__all__` is a public change even if it remains directly accessible.

**Support line**:
The series of `vX.Y.Z` releases associated with the `vx_y_support` branch for fixed numeric major and minor versions `X` and `Y`.

**API policy verdict**:
The target-dependent interpretation of a completed static public API comparison: no detected breaking changes, informational breaking changes on `master`, or rejection of detected breaking changes on a support target. The verdict is distinct from whether analysis and reporting succeeded; missing analysis or reporting is not evidence of compatibility, and no detected breaks does not establish complete behavioral compatibility.

### 2.2 Dependency and Environment Guarantees

**Dependency manifest**:
The authoritative declaration of Firecrown's dependency requirements, from which its generated dependency lists are derived.

**Dependency-list consistency**:
Agreement between Firecrown's generated dependency lists and the dependency manifest. This is distinct from whether those requirements can currently be satisfied.

**Environment rebuildability**:
The ability to freshly resolve and install Firecrown's declared dependency environment using currently available packages. This establishes neither runtime compatibility with Firecrown nor agreement with previously locked versions.

**Lock compatibility**:
Agreement between locked package selections and the dependency requirements applicable to each supported environment. Compatible selections need not be newly resolved or use the newest available package versions.

**Dependency consistency**:
Agreement between dependency declarations and committed artifacts, encompassing dependency-list consistency, project dependency and Python-requirement agreement with the dependency manifest, lock compatibility across all supported environments, and reproducibility of validated constraints from committed locks. It establishes neither environment rebuildability nor Firecrown runtime compatibility.

**Validated constraints**:
Dependency constraints derived from committed locked package selections. Reproducibility means those constraints agree with regeneration from the committed locks; a derived version range does not imply that every version within it was runtime-tested.

**PR rebuild canary**:
An advisory, technically non-blocking sample of environment rebuildability for a pull request, using a fresh resolve and installation on Linux/Python 3.12, whose failures require author investigation and reviewer acceptance of unresolved risk before merge. Its result establishes neither rebuildability for other supported environments, Firecrown runtime compatibility, nor agreement with locked versions.

### 2.3 SACC Test Data

**SACC fixture expectation**:
A reference redshift distribution or angular spectrum retaining the inputs supplied to SACC, or copies of those inputs, for comparison with extracted results.

**Synthetic SACC observation**:
A supplied test value for a recognized SACC observable, such as an angular power spectrum, angular correlation, or windowed bandpower. It is neither a survey measurement nor necessarily a theory prediction or physically plausible realization; a value with an unknown data-type label has no established observable interpretation.
