# Firecrown API, Dependency, and Release Language

Terms used when assessing Firecrown dependencies and the compatibility of releases and pull requests.

## Language

**Public API path**:
A Firecrown import path with no underscore-prefixed module component and a non-underscored exported name. In a public module, every non-underscored accessible name is public, including names imported from elsewhere; names listed in that module's `__all__` are explicitly public, and removing one from `__all__` is a public change even if it remains directly accessible.

**Support line**:
The series of `vX.Y.Z` releases associated with the `vx_y_support` branch for fixed numeric major and minor versions `X` and `Y`.

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

**API policy verdict**:
The target-dependent interpretation of a completed static public API comparison: no detected breaking changes, informational breaking changes on `master`, or rejection of detected breaking changes on a support target. The verdict is distinct from whether analysis and reporting succeeded; missing analysis or reporting is not evidence of compatibility, and no detected breaks does not establish complete behavioral compatibility.
