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
