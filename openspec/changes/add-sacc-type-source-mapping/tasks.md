## 1. Mapping and validation model

- [ ] 1.1 Define the public mapping and validation-mode types using the existing `TypeSource` and preserve omitted-versus-empty mapping semantics.
- [ ] 1.2 Implement generic validation of mapping keys against SACC tracer names, including aggregate errors for unknown keys and missing relevant tracers.
- [ ] 1.3 Implement warning-mode fallback that emits tracer-specific warnings and assigns `TypeSource.DEFAULT` to omitted tracers.

## 2. Full SACC extraction

- [ ] 2.1 Thread the optional tracer-to-`TypeSource` mapping and validation mode through full harmonic and real extraction APIs.
- [ ] 2.2 Apply mapped values when constructing every supported projected-field type without coupling the mapping mechanism to concrete physics factories.
- [ ] 2.3 Preserve the deprecated indices-only extraction path as `TypeSource.DEFAULT`-only and add regression coverage for its unchanged behavior.

## 3. YAML integration

- [ ] 3.1 Add the mapping and validation configuration under `DataSourceSacc` with strict Pydantic/YAML validation.
- [ ] 3.2 Forward YAML data-source settings through `TwoPointExperiment` and SACC-backed likelihood construction.
- [ ] 3.3 Add end-to-end tests for omitted mapping, explicit empty mapping, complete mapping, warning mode, and unknown mapping keys.

## 4. Documentation and tutorials

- [ ] 4.1 Update the `TypeSource` class docstring to describe it as an opaque, analysis-defined source-category identifier, including current examples such as red/blue galaxies and Planck/SPT CMB data.
- [ ] 4.2 Add a focused factory-system tutorial showing `TypeSource`, full SACC mapping, matching source factories, inspection, and validation modes.
- [ ] 4.3 Update the existing factory and SACC tutorials to explain external mappings, omitted-versus-empty behavior, and the indices-only `DEFAULT`-only limitation.
- [ ] 4.4 Cross-link the focused tutorial from the existing two-point workflow/tutorial navigation and verify documentation examples.

## 5. Verification and refactoring boundary

- [ ] 5.1 Add focused extraction and factory integration tests covering all currently supported SACC tracer categories.
- [ ] 5.2 Run the relevant test and documentation checks, confirming no SACC schema changes or physics-package dependencies were introduced.
- [ ] 5.3 Record unused-factory detection as a separate follow-up issue or planning item rather than implementing it here.
