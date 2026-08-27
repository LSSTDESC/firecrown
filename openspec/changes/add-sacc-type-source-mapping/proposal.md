## Why

SACC does not encode Firecrown `TypeSource` values, so full extraction currently assigns `TypeSource.DEFAULT` to every projected field.
This prevents analyses from applying different source-model configurations to distinct SACC tracers and makes the intended source classification implicit.
The change adds explicit, externally configured mapping while preserving existing YAML behavior and supporting the repository's future model-layer refactoring.

This work also aligns with the incremental architecture split where shared metadata/model abstractions move toward Sunbird, concrete physics models move into domain packages (the existing `crow` package for cluster physics, and planned `owl` and `phoenix` packages), and Firecrown focuses on likelihood assembly and sampler integration.
The mapping is therefore designed as generic SACC-to-metadata binding, not as coupling to any concrete physics package.

## What Changes

- Add an optional tracer-to-`TypeSource` mapping to the full SACC extraction workflow.
- Expose the mapping under the YAML `data_source` configuration.
- Support all currently recognized SACC tracer types through a generic tracer-name mapping that can extend to future types.
- Preserve legacy behavior when the mapping is omitted: extracted fields use `TypeSource.DEFAULT` without mapping validation.
- Treat an explicitly supplied empty mapping as opt-in to validation; default validation reports unmapped relevant tracers as errors.
- Add a configurable warning mode that reports unmapped tracers and assigns `TypeSource.DEFAULT` instead of raising.
- Reject mapping entries whose tracer names are absent from the SACC object.
- Keep the deprecated indices-only extraction path limited to `TypeSource.DEFAULT`.
- Do not modify or extend the SACC format; the mapping remains external analysis configuration.
- Improve the `TypeSource` documentation and add a focused tutorial explaining source classification, mappings, validation modes, factory selection, and the full versus indices-only workflows.
- Leave unused-factory detection as a separate follow-up change.

## Capabilities

### New Capabilities

- `sacc-type-source-mapping`: Explicit analysis-controlled assignment and validation of `TypeSource` values during full SACC extraction.

### Modified Capabilities

- None.

## Impact

Affected areas include the full SACC metadata/data extraction APIs, the YAML-driven `DataSourceSacc`/`TwoPointExperiment` path, metadata documentation, tutorial content, and focused extraction/factory integration tests.
The change uses the existing `firecrown.metadata_types.TypeSource`, introduces no dependency on `crow`, `owl`, or `phoenix`, does not expand the deprecated indices-only API, and does not alter SACC files or their schema.
