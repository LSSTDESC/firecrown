## Context

The current full SACC extraction path constructs projected fields with `TypeSource.DEFAULT` because SACC does not encode Firecrown `TypeSource` values. The existing metadata classes already carry `TypeSource`, while the YAML-driven SACC workflow centralizes data configuration in `DataSourceSacc`. The deprecated indices-only workflow intentionally exposes only tracer names and measurement types and will remain unchanged.

The change must support the incremental Sunbird/Firecrown refactoring: it uses the current `firecrown.metadata_types.TypeSource`, does not introduce physics-package dependencies, and keeps SACC interpretation separate from concrete source factories or future owl, CROW, and phoenix implementations.

## Goals / Non-Goals

**Goals:**

- Add one generic tracer-name mapping mechanism shared by harmonic and real full extraction.
- Make the mapping available through direct extraction APIs and YAML under `data_source`.
- Preserve existing configurations that omit the mapping.
- Make explicit mapping configuration auditable through error or warning validation.
- Support all projected-field tracer kinds through the same mapping interface.
- Explain the feature through a focused factory tutorial and updates to existing SACC/factory documentation.

**Non-Goals:**

- Detecting configured factories that are never used; that is a separate follow-up from issue #547.
- Extending the deprecated indices-only extraction workflow.
- Inferring `TypeSource` from tracer naming conventions.
- Modifying the SACC format or storing the mapping in SACC files.
- Adding Sunbird, owl, CROW, or phoenix dependencies or migrating physics classes in this change.

## Decisions

1. **Use the existing `TypeSource` value type.** The new API will continue to use `firecrown.metadata_types.TypeSource`. It is an opaque, analysis-defined string-like label; callers instantiate values or provide strings that are normalized at the boundary. Subclassing `TypeSource` is not part of the design.

2. **Keep mapping at the SACC extraction boundary.** The mapping will be accepted by the metadata/projected-field extraction layer and forwarded by the full data extraction functions. This lets metadata carry source classification without coupling extraction to `TwoPointFactory`, `NumberCountsFactory`, `WeakLensingFactory`, or any future physics package.

3. **Configure the YAML mapping under `data_source`.** The mapping describes interpretation of a particular SACC data source, so it belongs next to `sacc_data_file`, not inside `two_point_factory`. `DataSourceSacc` will carry the mapping and validation mode into the SACC-backed likelihood builder.

4. **Use an omitted-versus-empty distinction for compatibility and opt-in validation.** If the mapping field is omitted, the workflow follows the legacy implicit-default behavior and performs no mapping validation. If the mapping is explicitly present, including as `{}`, mapping mode is active; its default validation mode is `error`.

5. **Make unknown mapping keys unconditional errors.** A mapping entry naming no SACC tracer is a configuration error and will raise in both validation modes. In contrast, a known SACC tracer omitted from the mapping is an incomplete classification: error in `error` mode, warning plus `TypeSource.DEFAULT` in `warning` mode.

6. **Validate against SACC tracer names generically.** The mapping mechanism will validate names against the SACC tracer collection, not against a hard-coded list of tracer classes. Projected-field construction will apply the mapping when each supported SACC tracer type is converted to metadata; future supported types can reuse the same mapping contract.

7. **Keep indices-only extraction default-only.** Adding mapping data to index dictionaries or changing `TwoPoint.from_metadata_index` would expand a deprecated API and complicate its compatibility surface. The new capability is limited to full metadata/data extraction.

8. **Use current terminology in tutorials.** Documentation will use `TypeSource` directly and explain its meaning, rather than introduce a parallel neutral public name. Examples will distinguish source classification from factory selection while showing the current factory system.

## Risks / Trade-offs

- **[Risk] Existing YAML behavior could change accidentally.** -> Treat an omitted mapping as a distinct legacy mode and add regression tests for existing configurations.
- **[Risk] Warning mode could hide an omitted classification.** -> Include tracer names in warnings, document that error mode is the default for explicit mappings, and demonstrate explicit empty mappings as an audit mechanism.
- **[Risk] Mapping validation could become coupled to current tracer classes.** -> Validate names generically and keep conversion-specific behavior in the projected-field extraction layer.
- **[Risk] Users may infer `TypeSource` from tracer names.** -> Do not implement automatic naming conventions; require explicit configuration when mapping mode is enabled.
- **[Risk] YAML and direct APIs could diverge.** -> Use the same extraction options and validation logic for both entry points and cover both with tests.
- **[Risk] The feature could make future package migration harder.** -> Keep the mapping independent of concrete physics implementations and use only the existing metadata type that is already a planned migration boundary.
