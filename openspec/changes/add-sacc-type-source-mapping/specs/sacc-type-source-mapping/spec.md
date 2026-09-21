## Purpose

Allow analyses to explicitly classify SACC tracers with `TypeSource` values during full extraction, while preserving legacy behavior and avoiding any change to the SACC format.

## ADDED Requirements

### Requirement: Full extraction accepts external tracer mappings

The full real-space and harmonic-space SACC extraction workflows SHALL accept an optional mapping from SACC tracer names to `TypeSource` values. Mapping values supplied as strings SHALL be normalized to `TypeSource` values, and the mapping SHALL apply generically to every supported SACC tracer type.

#### Scenario: Mapped tracers receive their configured source

- **WHEN** full extraction is given a mapping containing the names of SACC tracers
- **THEN** each resulting projected field for those tracers has the corresponding `TypeSource` value

#### Scenario: Mapping supports non-galaxy tracers

- **WHEN** full extraction processes a supported non-galaxy SACC tracer whose name appears in the mapping
- **THEN** the resulting projected field receives the mapped `TypeSource` without requiring a tracer-type-specific mapping API

### Requirement: Mapping configuration is external to SACC

The tracer-to-`TypeSource` mapping SHALL be accepted as external analysis configuration and SHALL NOT require new fields in, or modifications to, SACC files.

#### Scenario: YAML data source supplies the mapping

- **WHEN** a YAML data source configuration contains a tracer-to-`TypeSource` mapping
- **THEN** the SACC-backed likelihood workflow uses that mapping during full extraction

#### Scenario: Existing YAML omits the mapping

- **WHEN** an existing YAML data source configuration has no tracer mapping
- **THEN** the workflow preserves the legacy behavior of assigning `TypeSource.DEFAULT` without mapping validation

### Requirement: Mapping validation is configurable

When a mapping is explicitly supplied, the system SHALL validate that mapping keys name tracers present in the SACC object and SHALL validate omitted relevant tracer names according to a configurable validation mode. The default mode SHALL be `error`; a `warning` mode SHALL emit warnings and assign `TypeSource.DEFAULT` to omitted tracers.

#### Scenario: Explicit empty mapping activates default validation

- **WHEN** an empty mapping is supplied without an explicit validation mode
- **THEN** extraction reports all relevant SACC tracers omitted from the mapping as an error

#### Scenario: Warning mode permits omitted tracers

- **WHEN** a mapping is supplied with warning validation and relevant tracers are omitted
- **THEN** extraction emits warnings identifying the omitted tracers and assigns `TypeSource.DEFAULT` to them

#### Scenario: Unknown mapping key is rejected

- **WHEN** a supplied mapping contains a tracer name absent from the SACC object
- **THEN** extraction raises an error identifying the unknown mapping key

#### Scenario: Complete mapping succeeds

- **WHEN** every relevant SACC tracer is present in the supplied mapping
- **THEN** extraction succeeds without missing-mapping errors or warnings

### Requirement: Legacy indices-only extraction remains default-only

The deprecated indices-only extraction workflow SHALL continue to construct metadata using `TypeSource.DEFAULT` and SHALL NOT gain support for non-default tracer mappings in this change.

#### Scenario: Indices-only extraction ignores non-default mapping capability

- **WHEN** users use the deprecated indices-only workflow
- **THEN** it remains limited to `TypeSource.DEFAULT` and the full-extraction mapping API is not required

### Requirement: Source classification is documented

The documentation SHALL explain `TypeSource` as an opaque, analysis-defined identifier used to distinguish tracers requiring different modeling treatment, and SHALL document full extraction mappings, validation modes, the legacy omission behavior, and the indices-only limitation.

#### Scenario: Focused factory tutorial demonstrates mapping

- **WHEN** a user follows the focused factory tutorial
- **THEN** the tutorial shows how to assign distinct `TypeSource` values to SACC tracers and configure matching factories

#### Scenario: Tutorial explains validation choices

- **WHEN** a user reads the SACC and factory documentation
- **THEN** the documentation distinguishes an omitted mapping from an explicit empty mapping and explains error versus warning validation
