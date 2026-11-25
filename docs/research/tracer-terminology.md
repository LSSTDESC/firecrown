# Tracer terminology in Firecrown, SACC, and CCL

## Summary

“Tracer” names three different objects in the Firecrown two-point pipeline:

| Term | Role | What it contains or represents |
| --- | --- | --- |
| **SACC tracer** | Data-side identity | A named SACC object that data points refer to. Depending on its subtype, it may contain an $n(z)$, a beam, cluster-bin metadata, or only a name. |
| **Firecrown `Tracer`** | Firecrown theory-side component | A wrapper around one `pyccl.Tracer`, with additional 3D-field, perturbation-theory, halo-model, and component-name information. |
| **CCL tracer** | Theory-side projection | A `pyccl.Tracer` describing one or more linearly combined projected contributions used to calculate angular power spectra. |

A Firecrown **`Source`** is a related but distinct fourth concept. It is a
model component associated with one SACC tracer name. It creates one or more
Firecrown `Tracer` objects from that SACC data and its modeling
configuration.

The usual galaxy or CMB two-point relationship is:

```text
SACC tracer -> ProjectedField -> Source -> Firecrown Tracer(s)
                                      -> one pyccl.Tracer per Firecrown Tracer
```

This mapping does not cover every Firecrown `Statistic`. Supernova and cluster
statistics use SACC tracers without using a `Source` or a CCL tracer.

## Evidence

### SACC tracers

SACC defines a tracer as a named object associated with astronomical data.
Data points refer to tracers by name, and a two-point data point has a
combination of two tracer names.

Important subtypes include:

- `NZTracer`, which stores a tomographic redshift distribution;
- `MapTracer`, which stores map metadata such as a beam and map unit;
- `MiscTracer`, which has only a name and tags miscellaneous data;
- survey and cluster-bin tracers used for cluster analyses.

Primary sources:

- [SACC tracers documentation](https://sacc.readthedocs.io/en/latest/tracers.html)
- [`BaseTracer` source](https://github.com/LSSTDESC/sacc/blob/master/sacc/tracers/base.py)
- [SACC data types documentation](https://sacc.readthedocs.io/en/latest/data_types.html)
- [SACC container API](https://sacc.readthedocs.io/en/latest/sacc.html)

Therefore, a SACC tracer is not, by itself, an observation or a complete
forward model. It is a named data-side identity and may also hold calibration
or bin metadata.

### CCL tracers

CCL describes tracers as projected quantities that can be cross-correlated to
calculate angular power spectra. A `pyccl.Tracer` describes contributions of
a sky observable through ingredients including radial kernels, transfer
functions, ell-dependent prefactors, and Bessel-derivative orders.

A CCL `Tracer` object can contain a list of such contributions that are combined
linearly. Thus, calling a CCL tracer “one contribution” is slightly too narrow:
it represents a projected quantity and can contain multiple contributions.

The inputs to `pyccl.angular_cl` are two `pyccl.Tracer` objects:

```text
angular_cl(cosmology, tracer1, tracer2, ell, ...)
```

Primary sources:

- [CCL tracers API](https://ccl.readthedocs.io/en/latest/api/pyccl.tracers.html)
- [CCL `Tracer` API](https://ccl.readthedocs.io/en/latest/api/pyccl.tracers.html#pyccl.tracers.Tracer)
- [CCL tracer implementation](https://ccl.readthedocs.io/en/latest/_modules/pyccl/tracers.html)
- [CCL angular power spectra API](https://ccl.readthedocs.io/en/latest/api/pyccl.cells.html#pyccl.cells.angular_cl)

CCL supports more than number-count and lensing tracers. For example, CCL also
provides tSZ, CIB, and ISW tracers. A CCL tracer therefore cannot be defined
universally as a population tracing the matter density field.

### Firecrown tracers

Firecrown's `Tracer` is neither a SACC tracer nor a raw CCL tracer. It wraps a
`pyccl.Tracer` and adds Firecrown-specific information:

- a component name such as `shear`, `galaxies`, `rsd`, or
  `cmb_convergence`;
- the associated 3D field;
- optional perturbation-theory tracer information;
- optional halo-model profile and two-point information.

See:

- [`Source` and `Tracer`](../../firecrown/likelihood_base.py)
- [`TwoPoint`](../../firecrown/likelihood/_two_point.py)
- [two-point theory model](../../firecrown/models/two_point/_theory.py)

A `Source` obtains a SACC tracer by its configured name, reads relevant
calibration or systematics information, and creates one or more Firecrown
`Tracer` objects. A single `Source` may therefore produce several Firecrown
components. Examples include:

- weak-lensing shear, intrinsic alignment modeled perturbatively, and
  halo-model intrinsic alignment;
- galaxy number counts, magnification, and redshift-space distortions;
- CMB lensing convergence.

`TwoPoint` pairs the Firecrown tracers created by two `Source` objects and uses
their embedded `pyccl.Tracer` objects to calculate angular spectra.

## Assessment of the cosmological usage

In broad cosmological prose, “tracer” can reasonably mean a population or
projected field whose distribution carries information about cosmic structure.
Galaxies, quasars, clusters, line-intensity fields, and weak-lensing fields are
common examples. This broad scientific use explains terms such as “tracer
bias,” “tomographic tracer,” and “lens/source tracers.”

That broad meaning is useful, but it is not identical to any single software
object in Firecrown, SACC, or CCL:

- A **SACC tracer** is a data identity and possible calibration container.
- A **CCL tracer** is a theory-side projection model.
- A **Firecrown `Tracer`** is the wrapper that links CCL inputs to additional
  Firecrown model metadata.
- A **Firecrown `Source`** is the configurable component that creates
  Firecrown `Tracer` objects.

The statement that a tracer “defines a distinct data vector and forward model”
is too broad for the software terminology. A SACC tracer does not itself
define a data vector or forward model. In Firecrown, a `Statistic` exposes the
data and theory vectors, while one or more `Source` and `Tracer` objects may
participate in constructing a two-point theory vector.

Also, the idea that every tracer traces dark matter should not be part of the
definition. It is a useful property of some large-scale-structure tracers, but
CCL and SACC both admit tracers that are not naturally described this way.

## Additional meanings of “source”

“Source” has several independent meanings:

1. A Firecrown `Source` model component.
2. A source-galaxy sample, in contrast to a lens sample.
3. The astrophysical source category in a SACC data-type string.
4. A source plane, such as the CMB lensing source plane denoted by `z_source`.

These meanings must be qualified. In particular, a Firecrown `Source` may
represent either a lens sample or a source-galaxy sample.

## Canonical glossary vocabulary

### SACC tracer

A named, data-side object in a SACC file that identifies a tracer or external
data category and that SACC data points refer to by name. It may carry
calibration or bin metadata, such as an inferred redshift distribution, a map
beam, survey information, or cluster-bin coordinates. It may also carry no
attributes beyond its name.

### SACC tracer name

The string identifying a SACC tracer. Firecrown's `Source.sacc_tracer` and
supernova `sacc_tracer` arguments hold this name, not a CCL tracer object.

### CCL tracer

A `pyccl.Tracer` describing the projection of a sky observable for use in
angular spectra. It may contain one or more contributions that are combined
linearly by CCL.

### Firecrown `Tracer`

A Firecrown wrapper around one CCL tracer, with a component name and optional
3D-field, perturbation-theory, and halo-model information.

### Tracer component name

The name of a contribution represented by a Firecrown `Tracer`, such as
`shear`, `galaxies`, `magnification`, or `rsd`. This is not a SACC tracer
name.

### Projected field

Firecrown metadata derived from a SACC tracer for use in constructing a
two-point `Source`. Examples include a `TomographicBin` and `CMBLensing`.
A projected field is not itself a SACC tracer, CCL tracer, or Firecrown
`Tracer`.

## Recommended language rule

Use unqualified **tracer** only when the surrounding audience has already
established a single meaning. In API documentation and tutorials, first use:

- **SACC tracer** or **SACC tracer name** for SACC-side objects and strings;
- **CCL tracer** for `pyccl.Tracer`;
- **Firecrown `Tracer`** for the Firecrown wrapper;
- **tracer component name** for values such as `shear` or `galaxies`;
- **Source** for the Firecrown model component.

Do not call a `Source` a tracer, and do not call a SACC tracer a CCL tracer.

## Current documentation concerns

High-priority ambiguities include:

1. [`likelihood_base.py`](../../firecrown/likelihood_base.py) uses “tracer” in
   several methods without stating whether it means a SACC tracer name or a
   Firecrown `Tracer`.
2. [`_two_point.py`](../../firecrown/likelihood/_two_point.py) uses
   `TracerNames` for both SACC tracer-name pairs and Firecrown component-name
   pairs.
3. [`_two_point_tracers.py`](../../firecrown/metadata_types/_two_point_tracers.py)
   still contains references to an obsolete `Tracer` protocol; the relevant
   current concept is `ProjectedField`.
4. Tutorial prose sometimes says “tracers are what you have observed,” which
   incorrectly identifies SACC tracer objects with measured values.
5. Tutorial prose sometimes says a `Source` produces `pyccl.Tracer` objects.
   Directly, it produces Firecrown `Tracer` wrappers containing
   `pyccl.Tracer` objects.
6. [`docs/sacc_usage.rst`](../sacc_usage.rst) describes SACC tracers as
   tomographic bins or matter-distribution tracers. This excludes several SACC
   tracer subtypes and conflates a software identity with a scientific
   population.

## Mapping limitations

- **Supernovae:** use a SACC `MiscTracer`, but no Firecrown `Source` or CCL
  tracer.
- **Clusters:** use survey and cluster-bin SACC tracers, but no
  Firecrown `Source`.
- **Multi-component sources:** one SACC tracer and one `Source` can produce
  multiple Firecrown `Tracer` objects.
- **Result keys:** `TwoPoint` component results are keyed by Firecrown tracer
  component names, while the associated SACC tracers have different names.

## Research limitations

The external pages and source were checked at their published `latest`
documentation and current primary-source branches. No Firecrown behavior was
executed for this terminology study; the Firecrown mappings were established
by reading the public docstrings and implementation.
