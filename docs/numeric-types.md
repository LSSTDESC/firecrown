# Numeric Types for Firecrown Likelihood Authors

This guide describes the numeric-type policy Firecrown intends to support for
physicists writing likelihoods, statistics, and source systematics. It assumes
familiarity with Python and NumPy, but not with Firecrown's internal implementation.

**Policy status:** this is the intended contract, not a claim that every released
version already implements it. As of October 18, 2026, the parameter-array getter
work is complete in this checkout; the remaining dtype-policy issues are still
pending. Check the documentation for your installed version. Constructing your
own scientific arrays explicitly as described here avoids relying on implicit
conversion in an older version.

## Choose the Type by Scientific Meaning

Use double precision, `np.float64`, for real-valued scientific calculations. Keep
exact integers, selections, and names in representations appropriate to their
meaning. A redshift can be supplied as an integer without becoming a discrete
quantity; conversely, an array position does not become a real-valued measurement
just because it is stored in a floating-point array.

| Quantity | Intended representation | Examples |
| --- | --- | --- |
| Real scientific values | `np.float64` | Redshift, scale factor, distance, mass, radius, power spectra, shear, distributions, bias, alignment amplitudes, window weights, uncertainties, covariance, predictions |
| Exact harmonic multipoles | `np.int64` | Integer support values of ell in an exact-multipole route |
| Fixed-width SACC/domain indices | `np.int64` | Positions selecting measurements or covariance rows |
| Native NumPy indices | `np.intp` | Results of sorting or finding selected positions, and local signed row maps |
| Genuine discrete counts | `np.int64` where fixed-width storage is required | Observed numbers of objects in a discrete-count interface |
| Selection masks | `np.bool_` | Scale cuts and valid-row selections |
| Identifiers | Strings; `np.str_` for string arrays | Tracer names and SACC data-type names |

Collections of statistic objects remain Python objects, not numerical data. An
object array containing numbers is nevertheless not a supported physical-number
input. The type policy is not a reason to convert every array to `float64`.

## Dtype Conversion Is Not Normalization

In the [glossary](../GLOSSARY.md), **normalization** means rescaling a distribution
so that its integral over its domain is one. **Dtype conversion** means converting
numerical values to a consistent numeric type, such as `float64`, without
intentionally changing their physical scale. NumPy calls an array's stored
element type its **dtype**.

Converting a redshift distribution to `float64` does not make its integral one.
Converting a distance array does not change its units. Distribution normalization,
unit conversion, and dtype conversion are separate operations. Dtype conversion
can still round values when the destination type cannot represent them exactly.

## Convert Before Calculating

The dtype of a NumPy array affects its arithmetic. Convert real scientific inputs
before subtracting, summing, squaring, interpolating, integrating, or updating
them in place. Converting only the final result cannot repair earlier overflow
or rounding loss.

For example, a measurement uncertainty might arrive as an integer array. The
variance of an uncertainty of 50,000 is 2,500,000,000, which is too large for a
signed 32-bit integer:

```python
import numpy as np

sigma = np.array([50_000], dtype=np.int32)

wrong_variance = (sigma**2).astype(np.float64)  # Integer overflow happens first.
sigma_real = sigma.astype(np.float64)
variance = sigma_real**2  # Double-precision arithmetic from the start.
```

The same principle applies to sums of `float32` window weights, photo-z
interpolation, point-mass integrals, and residuals between data and theory. Start
with `float64` before the precision-sensitive work, then reuse those arrays
rather than repeatedly converting them inside the calculation.

For data that you know are valid real numbers, explicit construction is simple:

```python
z = np.array([0.0, 0.5, 1.0], dtype=np.float64)
scale_factor = 1.0 / (1.0 + z)

# The amplitude is real even when its coordinate prototype happens to be integer.
integer_z = np.array([0, 1, 2], dtype=np.int64)
alignment = np.full_like(integer_z, 0.25, dtype=np.float64)
```

Without the explicit dtype, a `full_like` buffer based on integer coordinates
would truncate the fractional amplitude. Specify `dtype=np.float64` for real
`zeros_like`, `ones_like`, and `full_like` buffers as well.

When generating a grid from lower-precision scalar endpoints, convert the
endpoints before constructing the grid, not just its output:

```python
z_min = np.float32(0.1)
z_max = np.float32(1.7)
z_grid = np.linspace(np.float64(z_min), np.float64(z_max), 100, dtype=np.float64)
```

This uses the supplied endpoint values at double precision; it cannot restore
digits already rounded away when those endpoints were stored as `float32`.

## What You May Supply

At interfaces documented as accepting flexible real scientific input, the intended
policy accepts integer and real floating-point arrays and converts them to
`float64` for storage and calculation. Examples include supported observation,
source-distribution, and provider input paths. This is not a promise that every
Firecrown method accepts every numeric dtype: parameter getters and exact integer
interfaces have different contracts, described below.

Physical-number inputs must not be Boolean arrays, complex arrays, numeric-looking
strings, object arrays, structured records, or datetime/timedelta arrays. Do not
use a cast to disguise unsupported input. In particular, converting complex data
to real data by discarding the imaginary part is not supported likelihood science.
If your analysis defines a real observable from complex intermediate quantities,
that must be an explicit scientific operation before supplying the real observable.

For masked arrays or masked table columns, follow the interface's explicit
missing-data policy. If it has none, masked numerical input is rejected rather
than silently using hidden values. An ordinary selection mask is different: it
is a Boolean array specifying which rows to include. When removing rows yourself,
apply the same selection coherently to measurements, coordinates, and covariance.

For FITS, Astropy, or other table data, select the appropriate numerical columns
and resolve units and missing-data meaning before handing them to the likelihood.
Do not assume a file reader or an external library has supplied the correct dtype.
Conversion may yield an ordinary NumPy array; third-party array subclass identity
is not guaranteed.

### Precision Limits

`float64` is double precision, not exact arithmetic. It provides roughly 15 to 16
significant decimal digits. Converting `float32` input cannot recover its lost
digits; `float64` represents every integer up to magnitude `2**53` exactly, but
not every larger integer. Extended-precision input may also lose information on
conversion, and its precision varies by platform.

Accepting lower-precision input does not mean the likelihood supports computation
in `float32` or extended precision. This policy does not add complex-valued
likelihoods, GPU arrays, or alternate array backends.

## Data and Theory Vectors

`DataVector` represents observations; `TheoryVector` represents predictions. Keep
that distinction when writing a statistic. Their intended scientific storage is
`float64`, and supported constructors are intended to establish it before residual
arithmetic. Explicitly supplying `float64` also works without waiting for that
constructor conversion to be implemented:

```python
from firecrown.data_types import DataVector, TheoryVector, residuals

data = DataVector.create(np.array([1.25, 2.5], dtype=np.float64))
theory = TheoryVector.create(np.array([1.0, 2.75], dtype=np.float64))
delta = residuals(data, theory)  # Ordinary float64 array: [0.25, -0.25].
```

A NumPy dtype `view` reinterprets the existing bytes and can produce unrelated
numerical values; it is not a value conversion. A vector-class view changes the
array's class without changing its dtype. Neither form of view, nor a Python type
annotation, performs the required dtype conversion.

Already-conforming arrays may share storage where the interface supports it.
Converting a different dtype may allocate a new array. Do not assume that every
constructor or getter copies, or that all of them share. Consult that interface's
ownership contract; take an explicit copy when you need independent mutable data.

## Multipoles, Indices, and Counts

### Exact Multipoles Versus Real Labels

An exact harmonic multipole and a fractional bin label are different quantities.
Store exact multipoles as integer arrays. Store real sampling coordinates,
effective window labels, and beam coordinates as `float64` where the selected
route supports their meaning:

```python
ells = np.array([2, 3, 4], dtype=np.int64)
effective_ells = np.array([20.5, 45.75], dtype=np.float64)
```

Bandpower-window support is exact and requires an integer dtype; `[2.0, 3.0, 4.0]`
is not an integer representation of that support. Another integer dtype may require
a range-checked conversion. Unwindowed SACC multipoles and generated harmonic
bin centers are sampling coordinates and may be `float64`, including fractional
values. Keep those values through angular-Cl evaluation and diagnostics. Do not
round them to make them fit an exact-multipole route. SACC indices remain integer
regardless of how the multipoles are represented.

### Indices and Masks

Keep fixed-width SACC indices integer and retain native NumPy sorting/selection
results as `np.intp` until an interface requires another representation. Construct
empty index arrays with an explicit integer dtype:

```python
indices = np.array([0, 2], dtype=np.int64)
empty_indices = np.array([], dtype=np.int64)
keep = np.array([True, False, True], dtype=np.bool_)
measurements = np.array([1.25, 2.5, 3.75], dtype=np.float64)
selected = measurements[keep]  # [1.25, 3.75].
```

Floating zero/one arrays are not Boolean masks. Fractional indices are invalid;
casting them to integers merely hides the error. Range, dimensionality, and index
bounds still need validation. Local signed maps may use `-1` to mean no association;
retain that sentinel and exclude unassociated entries before using such a map to
index an array.

### Observed Counts Versus Expected Counts

Use integers when an interface represents exact observed counts. Model-predicted
counts, integrated fractions, means, standard deviations, variances, and covariance
are real quantities and use `float64`. A prediction of 12.7 objects in a bin is
not rounded to 13 just because its name contains "count."

Preserve discrete-count identity where required, with range checks for integer
conversion. Any scientific conversion from a real result to a discrete count must
follow that interface's explicit validated rule. `int(value)` alone is not
validation. Convert count uncertainties to `float64` before squaring them to
construct a covariance, even if their supplied values happen to be whole numbers.

## Distributions, Windows, and Scientific Conventions

Redshift distributions, photo-z transformations, galaxy-bias and alignment
amplitudes, and point-mass integrands use `float64`. Dtype conversion does not
change their distribution normalization, sample grids, extrapolation, or
integration method. For supernovae, convert real redshift data before calculating
scale factors and distance moduli. For cluster profiles, do the same before
logarithms, integrations, and statistical calculations.

`TomographicBin`, `SourceGalaxyArgs`, `NumberCountsArgs`, and `WeakLensingArgs`
accept real integer and floating arrays for redshift coordinates and distribution
values, converting them to float64 storage before source calculations. The public
active and passive photo-z transforms also accept these representations and
convert before weighted averages and interpolation. Conforming arrays may share
storage, including strided arrays; converting other real dtypes requires storage
and may round values. SACC source reads retain their copy, flatten, and sort
behavior. These boundaries reject masked arrays even when every mask entry is
false, since there is no missing-data policy. Tabulated bias and alignment fields
have float64 contracts and are validated during source-argument construction.
Tracer preparation supplies double-valued arrays to pyccl; external callers do
not have to supply float64 redshifts to Firecrown's flexible ingress interfaces.

Window support multipoles remain exact integers; window weights and effective
labels are real. `TwoPointHarmonic`, `TwoPointReal`, and `TwoPointMeasurement`
accept real integer and floating window weights, effective labels, angular
coordinates, and measured values, and store them as `float64`. Boolean, complex,
string, object, and masked arrays are rejected. SACC extraction converts window
weights to `float64` before the zero-total check and column sums in both modes, so
reduced-precision weights are never accumulated at their input precision. This
also converts big-endian arrays, such as those read from FITS files, to native
`float64`. Measurement values and real angular coordinates are converted at the
same ingress. The raw-buffer covariance identifier is still computed from the
SACC covariance in its original representation.
An existing option named `normalize` controls **sum-to-one window rescaling**, not
dtype conversion. When that option is false, preserve the supplied weight scale;
conversion to `float64` is not permission to divide by the sum. Keep the existing
definition of effective labels rather than introducing a new weighted-mean formula.

NumPy covariance-correlation calculations and pyccl angular correlations are
different operations. Some explicitly generic utilities can compare non-real
arrays or compute with complex values; that does not extend the real likelihood
contract. Correlation display remains real-only and must not silently discard
imaginary information.

## Parameters and External Calculations

`NamedParameters` array getters retrieve an existing representation; they do not
convert one numeric type into another. `get_float_array` requires native
`np.float64` storage; `get_int_array` requires native `np.int64`. A mismatch raises
`TypeError`, even if an integer could numerically be written as a float:

```python
from firecrown.likelihood import NamedParameters

parameters = NamedParameters(
    {
        "amplitudes": np.array([0.25, 0.5], dtype=np.float64),
        "ells": np.array([2, 3], dtype=np.int64),
    }
)
amplitudes = parameters.get_float_array("amplitudes")
parameter_ells = parameters.get_int_array("ells")
# parameters.get_float_array("ells") would raise TypeError, not convert the array.
```

These getters retain shared storage, so mutating the returned array affects the
stored parameter. Prepare the desired representation before constructing the
parameters. Non-native byte-order arrays also need appropriate conversion first.
Supported basic-dictionary conversion keeps integer and floating sequences
distinct. Boolean sequences in this particular parameter interface become integer
zero/one arrays; selection masks elsewhere remain Boolean. An empty basic sequence
has the existing `float64` representation, so use an explicit `int64` array when
you need an empty integer parameter. Basic-dictionary array serialization flattens
arrays; it is not a general mechanism for preserving arbitrary shapes or dtypes.

For direct pyccl or SciPy calculations in your own likelihood, use `float64` real
inputs before the calculation, with the documented units, shapes, and ordering.
An external routine accepting a dtype does not establish its physical validity
or precision. Firecrown's double-valued library arguments and real results follow
the same double-precision policy; native provider APIs keep their own interfaces.
Do not change stored exact multipoles to floats merely because a library needs a
temporary double-valued argument, or to a smaller integer type because a connector
needs it. Integer narrowing must be range checked at the interface that needs it.

Covariance identifiers are representation-dependent, not hashes of an abstract
mathematical matrix. Numerically equal matrices stored differently need not have
the same identifier. Do not rely on dtype conversion preserving an identifier or
use it to rewrite source covariance buffers solely to obtain a matching hash.

## Check Your Likelihood

- Verify values as well as actual dtypes through data reading, parameter updates,
  preparation, and prediction or likelihood evaluation.
- Exercise valid integer and lower-precision real inputs at interfaces intended
  to accept them, and verify independent double-precision expectations.
- Check that unsupported input is rejected rather than silently cast, and that
  masks, indices, identifiers, and discrete counts retain their meaning.
- Keep scientific shape, units, ordering, positivity, finite-value, and covariance
  validation appropriate to your model; a correct dtype establishes none of these.
- Check fractional amplitudes and nonzero residuals so that integer truncation or
  precision loss cannot hide behind a zero-result test.
- Investigate numerical differences rather than broadly loosening tolerances.
  Double precision does not change your model, algorithm, bins, or error budget.

The practical rule is: **convert real scientific values to double precision before
doing arithmetic, while preserving the distinct meaning of integers, masks, and
identifiers. Normalize a distribution only when the scientific model calls for it.**
