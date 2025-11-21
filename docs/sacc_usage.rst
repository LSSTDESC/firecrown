.. _sacc_usage:

SACC Usage and Conventions
==========================

This guide explains how Firecrown uses the SACC format, the naming conventions that
must be followed, and how to handle and fix common issues.

What is SACC?
-------------

`SACC <https://sacc.readthedocs.io/en/latest/>`_ (Save All Correlations and
Covariances) is a standardized format for storing measurements from cosmological surveys,
including two-point correlation functions, SNIa observations, cluster counts, and other
related cosmological probes. SACC files organize measurements from multiple tracers
(e.g., tomographic bins) and provide a consistent interface for accessing data and metadata.

**Note**: This guide focuses on the naming conventions and best practices for
**two-point measurements** specifically. While SACC supports diverse measurement
types, the conventions described here are tailored to two-point galaxy correlations.

Key Features of SACC:

- **Tracers**: Represent tomographic bins or other matter distribution tracers (e.g., galaxies in a redshift range, CMB temperature map)
- **Data Points**: Individual measurements of correlations between pairs of tracers
- **Metadata**: Including covariances, window functions, and measurement types
- **Data Types**: Standardized strings that identify what kind of measurement is stored
  (e.g., ``galaxy_shear_xi_plus``)

SACC Naming Convention
----------------------

SACC defines a strict **naming convention** for the association between measurement
types and tracers. This convention is essential for unambiguous interpretation of
two-point measurements.

**Important Notice**: Firecrown **will enforce this naming convention strictly in a
future release**. Currently, Firecrown provides compatibility features to allow loading
older SACC files that do not fully comply with the convention. However, if you use such
non-compliant files, you must be careful to understand the implications:

- Non-compliant files may produce unexpected results if tracers and measurement types
  are misaligned
- Auto-correction features are **deprecated** and will be removed
- You should update your SACC files to comply with the convention
- Future versions of Firecrown will reject non-compliant SACC files without an
  auto-correction option

Convention Rules
~~~~~~~~~~~~~~~~

The fundamental rule is:

    **The order of measurement types in a SACC data type string must match the order of
    the tracers.**

Example 1: Single Measurement Type (Same Type for Both Tracers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the two tracers have the **same measurement type**, the measurement type name
appears only **once** in the data type string:

Consider the data type ``galaxy_shear_xi_plus`` measured for tracers ``bin_0`` and
``bin_1``:

.. code-block:: text

    Data Type:   galaxy_shear_xi_plus
    Tracers:     (bin_0, bin_1)
    Meaning:     Both bin_0 and bin_1 are galaxy shear measurements
    
Notice that "shear" appears only once in the data type string, even though both tracers
are shear measurements. This is the naming convention for measurements between two tracers
of the same type. Other examples include:

- ``galaxy_density_xi`` - density auto-correlation
- ``galaxy_shear_cl_ee`` - shear-shear correlation in harmonic space

In these cases, there is no ambiguity about which tracer corresponds to which measurement
type, since both tracers have the same type.

**Important**: The tracer names (``bin_0``, ``bin_1``) are **not** used to determine
measurement types. The measurement type is determined entirely from the data type string
and validated against the tracers involved. You could name tracers anything
(e.g., ``alpha``, ``beta``) and the convention would still apply—both would still be
shear measurements.

Example 2: Two Different Measurement Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the two tracers have **different measurement types**, both measurement type names
appear in the data type string, in a specific order that must match the tracer order.

Consider the data type ``galaxy_shearDensity_cl`` measured for tracers ``bin_0``
and ``bin_1``:

.. code-block:: text

    Data Type:   galaxy_shearDensity_cl
    Tracers:     (bin_0, bin_1)
    
The naming convention requires:

- The **first** measurement type in the data type string (``shear``) corresponds to the
  **first** tracer (``bin_0``)
- The **second** measurement type in the data type string (``Density``) corresponds to
  the **second** tracer (``bin_1``)

It would be an error to have them reversed (``bin_1``, ``bin_0``), as SACC
follows a canonical ordering (see next section).

**Note**: Again, the tracer names here (``bin_0``, ``bin_1``) are arbitrary and chosen
only to illustrate the concept. The measurement type determination is based on the data
type string and the order of tracers in the measurement, not on their names.

Canonical Ordering in Firecrown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SACC additionally enforces a **canonical ordering** of measurement types:

    **CMB < Clusters < Galaxies**

This ordering applies when combining measurements across multiple types. For example:

- CMB convergence measurements always come before cluster measurements
- Cluster measurements always come before galaxy measurements
- Within each category, measurements are ordered by tracer name

This canonical ordering ensures deterministic ordering regardless of how measurements
are specified in the SACC file.

**Internal Enum Ordering**: Within each measurement type category (CMB, Clusters, Galaxies),
the measurement types are also ordered canonically. For example, within ``Galaxies``:

    **Galaxies.SHEAR_E < Galaxies.COUNTS**

This means that when determining tracer order for cross-measurements (e.g., shear
cross-correlation with counts), shear measurements always come before count measurements.
Similarly, CMB and cluster measurement types are ordered internally according to their
relative positions in their respective enums. This internal ordering is applied
automatically by Firecrown to ensure consistency.

Understanding Measurement Types
-------------------------------

Firecrown recognizes the following measurement types:

**Galaxy Measurements:**

- ``SHEAR_E`` or ``SHEAR_T``: Weak lensing shear components
- ``COUNTS``: Galaxy number counts
- ``PART_OF_XI_MINUS`` or ``PART_OF_XI_PLUS``: Components of two-point shear
  correlation functions

**CMB Measurements:**

- ``CONVERGENCE``: CMB lensing convergence

**Cluster Measurements:**

- ``COUNTS``: Cluster abundance

Measurement type associations are stored in the SACC file via the data type strings.
Firecrown uses the ``MEASURED_TYPE_STRING_MAP`` to decode which measurement types are
involved in each measurement.

Single-Type vs. Two-Type Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Measurement Type Sets**: In Firecrown, measurements are organized into sets based on their
conceptual role. For galaxies, these include:

- ``GALAXY_SOURCE_TYPES``: Shear measurements (``SHEAR_E``, ``SHEAR_T``, ``PART_OF_XI_MINUS``, ``PART_OF_XI_PLUS``)
- ``GALAXY_LENS_TYPES``: Galaxy counts (``COUNTS``)
- ``CMB_TYPES``: CMB convergence (``CONVERGENCE``)
- ``CLUSTER_TYPES``: Cluster abundance

A tracer may have multiple measurement types **as long as they belong to the same measurement set**.
For example, a galaxy source tracer might have both ``SHEAR_T`` and ``PART_OF_XI_MINUS`` measurements,
since both are part of ``GALAXY_SOURCE_TYPES``. However, mixing types from different sets
(e.g., shear and counts) in the same tracer is not allowed by default.

**Swap Detection and Auto-Correction (Deprecated)**: When Firecrown processes a SACC file,
it may detect that a tracer was mistakenly assigned to the wrong measurement set. If tracers
are reversed (e.g., a source tracer mixed with a lens tracer) in a measurement, Firecrown will:

1. Test if swapping the tracers would fix the issue
2. If a swap corrects the assignment, emit a ``DeprecationWarning`` and perform the swap
3. Apply the corrected measurement types to the tracers

**Important**: This auto-correction via tracer swapping is **deprecated** and will be removed
in a future release. When strict enforcement is enabled:

- If a tracer has mixed types from different sets and ``allow_mixed_types=False``, Firecrown
  will **raise an error** instead of attempting to auto-correct
- You should fix your SACC file to ensure tracers are properly ordered according to the
  measurement types in the data type string

**Single-Type Measurements**: When both tracers belong to the same measurement set and have
compatible types (e.g., both are sources or both are lenses), there is typically no ambiguity.

**Two-Type Measurements**: When the two tracers belong to different measurement sets or have
incompatible types (e.g., source paired with lens), the order of tracers must match the order
of measurement types in the data type string, as explained in the convention examples above.

Handling Mixed-Type Measurements
---------------------------------

**What are Mixed-Type Measurements?**: In the context of Firecrown, "mixed-type measurements"
refers to a tracer being associated with measurement types from **different measurement sets**
(e.g., both source types and lens types, or galaxy measurements mixed with CMB measurements).

**Within the same set is allowed**: A tracer **may** have multiple measurement types as long
as they belong to the same measurement set. For example:

- A source tracer can have both ``SHEAR_T`` and ``PART_OF_XI_MINUS`` (both in ``GALAXY_SOURCE_TYPES``)
- A lens tracer can only have ``COUNTS`` (the only type in ``GALAXY_LENS_TYPES``)

However, mixing types from **different sets** in the same tracer (e.g., shear and counts) is
not permitted by default.

**Use Cases for Mixed-Type Measurements**: In specialized analyses, you might intentionally
want to allow a single tracer to contain measurements from different sets, though this is
unusual. For example, a tracer might contain both galaxy shear and galaxy count measurements
in a joint analysis.

Using ``allow_mixed_types`` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When loading SACC data with :py:func:`extract_all_measured_types`, you can control
whether mixed-type measurements (types from different sets) are permitted:

.. code-block:: python

    from firecrown.metadata_functions import extract_all_measured_types
    
    # Strict mode (default): raises error if a tracer has types from different sets
    measured_types = extract_all_measured_types(sacc_data, allow_mixed_types=False)
    
    # Permissive mode: allows tracers to have types from different measurement sets
    measured_types = extract_all_measured_types(sacc_data, allow_mixed_types=True)

By default, ``allow_mixed_types=False``, which raises a ``ValueError`` if a tracer is
found to have measurement types from different sets. This strict behavior helps identify
potential inconsistencies and ensures proper separation between source and lens tracers.

**Note**: If ``allow_mixed_types=False`` and Firecrown detects reversed tracers that could
be fixed by swapping, it will emit a ``DeprecationWarning`` and perform the swap. However,
if swapping cannot fix the issue, an error will be raised. This auto-correction via tracer
swapping is deprecated and will be removed in a future release.

Fixing SACC Convention Violations
----------------------------------

Common Issues
~~~~~~~~~~~~~

The most common SACC convention violation occurs when tracers are misaligned with
measurement types due to reversed tracer order:

**Scenario**: You have a CMB convergence tracer (``cmb_bin_0``) and a galaxy counts
tracer (``lens_bin_0``), but they are reversed in one of your data type strings.

.. code-block:: text

    Correct:     (cmb_bin_0, lens_bin_0)  with data type "cmbGalaxy_convergenceDensity_cl"
    Incorrect:   (lens_bin_0, cmb_bin_0)  with same data type "cmbGalaxy_convergenceDensity_cl"

The data type string ``cmbGalaxy_convergenceDensity_cl`` specifies that the measurement
is CMB convergence paired with galaxy counts, and the canonical ordering requires CMB to
come first (CMB < Galaxies). In the incorrect case, the tracers are reversed: the first 
tracer is galaxy counts, but the data type string claims it should be CMB convergence.

**Detection and Auto-Correction (Deprecated)**: Firecrown can detect this error when other
measurements involving these tracers provide evidence of the correct ordering. For example,
if you have auto-correlations like:

- ``cmb_bin_0`` with ``cmb_bin_0`` in a ``cmb_convergence_cl`` measurement
- ``lens_bin_0`` with ``lens_bin_0`` in a ``galaxy_density_cl`` measurement

Then Firecrown can infer the correct types for each tracer and detect that the
cross-measurement tracers are reversed. When this happens with ``allow_mixed_types=False``,
Firecrown will:

1. Emit a ``DeprecationWarning`` explaining the issue
2. Automatically swap the tracers to fix the ordering
3. Continue processing with the corrected assignment

**Important**: This automatic tracer swapping is deprecated and will be removed in a future
release. You should fix your SACC file manually rather than relying on this auto-correction.

Deprecated Auto-Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``allow_mixed_types=False`` (the default), Firecrown will attempt to auto-correct
simple convention violations by swapping tracer labels. This is done as a convenience
for legacy SACC files that don't follow the convention.

**Important**: This auto-correction behavior is **deprecated** and will be **removed in a
future release** when Firecrown enforces strict SACC compliance. You should fix your SACC
file now instead of relying on this auto-correction. Once strict enforcement is enabled,
non-compliant SACC files will cause Firecrown to raise an error, and users will have to
manually fix their files.

The auto-correction process works as follows:

1. Firecrown detects that assigning the measurement types in their original tracer order
   would create mixed-type measurements (measurements with types from different sets)
2. Firecrown tests whether swapping the tracers would resolve the issue
3. If swapping would fix the problem, Firecrown performs the swap and emits a
   ``DeprecationWarning`` with detailed diagnostic information
4. Processing continues with the corrected tracer assignment

When auto-correction is performed, Firecrown issues a ``DeprecationWarning`` structured as
follows:

.. code-block:: text

    SACC Convention Violation Detected (DEPRECATED AUTO-FIX)
    
    Firecrown detected an inconsistency in how measurement types are assigned to tracers.
    Specifically, assigning measurement type 'SHEAR_E' to tracer 'lens0' and measurement
    type 'COUNTS' to tracer 'src0' would create mixed-type measurements (multiple distinct
    measurement types from different sets in the same tomographic bin).
    
    The data type string 'galaxy_shearDensity_cl' follows the SACC naming convention, where
    the order of measurement types in the string must match the order of tracers. However,
    your SACC file/object appears to violate this convention.
    
    AUTO-CORRECTION PERFORMED
    Because allow_mixed_types=False (the default), Firecrown attempted to correct this by
    swapping the tracer assignment, assuming the tracers were simply misaligned. This auto-
    correction is a convenience feature for legacy SACC files that don't follow the
    convention.
    
    ⚠️  DEPRECATION NOTICE ⚠️
    This automatic correction will be REMOVED in a future release. Going forward, files
    that violate the SACC convention will be interpreted as genuinely mixed-type
    measurements and will either raise an error (if allow_mixed_types=False) or be
    processed as-is (if allow_mixed_types=True).
    
    RECOMMENDED ACTION
    To future-proof your code, fix your SACC file to follow the naming convention. See the
    documentation for detailed instructions:
        https://firecrown.readthedocs.io/en/latest/sacc_usage.html

How to Fix Your SACC File
~~~~~~~~~~~~~~~~~~~~~~~~~~

To fix SACC convention violations, you need to reorder tracers in your data type
entries so that they match the measurement type order in the data type string.

**Manual Approach:**

1. Identify which measurements have convention violations (look for
   ``DeprecationWarning`` messages)
2. Understand which measurement types correspond to which tracers
3. Reorder tracers in the affected data type entries to match the measurement type
   order in the data type string
4. Save the corrected SACC file

**Example Fix:**

Suppose you have this incorrect entry:

.. code-block:: python

    # Incorrect: tracers are (cmb, shear) but data type implies (shear, cmb)
    sacc_data.add_data_point("galaxy_shearCMB_convergence_cl",
                              ("cmb_bin_0",      # First tracer (should be shear)
                              "shear_bin_0"),    # Second tracer (should be cmb)
                              ...)

To fix it, swap the tracer order:

.. code-block:: python

    # Correct: tracers are (shear, cmb) matching the data type
    sacc_data.add_data_point("galaxy_shearCMB_convergence_cl",
                              ("shear_bin_0",    # First tracer (shear) ✓
                              "cmb_bin_0"),      # Second tracer (cmb) ✓
                              ...)

**Programmatic Approach:**

For large SACC files with many violations, a dedicated repair tool (to be provided in a
future release) will automate this process. In the meantime, you can write a Python
script to:

1. Load the SACC file
2. Identify violations using Firecrown's extraction functions
3. Reorder tracer entries as needed
4. Save the corrected file

Best Practices
--------------

1. **Follow the Convention from the Start**
   
   When creating SACC files, follow the naming convention from the beginning:
   
   - Use clear, descriptive tracer names that indicate their measurement type
   - Order tracers in data type entries to match the order in the data type string
   - Test your SACC file with Firecrown using ``allow_mixed_types=False`` to validate
     compliance
   - When writing a SACC file you can use Firecrown ordering support to determine automatically the correct order.

2. **Use Descriptive Tracer Names**
   
   Make it obvious what each tracer represents:
   
   .. code-block:: text
   
       Good:    shear_bin_0, cmb_convergence_bin_0, cluster_counts_bin_0
       Unclear: tracer_0, tracer_1, bin_0

3. **Document Your SACC File**
   
   Include comments in your code describing:
   - What each tracer represents
   - Why mixed-type measurements are used (if applicable)
   - Any deviations from the standard convention

4. **Validate Early**
   
   When loading a SACC file, always validate it:
   
   .. code-block:: python
   
       import warnings
       from firecrown.metadata_functions import extract_all_measured_types
       
       # Catch deprecation warnings about convention violations
       with warnings.catch_warnings(record=True) as w:
           warnings.simplefilter("always", DeprecationWarning)
           measured_types = extract_all_measured_types(sacc_data, allow_mixed_types=False)
           
           if w:
               print("Convention violations detected:")
               for warning in w:
                   print(f"  {warning.message}")

Getting Help
------------

For more information about SACC:

- `SACC Documentation <https://sacc.readthedocs.io/>`_
- `SACC GitHub Repository <https://github.com/LSSTDESC/sacc>`_

For Firecrown-specific questions:

- :py:func:`firecrown.metadata_functions.extract_all_measured_types`
- :py:func:`firecrown.metadata_functions.extract_all_tracers_inferred_galaxy_zdists`
- Firecrown GitHub Issues and Discussions

See Also
--------

- :doc:`basic_usage` - Using Firecrown with your data
- :doc:`dev-notes` - Developer notes on the metadata system
