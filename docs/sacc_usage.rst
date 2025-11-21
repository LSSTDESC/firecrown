.. _sacc_usage:

SACC Usage and Conventions
==========================

This guide explains how Firecrown uses the SACC format, the naming conventions that
must be followed, and how to handle and fix common issues.

What is SACC?
-------------

`SACC <https://sacc.readthedocs.io/en/latest/>`_ (Save All Correlations and
Covariances) is a standardized format for storing two-point correlation function
measurements from cosmological surveys. SACC files organize measurements from multiple
tracers (e.g., tomographic bins) and provide a consistent interface for accessing data
and metadata.

Key Features of SACC:

- **Tracers**: Represent tomographic bins (e.g., galaxies in a redshift range, CMB temperature map)
- **Data Points**: Individual measurements of correlations between pairs of tracers
- **Metadata**: Including covariances, window functions, and measurement types
- **Data Types**: Standardized strings that identify what kind of measurement is stored
  (e.g., ``galaxy_shear_xi_plus``)

SACC Naming Convention
----------------------

Firecrown enforces a strict **naming convention** for the association between
measurement types and tracers. This convention is essential for unambiguous
interpretation of two-point measurements.

Convention Rules
~~~~~~~~~~~~~~~~

The fundamental rule is:

    **The order of measurement types in a SACC data type string must match the order of
    the tracers.**

Example 1: Single Measurement Type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the data type ``galaxy_shear_xi_plus`` measured for tracers ``bin_0`` and
``bin_1``:

.. code-block:: text

    Data Type:   galaxy_shear_xi_plus
    Tracers:     (bin_0, bin_1)
    Meaning:     Both bin_0 and bin_1 are galaxy shear measurements

In this case, both tracers represent the same measurement type (galaxy shear), so there
is no ambiguity.

Example 2: Two Different Measurement Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the data type ``galaxy_shearDensity_cl`` measured for tracers ``shear_bin_0``
and ``density_bin_1``:

.. code-block:: text

    Data Type:   galaxy_shearDensity_cl
    Tracers:     (shear_bin_0, density_bin_1)
    
The naming convention requires:

- The **first** measurement type in the data type string (``shear``) corresponds to the
  **first** tracer (``shear_bin_0``)
- The **second** measurement type in the data type string (``Density``) corresponds to
  the **second** tracer (``density_bin_1``)

It would be an error to have them reversed (``density_bin_1``, ``shear_bin_0``), we SACC
follows a canonical ordering.

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

Understanding Measurement Types
-------------------------------

Firecrown recognizes the following measurement types:

**Galaxy Measurements:**

- ``SHEAR_E`` or ``SHEAR_T``: Weak lensing shear (ellipticity or tangential components)
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

**Single-Type Measurements**: Both tracers have the same measurement type (e.g., galaxy
shear auto-correlation).

**Two-Type Measurements**: The two tracers have different measurement types (e.g.,
galaxy shear cross-correlation with CMB convergence).

When Firecrown processes a SACC file, it determines the measurement type for each
tracer by analyzing all measurements involving that tracer. Ideally:

- Each tracer should be associated with exactly one measurement type
- Measurements should be consistent with the SACC naming convention

If this is not the case, Firecrown will either automatically correct the assignment
(deprecated behavior) or raise an error.

Handling Mixed-Type Measurements
---------------------------------

In most cases, a tracer should be associated with only one measurement type. However,
in specialized analyses, you might intentionally want a single tomographic bin to
contain measurements of multiple types (mixed-type measurements).

Example: A tracer might contain both shear and count measurements, and you want to use
both in a joint analysis.

Using ``allow_mixed_types`` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When loading SACC data with :py:func:`extract_all_measured_types`, you can control
whether mixed-type measurements are permitted:

.. code-block:: python

    from firecrown.metadata_functions import extract_all_measured_types
    
    # Strict mode (default): raises error if a tracer has multiple measurement types
    measured_types = extract_all_measured_types(sacc_data, allow_mixed_types=False)
    
    # Permissive mode: allows mixed-type measurements
    measured_types = extract_all_measured_types(sacc_data, allow_mixed_types=True)

By default, ``allow_mixed_types=False``, which raises a ``ValueError`` if a tracer is
found to have multiple measurement types. This strict behavior helps identify
inconsistencies in SACC files that don't follow the convention.

Fixing SACC Convention Violations
----------------------------------

Common Issues
~~~~~~~~~~~~~

The most common SACC convention violation occurs when tracers are misaligned with
measurement types:

**Scenario**: You have a galaxy shear tracer (``shear_bin_0``) and a CMB convergence
tracer (``cmb_bin_0``), but they are reversed in one of your data type strings.

.. code-block:: text

    Correct:     (shear_bin_0, cmb_bin_0)  with data type "galaxy_shearCMB_convergence_cl"
    Incorrect:   (cmb_bin_0, shear_bin_0)  with same data type "galaxy_shearCMB_convergence_cl"

In the incorrect case, the data type string claims the first tracer is a shear
measurement, but it's actually CMB convergence.

Deprecated Auto-Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``allow_mixed_types=False`` (the default), Firecrown will attempt to auto-correct
simple convention violations by swapping tracer labels. This is done as a convenience
for legacy SACC files that don't follow the convention.

**Important**: This auto-correction behavior is **deprecated** and will be removed in a
future release. You should fix your SACC file instead of relying on this
auto-correction.

When auto-correction is performed, Firecrown issues a ``DeprecationWarning`` with
detailed information:

.. code-block:: text

    SACC Convention Violation Detected (DEPRECATED AUTO-FIX)
    
    Firecrown detected an inconsistency in how measurement types are assigned to tracers.
    ...
    AUTO-CORRECTION PERFORMED
    Because allow_mixed_types=False (the default), Firecrown attempted to correct this by
    swapping the tracer assignment...
    
    ⚠️  DEPRECATION NOTICE ⚠️
    This automatic correction will be REMOVED in a future release...
    
    RECOMMENDED ACTION
    To future-proof your code, fix your SACC file...

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
                              "cmb_bin_0",      # First tracer (should be shear)
                              "shear_bin_0",    # Second tracer (should be cmb)
                              ell=..., value=...)

To fix it, swap the tracer order:

.. code-block:: python

    # Correct: tracers are (shear, cmb) matching the data type
    sacc_data.add_data_point("galaxy_shearCMB_convergence_cl",
                              "shear_bin_0",    # First tracer (shear) ✓
                              "cmb_bin_0",      # Second tracer (cmb) ✓
                              ell=..., value=...)

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
               print("⚠️  Convention violations detected:")
               for warning in w:
                   print(f"  {warning.message}")

Troubleshooting
---------------

Error: "Tracer 'X' has multiple measurement types"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: A single tracer is associated with multiple measurement types, either due to:

- Convention violations with auto-correction disabled
- Intentional mixed-type measurements without ``allow_mixed_types=True``

**Solution**:

1. Check if the violation is intentional (true mixed-type measurement)
2. If intentional, set ``allow_mixed_types=True`` in your call to ``extract_all_measured_types``
3. If unintentional, fix your SACC file following the guidelines above

DeprecationWarning About Auto-Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Firecrown detected and auto-corrected a SACC convention violation.

**Solution**:

1. Understand the specific violation mentioned in the warning
2. Fix your SACC file permanently using the guidelines in "How to Fix Your SACC File"
3. Test that the fixed file works without warnings

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
