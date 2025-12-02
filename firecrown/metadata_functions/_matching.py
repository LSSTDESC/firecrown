"""Utilities for matching tracer names and measurements."""

import firecrown.metadata_types as mdt
from firecrown.metadata_functions._type_defs import (
    TwoPointRealIndex,
    TwoPointHarmonicIndex,
)


def match_name_type(
    tracer1: str,
    tracer2: str,
    a: mdt.Measurement,
    b: mdt.Measurement,
    require_convention: bool = False,
) -> tuple[bool, str, mdt.Measurement, str, mdt.Measurement]:
    """Use the naming convention to assign the right measurement to each tracer."""
    for n1, n2 in ((tracer1, tracer2), (tracer2, tracer1)):
        if mdt.LENS_REGEX.match(n1) and mdt.SOURCE_REGEX.match(n2):
            if a in mdt.GALAXY_SOURCE_TYPES and b in mdt.GALAXY_LENS_TYPES:
                return True, n1, b, n2, a
            if b in mdt.GALAXY_SOURCE_TYPES and a in mdt.GALAXY_LENS_TYPES:
                return True, n1, a, n2, b
            raise ValueError(
                "Invalid SACC file, tracer names do not respect "
                "the naming convention."
            )
    if require_convention:
        if mdt.LENS_REGEX.match(tracer1) and mdt.LENS_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b
        if mdt.SOURCE_REGEX.match(tracer1) and mdt.SOURCE_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b

        raise ValueError(
            f"Invalid tracer names ({tracer1}, {tracer2}) "
            f"do not respect the naming convention."
        )

    return False, tracer1, a, tracer2, b


def _make_two_point_xy_error_message(
    data_type: str,
    a: mdt.Measurement,
    b: mdt.Measurement,
    tracer_names: mdt.TracerNames,
    igz1: mdt.InferredGalaxyZDist,
    igz2: mdt.InferredGalaxyZDist,
) -> str:
    """Generate a detailed error message for SACC naming convention violations.

    :param data_type: The data type string that was requested.
    :param a: The first expected measurement type.
    :param b: The second expected measurement type.
    :param tracer_names: The tracer names that were provided.
    :param igz1: The inferred galaxy z distribution for the first tracer.
    :param igz2: The inferred galaxy z distribution for the second tracer.
    :return: A formatted error message explaining the violation.
    """
    return f"""
Tracer measurements do not match the SACC naming convention:

  Data type: {data_type}
  Expected measurements: ({a}, {b})
  Tracer '{tracer_names[0]}' has measurements: {igz1.measurements}
  Tracer '{tracer_names[1]}' has measurements: {igz2.measurements}

According to the SACC convention, the order of measurement types in the data type
string must match the order of tracers. The measurement type '{a}' should be associated
with tracer '{tracer_names[0]}', and '{b}' should be associated with tracer
'{tracer_names[1]}'.

For detailed information about fixing this issue, see:
  https://firecrown.readthedocs.io/en/latest/sacc_usage.html
""".strip()


def make_two_point_xy(
    inferred_galaxy_zdists_dict: dict[str, mdt.InferredGalaxyZDist],
    tracer_names: mdt.TracerNames,
    data_type: str,
) -> mdt.TwoPointXY:
    """Build a mdt.TwoPointXY object from the inferred galaxy z distributions.

    The mdt.TwoPointXY object is built from the inferred galaxy z distributions,
    the data type, and the tracer names.

    :param inferred_galaxy_zdists_dict: a dictionary of inferred galaxy z
        distributions.
    :param tracer_names: a tuple of tracer names.
    :param data_type: the data type.

    :return: a mdt.TwoPointXY object.
    :raises ValueError: If the tracer measurements do not match the data type
        specification according to the SACC naming convention.
    """
    a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]

    igz1 = inferred_galaxy_zdists_dict[tracer_names[0]]
    igz2 = inferred_galaxy_zdists_dict[tracer_names[1]]

    if (a not in igz1.measurements) or (b not in igz2.measurements):
        if (a in igz2.measurements) and (b in igz1.measurements):
            # swap the order of the tracers, this is a temporary fix following
            # _should_swap_tracers_for_convention in _extraction.py
            igz1, igz2 = igz2, igz1
        else:
            raise ValueError(
                _make_two_point_xy_error_message(
                    data_type, a, b, tracer_names, igz1, igz2
                )
            )
    return mdt.TwoPointXY(x=igz1, y=igz2, x_measurement=a, y_measurement=b)


def measurements_from_index(
    index: TwoPointRealIndex | TwoPointHarmonicIndex,
) -> tuple[str, mdt.Measurement, str, mdt.Measurement]:
    """Return the measurements from a TwoPointXiThetaIndex object."""
    a, b = index["tracer_types"]
    return index["tracer_names"].name1, a, index["tracer_names"].name2, b
