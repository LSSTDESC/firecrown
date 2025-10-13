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
    require_convetion: bool = False,
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
                "the naming convetion."
            )
    if require_convetion:
        if mdt.LENS_REGEX.match(tracer1) and mdt.LENS_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b
        if mdt.SOURCE_REGEX.match(tracer1) and mdt.SOURCE_REGEX.match(tracer2):
            return False, tracer1, a, tracer2, b

        raise ValueError(
            f"Invalid tracer names ({tracer1}, {tracer2}) "
            f"do not respect the naming convetion."
        )

    return False, tracer1, a, tracer2, b


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
    """
    a, b = mdt.MEASURED_TYPE_STRING_MAP[data_type]

    igz1 = inferred_galaxy_zdists_dict[tracer_names[0]]
    igz2 = inferred_galaxy_zdists_dict[tracer_names[1]]

    ab = a in igz1.measurements and b in igz2.measurements
    ba = b in igz1.measurements and a in igz2.measurements
    if a != b and ab and ba:
        raise ValueError(
            f"Ambiguous measurements for tracers {tracer_names}. "
            f"Impossible to determine which measurement is from which tracer."
        )
    XY = mdt.TwoPointXY(
        x=igz1, y=igz2, x_measurement=a if ab else b, y_measurement=b if ab else a
    )

    return XY


def measurements_from_index(
    index: TwoPointRealIndex | TwoPointHarmonicIndex,
) -> tuple[str, mdt.Measurement, str, mdt.Measurement]:
    """Return the measurements from a TwoPointXiThetaIndex object."""
    a, b = mdt.MEASURED_TYPE_STRING_MAP[index["data_type"]]
    _, n1, a, n2, b = match_name_type(
        index["tracer_names"].name1,
        index["tracer_names"].name2,
        a,
        b,
        require_convetion=True,
    )
    return n1, a, n2, b
