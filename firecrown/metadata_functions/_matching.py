"""Utilities for matching tracer names and measurements."""

import itertools as it
import re

import firecrown.metadata_types as mdt
from firecrown.metadata_functions._type_defs import (
    TwoPointRealIndex,
    TwoPointHarmonicIndex,
)

ConventionType = tuple[re.Pattern, tuple[mdt.Measurement, ...]]
CMB_CONVENTION: ConventionType = (mdt.CMB_REGEX, mdt.CMB_TYPES)
"""Naming convention for CMB tracers: regex and valid measurement types."""

LENS_CONVENTION: ConventionType = (mdt.LENS_REGEX, mdt.GALAXY_LENS_TYPES)
"""Naming convention for lens (galaxy density) tracers: regex and valid measurement
types."""

SOURCE_CONVENTION: ConventionType = (mdt.SOURCE_REGEX, mdt.GALAXY_SOURCE_TYPES)
"""Naming convention for source (galaxy shear) tracers: regex and valid measurement
types."""

CONVENTIONS: list[ConventionType] = [CMB_CONVENTION, LENS_CONVENTION, SOURCE_CONVENTION]
"""All recognised naming conventions, each pairing a tracer regex with its valid
measurements."""

CONVENTION_PAIRS: list[tuple[ConventionType, ConventionType]] = list(
    it.combinations(CONVENTIONS, 2)
)
"""All pairs of naming conventions, used to check for cross-correlations."""


def match_name_type(  # noqa: C901
    tracer1: str,
    tracer2: str,
    a: mdt.Measurement,
    b: mdt.Measurement,
    require_convention: bool = False,
) -> tuple[bool, str, mdt.Measurement, str, mdt.Measurement]:
    """Use the naming convention to assign the right measurement to each tracer.

    Given two tracer names and two measurements, this function inspects the
    tracer names against the lens, source, and CMB naming conventions to
    determine the correct pairing of names and measurements. The returned names
    and measurements may be reordered relative to the inputs so that the lens
    (or CMB) tracer always comes first. The function also returns a boolean
    indicating whether we are able to apply the naming convention to verify that
    the right type is associated with the right tracer name.

    If the function is able to determine that the tracer names and the measurement
    types are not compatible, then an exception is raised.

    If the tracer names follow the convention but the measurement types are not
    consistent with those name, then an exception is raised.

    If the tracer names follow the convention and the measurement types are consistent
    with those names and the association is not ambiguous, the boolean value is `True`
    and the names and measurements are returned in the order consistent with the
    convention.

    If none of the above applies, and if `require_convention` is `False`,, then the
    boolean value is `False` and the names and measurements are returned in the
    original order.

    If none of the above applies and if `require_convention` is `True`, then if the
    tracer names match the *same* convention, then the returned boolean value is
    `False`, and we return the tracer names and measurements in the original order.

    Otherwise, an exception is raised.

    :param tracer1: the name of the first tracer.
    :param tracer2: the name of the second tracer.
    :param a: the measurement associated with ``tracer1``.
    :param b: the measurement associated with ``tracer2``.
    :param require_convetion: if ``True``, raise a :class:`ValueError` when
        the tracer names do not match any recognised naming convention.

    :return: a tuple of (matched, tracer_name1, measurement1,
        tracer_name2, measurement2), where the names and measurements have
        been reordered to satisfy the naming convention when necessary, and
        *matched* indicates that an unambiguous association between names and
        measurement types has been made.
    """
    for n1, n2 in ((tracer1, tracer2), (tracer2, tracer1)):
        for (rg1, type1), (rg2, type2) in CONVENTION_PAIRS:
            if rg1.match(n1) and rg2.match(n2):
                if a in type1 and b in type2:
                    return True, n1, a, n2, b
                if b in type1 and a in type2:
                    return True, n1, b, n2, a
                msg = (
                    f"The given names '{n1}' and '{n2}' match the conventions\n"
                    f"But the one or both of the specified types '{a}' and '{b}' do"
                    f" not match expectations for the associated name.\n"
                )
                raise ValueError(msg)
    if not require_convention:
        return False, tracer1, a, tracer2, b

    for conv in CONVENTIONS:
        rg, typ = conv
        if rg.match(tracer1) and rg.match(tracer2):
            if a in typ and b in typ:
                return False, tracer1, a, tracer2, b
            msg = "The types specified do not match the matched tracer names."
            raise ValueError(msg)

    msg = (
        "One or both names did not match the convention, "
        "but require_convention is True."
    )
    raise ValueError(msg)


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
    """Return the tracer names and measurements from a two-point index.

    The tracer names and measurements are extracted from the index using the
    naming convention enforced by :func:`match_name_type`. The names and
    measurements are reordered to satisfy the convention when necessary.

    :param index: a two-point index containing tracer names and a data type.

    :return: a tuple of (tracer_name1, measurement1, tracer_name2, measurement2).
    """
    a, b = mdt.MEASURED_TYPE_STRING_MAP[index["data_type"]]
    _, n1, a, n2, b = match_name_type(
        index["tracer_names"].name1,
        index["tracer_names"].name2,
        a,
        b,
        require_convention=True,
    )
    return n1, a, n2, b
