"""Utilities for creating and manipulating Measurement objects."""

from typing import Any

import firecrown.metadata_types as mdt


def make_measurement(value: mdt.Measurement | dict[str, Any]) -> mdt.Measurement:
    """Create a Measurement object from a dictionary."""
    if isinstance(value, mdt.ALL_MEASUREMENT_TYPES):
        return value

    if not isinstance(value, dict):
        raise ValueError(f"Invalid Measurement: {value} is not a dictionary")

    if "subject" not in value:
        raise ValueError("Invalid Measurement: dictionary does not contain 'subject'")

    subject = value["subject"]

    match subject:
        case "Galaxies":
            return mdt.Galaxies[value["property"]]
        case "CMB":
            return mdt.CMB[value["property"]]
        case "Clusters":
            return mdt.Clusters[value["property"]]
        case _:
            raise ValueError(
                f"Invalid Measurement: subject: '{subject}' is not recognized"
            )


def make_measurements(
    value: set[mdt.Measurement] | list[dict[str, Any]],
) -> set[mdt.Measurement]:
    """Create a Measurement object from a dictionary."""
    if isinstance(value, set) and all(
        isinstance(v, mdt.ALL_MEASUREMENT_TYPES) for v in value
    ):
        return value

    measurements: set[mdt.Measurement] = set()
    for measurement_dict in value:
        measurements.update([make_measurement(measurement_dict)])
    return measurements


def make_measurement_dict(value: mdt.Measurement) -> dict[str, str]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return {"subject": type(value).__name__, "property": value.name}


def make_measurements_dict(value: set[mdt.Measurement]) -> list[dict[str, str]]:
    """Create a dictionary from a Measurement object.

    :param value: the measurement to turn into a dictionary
    """
    return [make_measurement_dict(measurement) for measurement in value]


def make_correlation_space(value: mdt.TwoPointCorrelationSpace | str):
    """Create a CorrelationSpace object from a string."""
    if not isinstance(value, mdt.TwoPointCorrelationSpace) and isinstance(value, str):
        try:
            return mdt.TwoPointCorrelationSpace(
                value.lower()
            )  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for TwoPointCorrelationSpace: {value}"
            ) from exc
    return value
