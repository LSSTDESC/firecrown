"""Type definitions for two-point metadata handling."""

from typing import TypedDict

import firecrown.metadata_types as mdt


class TwoPointRealIndex(TypedDict):
    """Intermediate object for reading SACC real-space two-point data.

    Internal use only - not intended for direct user interaction.
    """

    data_type: str
    tracer_names: mdt.TracerNames


class TwoPointHarmonicIndex(TypedDict):
    """Intermediate object for reading SACC harmonic-space two-point data.

    Internal use only - not intended for direct user interaction.
    """

    data_type: str
    tracer_names: mdt.TracerNames
