"""Example configuration for Firecrown."""

import dataclasses

from . import logging


@dataclasses.dataclass
class List(logging.Logging):
    """List available examples."""
