"""Utilities for loading and handling SACC data files.

.. deprecated::
    The SACC loading helpers now live in :mod:`firecrown.utils`. This module
    re-exports them for backward compatibility and may be removed in a future
    version. Import ``load_sacc_data`` and ``ensure_path`` from
    ``firecrown.utils`` instead.
"""

from firecrown.utils import ensure_path, load_sacc_data

__all__ = ["load_sacc_data", "ensure_path"]
