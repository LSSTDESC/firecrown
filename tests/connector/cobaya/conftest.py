"""Shared configuration for the Cobaya connector tests."""

from collections.abc import Iterator

import pytest
from cobaya import mpi


@pytest.fixture(autouse=True, scope="session")
def disable_cobaya_mpi() -> Iterator[None]:
    """Disable Cobaya's MPI support for the Cobaya connector tests.

    Cobaya initializes MPI through mpi4py on first use, and MPI initialization aborts
    the process on hosts whose network transport cannot be opened. These tests do not
    use MPI.
    """
    mpi.set_mpi_disabled(True)
    yield
    mpi.set_mpi_disabled(False)
