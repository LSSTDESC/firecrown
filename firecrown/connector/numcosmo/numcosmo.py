"""Backward-compatibility shim for the former ``numcosmo.numcosmo`` module path.

Installed versions of ``numcosmo_py`` (>=0.27,<0.28, the range pinned by this
project) import the connector's public names directly from
``firecrown.connector.numcosmo.numcosmo``:

* ``numcosmo_py.external.cosmosis`` does
  ``from firecrown.connector.numcosmo.numcosmo import NumCosmoFactory, MappingNumCosmo``
  unconditionally.
* ``numcosmo_py.app`` does ``import firecrown.connector.numcosmo.numcosmo`` to
  trigger GObject registration, guarded by ``except ImportError``.

This module exists solely to keep those external imports working. It is not
part of Firecrown's supported public API: new code, including code within
this repository, should import from :mod:`firecrown.connector.numcosmo`
directly. Remove this shim once the minimum supported ``numcosmo_py`` version
no longer imports this path.
"""

from firecrown.connector.numcosmo import (
    MappingNumCosmo,
    NumCosmoData,
    NumCosmoFactory,
    NumCosmoGaussCov,
    create_params_map,
    helpers,
)

# The compatibility shim intentionally mirrors the package's public exports.
# pylint: disable=duplicate-code
__all__ = [
    "MappingNumCosmo",
    "NumCosmoData",
    "NumCosmoGaussCov",
    "NumCosmoFactory",
    "create_params_map",
    "helpers",
]
# pylint: enable=duplicate-code
