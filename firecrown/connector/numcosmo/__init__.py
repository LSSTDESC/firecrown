"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.
"""

from numcosmo_py import GObject

from firecrown.connector.numcosmo._data import NumCosmoData, NumCosmoGaussCov
from firecrown.connector.numcosmo._factory import NumCosmoFactory
from firecrown.connector.numcosmo._mapping import MappingNumCosmo
from firecrown.connector.numcosmo._parameters import create_params_map

# These commands creates GObject types for the defined classes, enabling their use
# within the NumCosmo framework. It is essential to call these functions before
# initializing NumCosmo with the Ncm.init_cfg() function, as failure to do so
# will cause issues with MPI jobs using these objects.
GObject.type_register(MappingNumCosmo)
GObject.type_register(NumCosmoData)
GObject.type_register(NumCosmoGaussCov)

__all__ = [
    "MappingNumCosmo",
    "NumCosmoData",
    "NumCosmoGaussCov",
    "NumCosmoFactory",
    "create_params_map",
]
