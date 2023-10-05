"""Cluster Mass Richness proxy module

Define the Cluster Mass Richness proxy module and its arguments.
"""

from ..parameters import ParamsMap
from enum import Enum
from .mass_observable import *

MassObservableType = Enum("MassObservableType", "TRUE MU_SIGMA MURATA COSTANZI")


class MassObservableFactory:
    @staticmethod
    def create(mass_observable_type: MassObservableType, params: ParamsMap):
        if mass_observable_type == MassObservableType.TRUE:
            return TrueMass(params)
        elif mass_observable_type == MassObservableType.MU_SIGMA:
            return MassRichnessMuSigma(params)
        elif mass_observable_type == MassObservableType.MURATA:
            # return MurataMass(params)
            raise NotImplementedError("Murata mass observable not implemented.")
        elif mass_observable_type == MassObservableType.COSTANZI:
            # return CostanziMass(params)
            raise NotImplementedError("Costanzi mass observable not implemented.")
        else:
            raise ValueError(
                f"Mass observable type {mass_observable_type} not supported."
            )
