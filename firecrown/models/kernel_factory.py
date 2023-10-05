from enum import Enum
import numpy as np
from firecrown.parameters import ParamsMap
from firecrown.updatable import Updatable
from .kernel import *


KernelType = Enum("KernelType", "COMPLETENESS PURITY MISCENTERING")


class KernelFactory:
    @staticmethod
    def create(KernelType: KernelType, params: ParamsMap = None):
        if KernelType == KernelType.COMPLETENESS:
            return Completeness(params)
        elif KernelType == KernelType.PURITY:
            return Purity(params)
        elif KernelType == KernelType.MISCENTERING:
            return Miscentering(params)
        else:
            raise ValueError(f"Kernel type {KernelType} not supported.")
