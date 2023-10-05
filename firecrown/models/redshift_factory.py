from enum import Enum
import numpy as np

from firecrown.parameters import ParamsMap
from firecrown.models.kernel import Kernel
from .redshift import *

RedshiftType = Enum("RedshiftType", "SPEC DESY1_PHOTO")


class RedshiftFactory:
    @staticmethod
    def create(redshift_type: RedshiftType, params: ParamsMap = None):
        if redshift_type == RedshiftType.SPEC:
            return SpectroscopicRedshiftUncertainty(params)
        elif redshift_type == RedshiftType.DESY1_PHOTO:
            return DESY1PhotometricRedshiftUncertainty(params)
        else:
            raise ValueError(f"Redshift type {redshift_type} not supported.")
