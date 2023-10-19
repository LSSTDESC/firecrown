from abc import ABC
from enum import Enum
from typing import List, Tuple

import numpy as np
from firecrown.updatable import Updatable
import pdb

from firecrown.models.cluster_theory import kernel as theo_kernel


class Completeness(theo_kernel.Completeness):
    def set_parameters(self):
        self.pars = Updatable()


class Purity(theo_kernel.Purity):
    def set_parameters(self):
        self.pars = Updatable()


class TrueMass(theo_kernel.TrueMass):
    def set_parameters(self):
        self.pars = Updatable()


class SpectroscopicRedshift(theo_kernel.SpectroscopicRedshift):
    def set_parameters(self):
        self.pars = Updatable()


class DESY1PhotometricRedshift(theo_kernel.DESY1PhotometricRedshift):
    def set_parameters(self):
        self.pars = Updatable()
