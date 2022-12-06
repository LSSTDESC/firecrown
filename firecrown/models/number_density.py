"""Abstract Number Density function Class


This module provides the base class :python:`NumberDensity`, which is the class
from which all concrete number densities, or mass functions, classes from
firecrown must descend.

It also provides the function :python:`compute_number_density` which takes a Pyccl cosmology, a mass and a redshift
and compute the number density (mass function) for these variables. This class also provides the function
and :python:`compute_volume` which takes a Pyccl cosmology and a redshift to compute the volume density in comoving units
for a given redshift.

"""

from __future__ import annotations
from typing import List, Dict, Union, Optional
from abc import abstractmethod
import warnings
import importlib
import importlib.util
import os
import sys
import numpy as np
import numpy.typing as npt
import pyccl
import sacc

class NumberDensity:

    def __init__(self) -> None:
        self.density_func_type = None
        self.use_baryons = None
        
        
        
        
        
    def compute_number_density(self, cosmo: pyccl.Cosmology, logm, z) -> float: 
        """
        parameters
        
        cosmo : pyccl Cosmology
        logm: float
            Cluster mass given by log10(M) where M is in units of M_sun (comoving). 
        z : float
            Cluster Redshift.

        reuturn
        -------

        nm : float
            Number Density  pdf at z and logm in units of Mpc^-3 (comoving).
        """
        raise NotImplementedError("Method `compute_number_density` is not implemented!")    
    
    
    def compute_volume_density(self, cosmo: pyccl.Cosmology, z) -> float:
        """
        parameters

        cosmo : pyccl Cosmology
        z : float
            Cluster Redshift.

        reuturn
        -------

        dv : float
            Volume Density pdf at z in units of Mpc^3 (comoving).
        """
        raise NotImplementedError("Method `compute_volume_density` is not implemented!")    