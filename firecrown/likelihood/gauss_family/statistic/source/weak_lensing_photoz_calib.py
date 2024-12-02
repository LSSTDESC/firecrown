"""Weak lensing source and systematics

"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from .source import Source
from .source import Systematic
from .....parameters import (
    ParamsMap,
    RequiredParameters,
    parameter_get_full_name,
    DerivedParameterCollection,
)
from .....updatable import UpdatableCollection

__all__ = ["WeakLensing"]


@dataclass(frozen=True)
class WeakLensingArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray  # pylint: disable-msg=invalid-name
    dndz: np.ndarray
    ia_bias: Tuple[np.ndarray, np.ndarray]


class WeakLensingSystematic(Systematic):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    Parameters
    ----------
    mult_bias : str
       The name of the multiplicative bias parameter.
    """

    params_names = ["mult_bias"]
    m: float

    def __init__(self, sacc_tracer: str):
        """Create a MultipliciativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        # pylint: disable-next=invalid-name
        self.m = params.get_from_prefix_param(self.sacc_tracer, "mult_bias")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : a WeakLensingArgs object
            The WeakLensingArgs to which apply the shear bias.
        """
        # print("################################")
        # print('Default multipl. shear bias ')
        # print("################################")
        # print( (1.0 + self.m))
        # print('scale')
        # print(tracer_arg.scale)
        # print('z')
        # print(tracer_arg.z)
        # print('dndz')
        # print(tracer_arg.dndz)
        # print('iabias')
        # print(tracer_arg.ia_bias)
        # print('updated scale')
        # print(tracer_arg.scale * (1.0 + self.m))
        
        return WeakLensingArgs(
            scale=tracer_arg.scale * (1.0 + self.m),
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=tracer_arg.ia_bias,
        )
    
class HSCMultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    Parameters
    ----------
    mult_bias : str
       The name of the multiplicative bias parameter.
    """

    params_names = ["mult_bias"]
    m: float

    def __init__(self, sacc_tracer: str):
        """Create a MultipliciativeShearBias object that uses the named tracer.
        Parameters
        ----------
        sacc_tracer : The name of the multiplicative bias parameter.
        """
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        """Read the corresponding named tracer from the given collection of
        parameters."""
        # pylint: disable-next=invalid-name
        self.m = params.get_from_prefix_param(self.sacc_tracer, "mult_bias")
        
    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        tracer_arg : a WeakLensingArgs object
            The WeakLensingArgs to which apply the shear bias.
        """
        # print("################################")
        # print('HSC multipl. shear bias ')
        # print("################################")
        aux = sum(tracer_arg.dndz * tracer_arg.z)/sum(tracer_arg.dndz)
        # print(aux)
        
        if (aux <= 0.6):
            # print('bin 1')
            # print(aux)
            m_sel = 0.86/100
            m_R = 0.0
            factor = (1.0 + self.m)*(1 + m_sel + m_R)
            
            # print('BEFORE multiplicative shear bias')
            # print(tracer_arg.scale)
            # print('AFTER multiplicative shear bias')
            # print(f'(1+{self.m}/100) * (1 + {m_sel} + {m_R})')
            
        elif (aux > 0.6 and aux <= 0.9):
            # print('bin 2')
            # print(aux)
            m_sel = 0.99/100
            m_R = 0.0
            factor = (1.0 + self.m)*(1 + m_sel + m_R)
            
            # print('BEFORE multiplicative shear bias')
            # print(tracer_arg.scale)
            # print('AFTER multiplicative shear bias')
            # print(f'(1+{self.m}/100) * (1 + {m_sel} + {m_R})')
        
        elif (aux > 0.9 and aux <= 1.2):
            # print('bin 3')
            # print(aux)
            m_sel = 0.91/100
            m_R = 1.5/100
            factor = (1.0 + self.m)*(1 + m_sel + m_R)
            
            # print('BEFORE multiplicative shear bias')
            # print(tracer_arg.scale)
            # print('AFTER multiplicative shear bias')
            # print(f'(1+{self.m}/100) * (1 + {m_sel} + {m_R})')
            
        elif(aux > 1.2):
            # print('bin 4')
            # print(aux)
            m_sel = 0.91/100
            m_R = 3.0/100
            factor = (1.0 + self.m)*(1 + m_sel + m_R)
            
            # print('BEFORE multiplicative shear bias')
            # print(tracer_arg.scale)
            # print('AFTER multiplicative shear bias')
            # print(f'(1+{self.m}/100) * (1 + {m_sel} + {m_R})')
            
        # print('scale')
        # print(tracer_arg.scale)
        # print('z')
        # print(tracer_arg.z)
        # print('dndz')
        # print(tracer_arg.dndz)
        # print('iabias')
        # print(tracer_arg.ia_bias)
        # print('updated scale')
        # print(tracer_arg.scale * (1.0 + self.m/100) * (1 + m_sel + m_R) )

        return WeakLensingArgs(
            scale=tracer_arg.scale*factor,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class LinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.



    Methods
    -------
    apply : apply the systematic to a source
    """

    params_names = ["ia_bias", "alphaz", "alphag", "z_piv"]
    ia_bias: float
    alphaz: float
    alphag: float
    z_piv: float

    def __init__(self, sacc_tracer: Optional[str] = None):
        """Create a LinearAlignmentSystematic object, using the specified
        tracer name.

        Instance data are:

        alphaz : The redshift dependence parameter of the intrinsic alignment
        signal.

        alphag : The growth dependence parameter of the intrinsic alignment
        signal.

        z_piv : The pivot redshift parameter for the intrinsic alignment
        parameter.
        """
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.ia_bias = params.get_from_prefix_param(self.sacc_tracer, "ia_bias")
        self.alphaz = params.get_from_prefix_param(self.sacc_tracer, "alphaz")
        self.alphag = params.get_from_prefix_param(self.sacc_tracer, "alphag")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias
        
        # print('z')
        # print(tracer_arg.z)
        # print('linear IA')
        # print(ia_bias_array)

        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )

class NonLinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a NON-linear intrinsic alignment model systematic
    which varies with redshift and the growth function.

    Methods
    -------
    apply : apply the systematic to a source
    """
    
    params_names = ["a_ia", "eta_eff", "z_piv"]
    a_ia: float
    eta_eff: float
    z_piv: float

    def __init__(self, sacc_tracer: Optional[str] = None):
        """Create a NonLinearAlignmentSystematic object, using the specified
        tracer name.

        Instance data are:

        ia_bias : amplitude of the intrinsic alignment

        alphaz : redshift dependence of the intrinsic alignment
        
        alphag : null in our case

        z_piv : The pivot redshift parameter for the intrinsic alignment
        parameter.
        """
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.a_ia = params.get_from_prefix_param(self.sacc_tracer, "a_ia")
        self.eta_eff = params.get_from_prefix_param(self.sacc_tracer, "eta_eff")
        self.z_piv = params.get_from_prefix_param(self.sacc_tracer, "z_piv")

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        return DerivedParameterCollection([])

    def apply(
        self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new non-linear alignment systematic, based on the given
        tracer_arg, in the context of the given cosmology."""
        
        # C1 = 5*10**(-14)*cosmo.cosmo.params.h**(-2) # units h ** -2 M_sun ** -1 Mpc ** 3 constant taken from Hikage et al. 2019
        # C1 = 5*10**(-14) # units h ** -2 M_sun ** -1 Mpc ** 3 constant taken from Hikage et al. 2019 (watch out units, Javier)
        
        ########################################
        ###  Units of the prefactors ...     ###
        ########################################
        # C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
        # M_sun = 1.9891e30  # kg
        # Mpc_in_m = 3.0857e22  # meters
        # C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
        # # rho_crit_0 = 3 H^2 / 8 pi G
        # G = 6.67384e-11  # m^3 kg^-1 s^-2
        # H = 100  #  h km s^-1 Mpc^-1
        # H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
        # rho_crit_0 = 3 * H_SI ** 2 / (8 * np.pi * G)  #  h^2 kg m^-3
        # # Conversion to International System (no dimensions)
        # C1_SI_rho_crit_0 = C1_SI * rho_crit_0
        # print(f'C_1 (S.I.) * RHO_crit (S.I.) = {C1_SI_rho_crit_0}')
        # 
        # print('#########################################')
        # print('###        NLA DEBUGGING               ##')
        # print('#########################################')
        
        # print('Omega_cdm = ', cosmo.cosmo.params.Omega_c)
        # print('Omega_b = ', cosmo.cosmo.params.Omega_b)
        # print('h = ', cosmo.cosmo.params.h)
        # print('z_piv = ', self.z_piv)
        
        # Redshift dependance
        # pref =  ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.eta_eff
        # print('After z-dependance: ')
        # print(pref)
        # Growth and matter abundance
        # pref *= (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b)/pyccl.growth_factor(cosmo, 1.0 / (1.0 + tracer_arg.z))
        # print('After growth and Om_m: ')
        # print(pref)
        # Multiply by C1 and critical density
        # pref *= C1_SI_rho_crit_0
        # print('After C1 * rho: ')
        # print(pref)
        # Multiply by the amplitude and -1 [HSC analysis convention]
        # ia_bias_array = self.a_ia * pref
        # print('Resulting NLA: ')
        # print(ia_bias_array)
        
        # c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(cosmo, tracer_arg.z, a1=ia_bias_array, a1delta=0, a2=0, Om_m2_for_c2=False)
        # 
        # print('IA NLA after normalization')
        # print(c_1)
        
        #from pyccl rho_crit  ( h ** -2 M_sun ** -1 Mpc ** 3 ) ** -1 
        # ia_bias_array =  (-1) * self.a_ia * C1 * pyccl.physical_constants.RHO_CRITICAL * pref
#         ia_bias_array = pref * self.a_ia 
        # print('ia_bias_array = ', ia_bias_array)
        # print('norm = ', sum(ia_bias_array * tracer_arg.z)/sum(ia_bias_array))
        # print(ia_bias_array)
        
        #c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
        #    cosmo,
        #    tracer_arg.z,
        #    a1=ia_bias_array,
        #    a1delta=np.zeros_like(ia_bias_array),
        #    a2=np.zeros_like(ia_bias_array),
        #    Om_m2_for_c2=False,
        #)
        #print('ia_bias previous to translate_IA_norm: ')
        #print(ia_bias_array)
        #print('post translate_IA_norm: ')
        #print(c_1)
        
        # The rest of the factors are multiplied internally by pyccl
        # -1 * C1 * rho_cr * Om / D
        
        # References
        # CCL documentation - https://ccl.readthedocs.io/en/latest/_modules/pyccl/tracers.html?highlight=ia_bias#
        # CCL for LSST paper (page 9) - https://arxiv.org/pdf/1812.05995.pdf
        pref =  ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.eta_eff
        ia_bias_array = self.a_ia * pref
        
        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=tracer_arg.dndz,
            ia_bias=(tracer_arg.z, ia_bias_array),
            # ia_bias=(tracer_arg.z,c_1),
        )

class SourcePhotoZShift(WeakLensingSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]
    delta_z: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.delta_z = params.get_from_prefix_param(self.sacc_tracer, "delta_z")

    @final
    def _reset(self) -> None:
        pass

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply a shift to the photo-z distribution of a source."""

        # print("############################################")
        # print("Applying photo-z shift for sources")
        # print("############################################")
        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)
        # print('z')
        # print(tracer_arg.z)
        # print('dndz before shift')
        # print(tracer_arg.dndz)
        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0
        # print('dndz after shift')
        # print(dndz)

        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            ia_bias=tracer_arg.ia_bias,
        )
    
    
class SourcePhotoZStretch(WeakLensingSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["sigma_z"]
    sigma_z: float

    def __init__(self, sacc_tracer: str):
        super().__init__()

        self.sacc_tracer = sacc_tracer

    @final
    def _update(self, params: ParamsMap):
        self.sigma_z = params.get_from_prefix_param(self.sacc_tracer, "sigma_z")

    @final
    def _reset(self) -> None:
        pass

    @final
    def required_parameters(self) -> RequiredParameters:
        return RequiredParameters(
            [parameter_get_full_name(self.sacc_tracer, pn) for pn in self.params_names]
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: WeakLensingArgs):
        """Apply a shift to the photo-z distribution of a source."""

        # print("############################################")
        # print("Applying photo-z stretch for sources")
        # print("############################################")
        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)
        # print('z')
        # print(tracer_arg.z)
        # print('dndz before shift')
        # print(tracer_arg.dndz)
        
        # Compute mean redshift
        zmean = np.average(tracer_arg.z,weights=tracer_arg.dndz)
        
        dndz = dndz_interp(self.sigma_z*(tracer_arg.z-zmean)+zmean, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0
        # print('dndz after shift')
        # print(dndz)

        return WeakLensingArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            ia_bias=tracer_arg.ia_bias,
        )


class WeakLensing(Source):
    """Source class for weak lensing."""

    systematics: UpdatableCollection
    tracer_args: WeakLensingArgs

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        systematics: Optional[List[WeakLensingSystematic]] = None,
    ):
        """Initialize the WeakLensing object."""
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None
        self.current_tracer_args: Optional[WeakLensingArgs] = None
        self.systematics = UpdatableCollection(systematics)

    @final
    def _update_source(self, params: ParamsMap):
        """Implementation of Source interface `_update_source`.

        This updates all the contained systematics."""
        self.systematics.update(params)

    @final
    def _reset_source(self) -> None:
        self.systematics.reset()

    @final
    def required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = (
            derived_parameters + self.systematics.get_derived_parameters()
        )
        return derived_parameters

    def _read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)

        z = getattr(tracer, "z").copy().flatten()  # pylint: disable-msg=invalid-name
        nz = getattr(tracer, "nz").copy().flatten()  # pylint: disable-msg=invalid-name
        indices = np.argsort(z)
        z = z[indices]  # pylint: disable-msg=invalid-name
        nz = nz[indices]  # pylint: disable-msg=invalid-name

        self.tracer_args = WeakLensingArgs(scale=self.scale, z=z, dndz=nz, ia_bias=None)

    def create_tracer(self, cosmo: pyccl.Cosmology):
        """
        Render a source by applying systematics.

        """
        tracer_args = self.tracer_args

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        tracer = pyccl.WeakLensingTracer(
            cosmo, dndz=(tracer_args.z, tracer_args.dndz), ia_bias=tracer_args.ia_bias
        )
        self.current_tracer_args = tracer_args

        return tracer, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
