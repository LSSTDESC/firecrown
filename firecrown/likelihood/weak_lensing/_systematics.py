"""Systematic classes for weak lensing sources."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace

import numpy as np
import pyccl
import pyccl.nl_pt

from firecrown.likelihood.weak_lensing._args import WeakLensingArgs
from firecrown.likelihood_base import (
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import register_new_updatable_parameter


class WeakLensingSystematic(SourceGalaxySystematic[WeakLensingArgs]):
    """Abstract base class for all weak lensing systematics."""

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply method to include systematics in the tracer_arg."""


class PhotoZShiftandStretch(SourceGalaxyPhotoZShiftandStretch[WeakLensingArgs]):
    """Photo-z shift systematic."""


class PhotoZShift(SourceGalaxyPhotoZShift[WeakLensingArgs]):
    """Photo-z shift systematic."""


class SelectField(SourceGalaxySelectField[WeakLensingArgs]):
    """Systematic to select 3D field."""


MULTIPLICATIVE_SHEAR_BIAS_DEFAULT_BIAS = 1.0


class MultiplicativeShearBias(WeakLensingSystematic):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar mult_bias: the multiplicative shear bias parameter.
    """

    def __init__(self, sacc_tracer: str) -> None:
        """Create a MultiplicativeShearBias object that uses the named tracer.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.mult_bias = register_new_updatable_parameter(
            default_value=MULTIPLICATIVE_SHEAR_BIAS_DEFAULT_BIAS
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Apply multiplicative shear bias to a source.

        The `scale_` of the source is multiplied by `(1 + m)`.

        :param tools: A ModelingTools object.
        :param tracer_arg: The WeakLensingArgs to which apply the shear bias.

        :return: A new WeakLensingArgs object with the shear bias applied.
        """
        return replace(
            tracer_arg,
            scale=tracer_arg.scale * (1.0 + self.mult_bias),
        )


LINEAR_ALIGNMENT_DEFAULT_IA_BIAS = 0.5
LINEAR_ALIGNMENT_DEFAULT_ALPHAZ = 0.0
LINEAR_ALIGNMENT_DEFAULT_ALPHAG = 1.0
LINEAR_ALIGNMENT_DEFAULT_Z_PIV = 0.5


class LinearAlignmentSystematic(WeakLensingSystematic):
    """Linear alignment systematic.

    This systematic adds a linear intrinsic alignment model systematic
    which varies with redshift and the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_bias: the intrinsic alignment bias parameter.
    :ivar alphaz: the redshift dependence of the intrinsic alignment bias.
    :ivar alphag: the growth function dependence of the intrinsic alignment bias.
    :ivar z_piv: the pivot redshift for the intrinsic alignment bias.
    """

    def __init__(self, sacc_tracer: None | str = None, alphag: None | float = 1.0):
        """Create a LinearAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.ia_bias = register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_IA_BIAS
        )
        self.alphaz = register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_ALPHAZ
        )
        self.alphag = register_new_updatable_parameter(
            alphag, default_value=LINEAR_ALIGNMENT_DEFAULT_ALPHAG
        )
        self.z_piv = register_new_updatable_parameter(
            default_value=LINEAR_ALIGNMENT_DEFAULT_Z_PIV
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic.

        This choice is based on the given tracer_arg, in the context of the given
        cosmology.
        """
        ccl_cosmo = tools.get_ccl_cosmology()

        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= pyccl.growth_factor(ccl_cosmo, 1.0 / (1.0 + tracer_arg.z)) ** (
            self.alphag - 1.0
        )

        ia_bias_array = pref * self.ia_bias

        return replace(
            tracer_arg,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


MASSDEP_LINEAR_ALIGNMENT_DEFAULT_IA_BIAS = 5.74
MASSDEP_LINEAR_ALIGNMENT_DEFAULT_IA_SCALING = 0.44
MASSDEP_LINEAR_ALIGNMENT_DEFAULT_RED_FRACTION = 1.0
MASSDEP_LINEAR_ALIGNMENT_DEFAULT_LOG10_AVERAGE_HALO_MASS = 13.5


class MassDependentLinearAlignmentSystematic(WeakLensingSystematic):
    """Mass-dependent linear alignment systematic.

    Adds a linear intrinsic alignment model systematic
    the amplitude of which depends on the assumed model mass scaling,
    red fraction, and average halo mass of the tracer. Blue galaxies are
    assumed to have zero intrinsic alignment amplitude.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_amplitude: the intrinsic alignment amplitude at the pivot halo mass.
    :ivar ia_mass_scaling: the power-law index of the model's mass scaling.
    :ivar red_fraction: the red galaxy fraction of the tracer sample.
    :ivar log10_average_halo_mass: the 10-base logarithm of the average halo mass of the
        tracer sample (mass should be given in units of solar mass / h).

    The following parameter is an InternalParameter that will not be provided
    by the sampler, instead the value given will be used throughout all
    calculations:

    :ivar pivot_log10_halo_mass: the log10-base of the pivot halo mass of the model
        (default=13.5, pivot mass in M_sun/h).
    """

    def __init__(self, sacc_tracer: None | str = None):
        """Create a MassDependentLinearAlignmentSystematic object.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.ia_amplitude = register_new_updatable_parameter(
            default_value=MASSDEP_LINEAR_ALIGNMENT_DEFAULT_IA_BIAS, shared=True
        )
        self.ia_mass_scaling = register_new_updatable_parameter(
            default_value=MASSDEP_LINEAR_ALIGNMENT_DEFAULT_IA_SCALING, shared=True
        )
        self.red_fraction = register_new_updatable_parameter(
            default_value=MASSDEP_LINEAR_ALIGNMENT_DEFAULT_RED_FRACTION
        )
        self.log10_average_halo_mass = register_new_updatable_parameter(
            default_value=MASSDEP_LINEAR_ALIGNMENT_DEFAULT_LOG10_AVERAGE_HALO_MASS
        )
        self.pivot_log10_halo_mass = register_new_updatable_parameter(
            value=13.5,
            default_value=MASSDEP_LINEAR_ALIGNMENT_DEFAULT_LOG10_AVERAGE_HALO_MASS,
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a mass-dependent linear alignment systematic.

        This choice is based on the given tracer_arg, in the context of the given
        cosmology.
        """
        pref = (
            self.ia_amplitude
            * self.red_fraction
            * (10**self.log10_average_halo_mass / 10**self.pivot_log10_halo_mass)
            ** self.ia_mass_scaling
        )

        ia_bias_array = np.full_like(tracer_arg.z, pref)

        return replace(
            tracer_arg,
            ia_bias=(tracer_arg.z, ia_bias_array),
        )


TATT_ALIGNMENT_DEFAULT_IA_A_1 = 1.0
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_1 = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_1 = 0.0
TATT_ALIGNMENT_DEFAULT_IA_A_2 = 0.5
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_2 = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_2 = 0.0
TATT_ALIGNMENT_DEFAULT_IA_A_D = 0.5
TATT_ALIGNMENT_DEFAULT_IA_ZPIV_D = 0.62
TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_D = 0.0


class TattAlignmentSystematic(WeakLensingSystematic):
    r"""TATT alignment systematic.

    This systematic adds a TATT (nonlinear) intrinsic alignment model systematic.

    The amplitude of each contribution to the TATT model
    (i.e. linear, density-dependent, or quadratic terms) can be expressed as
    a function in redshift, parameterized by the relationship:
    $A_i \times \frac{1 + z}{1 + z_{piv,i}}^{\alpha_i}$

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_a_1: the amplitude of the linear alignment model.
    :ivar ia_zpiv_1: the pivot redshift of the linear alignment model.
    :ivar ia_alphaz_1: the redshift dependence of the linear alignment model.
    :ivar ia_a_2: the amplitude of the quadratic alignment model.
    :ivar ia_zpiv_2: the pivot redshift of the quadratic alignment model.
    :ivar ia_alphaz_2: the redshift dependence of the quadratic alignment model.
    :ivar ia_a_d: the amplitude of the density-dependent alignment model.
    :ivar ia_zpiv_d: the pivot redshift of the density-dependent alignment model.
    :ivar ia_alphaz_d: the redshift dependence of the density-dependent alignment model.
    """

    def __init__(
        self, sacc_tracer: None | str = None, include_z_dependence: bool = False
    ):
        """Create a TattAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.ia_a_1 = register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_1
        )
        self.ia_zpiv_1 = register_new_updatable_parameter(
            value=(None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ZPIV_1),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_1,
        )
        self.ia_alphaz_1 = register_new_updatable_parameter(
            value=(
                None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_1
            ),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_1,
        )
        self.ia_a_2 = register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_2
        )
        self.ia_zpiv_2 = register_new_updatable_parameter(
            value=(None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ZPIV_2),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_2,
        )
        self.ia_alphaz_2 = register_new_updatable_parameter(
            value=(
                None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_2
            ),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_2,
        )
        self.ia_a_d = register_new_updatable_parameter(
            default_value=TATT_ALIGNMENT_DEFAULT_IA_A_D
        )
        self.ia_zpiv_d = register_new_updatable_parameter(
            value=(None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ZPIV_D),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ZPIV_D,
        )
        self.ia_alphaz_d = register_new_updatable_parameter(
            value=(
                None if include_z_dependence else TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_D
            ),
            default_value=TATT_ALIGNMENT_DEFAULT_IA_ALPHAZ_D,
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new linear alignment systematic.

        This choice is based on the given tracer_arg, in the context of the given
        cosmology.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        z = tracer_arg.z
        c_1, c_d, c_2 = pyccl.nl_pt.translate_IA_norm(
            ccl_cosmo,
            z=z,
            a1=self.ia_a_1,
            a1delta=self.ia_a_d,
            a2=self.ia_a_2,
            Om_m2_for_c2=False,
        )

        c_1 *= ((1.0 + z) / (1.0 + self.ia_zpiv_1)) ** self.ia_alphaz_1
        c_d *= ((1.0 + z) / (1.0 + self.ia_zpiv_d)) ** self.ia_alphaz_d
        c_2 *= ((1.0 + z) / (1.0 + self.ia_zpiv_2)) ** self.ia_alphaz_2

        return replace(
            tracer_arg,
            has_pt=True,
            ia_pt_c_1=(z, c_1),
            ia_pt_c_d=(z, c_d),
            ia_pt_c_2=(z, c_2),
        )


HM_ALIGNMENT_DEFAULT_IA_A_1H = 1e-4
HM_ALIGNMENT_DEFAULT_IA_A_2H = 1.0


class HMAlignmentSystematic(WeakLensingSystematic):
    """Halo model intrinsic alignment systematic.

    This systematic adds a halo model based intrinsic alignment systematic
    which, at the moment, is fixed within the redshift bin.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar ia_a_1h: the 1-halo intrinsic alignment bias parameter (satellite galaxies).
    :ivar ia_a_2h: the 2-halo intrinsic alignment bias parameter (central galaxies).
    """

    def __init__(self, _: None | str = None):
        """Create a HMAlignmentSystematic object, using the specified tracer name.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__()

        self.ia_a_1h = register_new_updatable_parameter(
            default_value=HM_ALIGNMENT_DEFAULT_IA_A_1H
        )
        self.ia_a_2h = register_new_updatable_parameter(
            default_value=HM_ALIGNMENT_DEFAULT_IA_A_2H
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: WeakLensingArgs
    ) -> WeakLensingArgs:
        """Return a new halo-model alignment systematic.

        :param tools: A ModelingTools object.
        :param tracer_arg: The WeakLensingArgs to which apply the systematic.
        :return: A new WeakLensingArgs object with the systematic applied.
        """
        return replace(
            tracer_arg, has_hm=True, ia_a_1h=self.ia_a_1h, ia_a_2h=self.ia_a_2h
        )
