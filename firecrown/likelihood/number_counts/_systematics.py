"""Systematic classes for number counts sources."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace

import numpy as np
import pyccl

from firecrown import parameters
from firecrown.likelihood.number_counts._args import NumberCountsArgs
from firecrown.likelihood._base import (
    SourceGalaxyPhotoZShift,
    SourceGalaxyPhotoZShiftandStretch,
    SourceGalaxySpecZStretch,
    SourceGalaxySelectField,
    SourceGalaxySystematic,
)
from firecrown.modeling_tools import ModelingTools


class NumberCountsSystematic(SourceGalaxySystematic[NumberCountsArgs]):
    """Abstract base class for systematics for Number Counts sources.

    Derived classes must implement :meth:`apply` with the correct signature.
    """

    @abstractmethod
    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply method to include systematics in the tracer_arg.

        This does not modify the supplied tracer_arg; it returns a new
        one that has been updated.

        :param tools: the Modeling tools used to update the tracer_arg
        :param tracer_arg: the original NumberCountsArgs to which to apply the
             systematic
        :return: the updated NumberCountsArgs
        """


class PhotoZShift(SourceGalaxyPhotoZShift[NumberCountsArgs]):
    """Photo-z shift systematic."""


class PhotoZShiftandStretch(SourceGalaxyPhotoZShiftandStretch[NumberCountsArgs]):
    """Photo-z shift systematic."""


class SpecZStretch(SourceGalaxySpecZStretch[NumberCountsArgs]):
    """Spec-z stretch systematic."""


class SelectField(SourceGalaxySelectField[NumberCountsArgs]):
    """Systematic to select 3D field."""


LINEAR_BIAS_DEFAULT_ALPHAZ = 0.0
LINEAR_BIAS_DEFAULT_ALPHAG = 0.0
LINEAR_BIAS_DEFAULT_Z_PIV = 0.5


class LinearBiasSystematic(NumberCountsSystematic):
    """Linear bias systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar alphaz: the redshift exponent of the bias.
    :ivar alphag: the growth function exponent of the bias.
    :ivar z_piv: the pivot redshift of the bias.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the LinearBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.alphaz = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_ALPHAZ
        )
        self.alphag = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_ALPHAG
        )
        self.z_piv = parameters.register_new_updatable_parameter(
            default_value=LINEAR_BIAS_DEFAULT_Z_PIV
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a linear bias systematic.

        :param tools: the ModelingTools used to update the tracer_arg
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: the updated NumberCountsArgs object
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        pref = ((1.0 + tracer_arg.z) / (1.0 + self.z_piv)) ** self.alphaz
        pref *= (
            pyccl.growth_factor(ccl_cosmo, 1.0 / (1.0 + tracer_arg.z)) ** self.alphag
        )

        if tracer_arg.bias is None:
            bias = np.ones_like(tracer_arg.z)
        else:
            bias = tracer_arg.bias
        bias = bias * pref

        return replace(
            tracer_arg,
            bias=bias,
        )


PT_NON_LINEAR_BIAS_DEFAULT_B_2 = 1.0
PT_NON_LINEAR_BIAS_DEFAULT_B_S = 1.0


class PTNonLinearBiasSystematic(NumberCountsSystematic):
    """Non-linear bias systematic.

    This systematic adds a linear bias model which varies with redshift and
    the growth function.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar b_2: the quadratic bias.
    :ivar b_s: the stochastic bias.
    """

    def __init__(self, sacc_tracer: None | str = None):
        """Initialize the PTNonLinearBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.

        """
        super().__init__(parameter_prefix=sacc_tracer)
        self.b_2 = parameters.register_new_updatable_parameter(
            default_value=PT_NON_LINEAR_BIAS_DEFAULT_B_2
        )
        self.b_s = parameters.register_new_updatable_parameter(
            default_value=PT_NON_LINEAR_BIAS_DEFAULT_B_S
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a non-linear bias systematic.

        :param tools: currently unused, but required by interface
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: the updated NumberCountsArgs object
        """
        z = tracer_arg.z
        b_2_z = self.b_2 * np.ones_like(z)
        b_s_z = self.b_s * np.ones_like(z)
        # b_1 uses the "bias" field
        return replace(
            tracer_arg,
            has_pt=True,
            b_2=(z, b_2_z),
            b_s=(z, b_s_z),
        )


class MagnificationBiasSystematic(NumberCountsSystematic):
    """Magnification bias systematic.

    This systematic adds a magnification bias model for galaxy number contrast
    following Joachimi & Bridle (2010), arXiv:0911.2454.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar r_lim: the limiting magnitude.
    :ivar sig_c: the intrinsic dispersion of the source redshift distribution.
    :ivar eta: the slope of the luminosity function.
    :ivar z_c: the characteristic redshift of the source distribution.
    :ivar z_m: the slope of the source redshift distribution.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the MagnificationBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.r_lim = parameters.register_new_updatable_parameter(default_value=24.0)
        self.sig_c = parameters.register_new_updatable_parameter(default_value=9.83)
        self.eta = parameters.register_new_updatable_parameter(default_value=19.0)
        self.z_c = parameters.register_new_updatable_parameter(default_value=0.39)
        self.z_m = parameters.register_new_updatable_parameter(default_value=0.055)

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a magnification bias systematic.

        :param tools: currently unused, but required by the interface
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: an updated NumberCountsArgs object
        """
        z_bar = self.z_c + self.z_m * (self.r_lim - 24.0)
        # The slope of log(n_tot(z,r_lim)) with respect to r_lim
        # where n_tot(z,r_lim) is the luminosity function after using fit (C.1)
        s = (
            self.eta / self.r_lim
            - 3.0 * self.z_m / z_bar
            + 1.5 * self.z_m * np.power(tracer_arg.z / z_bar, 1.5) / z_bar
        )

        if tracer_arg.mag_bias is None:
            mag_bias = np.ones_like(tracer_arg.z)
        else:
            mag_bias = tracer_arg.mag_bias[1]
        mag_bias = mag_bias * s / np.log(10)

        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, mag_bias),
        )


CONSTANT_MAGNIFICATION_BIAS_DEFAULT_MAG_BIAS = 1.0


class ConstantMagnificationBiasSystematic(NumberCountsSystematic):
    """Simple constant magnification bias systematic.

    This systematic adds a constant magnification bias model for galaxy number
    contrast.

    The following parameters are special Updatable parameters, which means that
    they can be updated by the sampler, sacc_tracer is going to be used as a
    prefix for the parameters:

    :ivar mag_bias: the magnification bias.
    """

    def __init__(self, sacc_tracer: str):
        """Initialize the ConstantMagnificationBiasSystematic.

        :param sacc_tracer: the name of the tracer in the SACC file. This is used
            as a prefix for its parameters.
        """
        super().__init__(parameter_prefix=sacc_tracer)

        self.mag_bias = parameters.register_new_updatable_parameter(
            default_value=CONSTANT_MAGNIFICATION_BIAS_DEFAULT_MAG_BIAS
        )

    def apply(
        self, tools: ModelingTools, tracer_arg: NumberCountsArgs
    ) -> NumberCountsArgs:
        """Apply a constant magnification bias systematic.

        :param tools: currently unused, but required by interface
        :param tracer_arg: a NumberCountsArgs object with values to be updated

        :return: an updated NumberCountsArgs object
        """
        return replace(
            tracer_arg,
            mag_bias=(tracer_arg.z, np.ones_like(tracer_arg.z) * self.mag_bias),
        )
