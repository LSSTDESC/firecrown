"""This module contains the CCLFactory class.

The CCLFactory class is a factory class that creates instances of the
`pyccl.Cosmology` class.
"""

from typing import Annotated
from enum import Enum, auto

# To be moved to the import from typing when migrating to Python 3.11
from typing_extensions import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    BeforeValidator,
    SerializerFunctionWrapHandler,
    SerializationInfo,
    Field,
    field_serializer,
    model_serializer,
)

import pyccl
from pyccl.neutrinos import NeutrinoMassSplits

from firecrown.updatable import Updatable
from firecrown.parameters import register_new_updatable_parameter
from firecrown.utils import YAMLSerializable

PowerSpec = TypedDict(
    "PowerSpec",
    {
        "a": npt.NDArray[np.float64],
        "k": npt.NDArray[np.float64],
        "delta_matter:delta_matter": npt.NDArray[np.float64],
    },
)

Background = TypedDict(
    "Background",
    {
        "a": npt.NDArray[np.float64],
        "chi": npt.NDArray[np.float64],
        "h_over_h0": npt.NDArray[np.float64],
    },
)

CCLCalculatorArgs = TypedDict(
    "CCLCalculatorArgs",
    {
        "background": Background,
        "pk_linear": NotRequired[PowerSpec],
        "pk_nonlin": NotRequired[PowerSpec],
    },
)


class PoweSpecAmplitudeParameter(YAMLSerializable, str, Enum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.lower()

    AS = auto()
    SIGMA8 = auto()


def _validade_amplitude_parameter(value):
    if isinstance(value, str):
        try:
            return PoweSpecAmplitudeParameter(
                value.lower()
            )  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for PoweSpecAmplitudeParameter: {value}"
            ) from exc
    return value


def _validate_neutrino_mass_splits(value):
    if isinstance(value, str):
        try:
            return NeutrinoMassSplits(value.lower())  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(f"Invalid value for NeutrinoMassSplits: {value}") from exc
    return value


class CCLFactory(Updatable, BaseModel):
    """Factory class for creating instances of the `pyccl.Cosmology` class."""

    model_config = ConfigDict(extra="allow")
    require_nonlinear_pk: Annotated[bool, Field(frozen=True)] = False
    amplitude_parameter: Annotated[
        PoweSpecAmplitudeParameter,
        BeforeValidator(_validade_amplitude_parameter),
        Field(frozen=True),
    ] = PoweSpecAmplitudeParameter.SIGMA8
    mass_split: Annotated[
        NeutrinoMassSplits,
        BeforeValidator(_validate_neutrino_mass_splits),
        Field(frozen=True),
    ] = NeutrinoMassSplits.NORMAL

    def __init__(self, **data):
        """Initialize the CCLFactory object."""
        parameter_prefix = parameter_prefix = data.pop("parameter_prefix", None)
        BaseModel.__init__(self, **data)
        Updatable.__init__(self, parameter_prefix=parameter_prefix)

        self._ccl_cosmo: None | pyccl.Cosmology = None

        ccl_cosmo = pyccl.CosmologyVanillaLCDM()

        self.Omega_c = register_new_updatable_parameter(
            default_value=ccl_cosmo["Omega_c"]
        )
        self.Omega_b = register_new_updatable_parameter(
            default_value=ccl_cosmo["Omega_b"]
        )
        self.h = register_new_updatable_parameter(default_value=ccl_cosmo["h"])
        self.n_s = register_new_updatable_parameter(default_value=ccl_cosmo["n_s"])
        self.Omega_k = register_new_updatable_parameter(
            default_value=ccl_cosmo["Omega_k"]
        )
        self.Neff = register_new_updatable_parameter(default_value=ccl_cosmo["Neff"])
        self.m_nu = register_new_updatable_parameter(default_value=ccl_cosmo["m_nu"])
        self.w0 = register_new_updatable_parameter(default_value=ccl_cosmo["w0"])
        self.wa = register_new_updatable_parameter(default_value=ccl_cosmo["wa"])
        self.T_CMB = register_new_updatable_parameter(default_value=ccl_cosmo["T_CMB"])

        match self.amplitude_parameter:
            case PoweSpecAmplitudeParameter.AS:
                # VanillaLCDM has does not have A_s, so we need to add it
                self.A_s = register_new_updatable_parameter(default_value=2.1e-9)
            case PoweSpecAmplitudeParameter.SIGMA8:
                assert ccl_cosmo["sigma8"] is not None
                self.sigma8 = register_new_updatable_parameter(
                    default_value=ccl_cosmo["sigma8"]
                )

    @model_serializer(mode="wrap")
    def serialize_model(self, nxt: SerializerFunctionWrapHandler, _: SerializationInfo):
        """Serialize the CCLFactory object."""
        model_dump = nxt(self)
        exclude_params = [param.name for param in self._sampler_parameters] + list(
            self._internal_parameters.keys()
        )

        return {k: v for k, v in model_dump.items() if k not in exclude_params}

    @field_serializer("amplitude_parameter")
    @classmethod
    def serialize_amplitude_parameter(cls, value: PoweSpecAmplitudeParameter) -> str:
        """Serialize the amplitude parameter."""
        return value.name

    @field_serializer("mass_split")
    @classmethod
    def serialize_mass_split(cls, value: NeutrinoMassSplits) -> str:
        """Serialize the mass split parameter."""
        return value.name

    def model_post_init(self, __context) -> None:
        """Initialize the WeakLensingFactory object."""

    def create(
        self, calculator_args: CCLCalculatorArgs | None = None
    ) -> pyccl.Cosmology:
        """Create a `pyccl.Cosmology` object."""
        if not self.is_updated():
            raise ValueError("Parameters have not been updated yet.")

        if self._ccl_cosmo is not None:
            raise ValueError("CCLFactory object has already been created.")

        # pylint: disable=duplicate-code
        ccl_args = {
            "Omega_c": self.Omega_c,
            "Omega_b": self.Omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "Omega_k": self.Omega_k,
            "Neff": self.Neff,
            "m_nu": self.m_nu,
            "w0": self.w0,
            "wa": self.wa,
            "T_CMB": self.T_CMB,
            "mass_split": self.mass_split.value,
        }
        # pylint: enable=duplicate-code
        match self.amplitude_parameter:
            case PoweSpecAmplitudeParameter.AS:
                ccl_args["A_s"] = self.A_s
            case PoweSpecAmplitudeParameter.SIGMA8:
                ccl_args["sigma8"] = self.sigma8

        assert ("A_s" in ccl_args) or ("sigma8" in ccl_args)

        if calculator_args is not None:
            ccl_args.update(calculator_args)
            if ("pk_nonlin" not in ccl_args) and self.require_nonlinear_pk:
                ccl_args["nonlinear_model"] = "halofit"
            else:
                ccl_args["nonlinear_model"] = None

            return pyccl.CosmologyCalculator(**ccl_args)

        if self.require_nonlinear_pk:
            ccl_args["matter_power_spectrum"] = "halofit"

        self._ccl_cosmo = pyccl.Cosmology(**ccl_args)
        return self._ccl_cosmo

    def _reset(self) -> None:
        """Reset the CCLFactory object."""
        self._ccl_cosmo = None

    def get(self) -> pyccl.Cosmology:
        """Return the `pyccl.Cosmology` object."""
        if self._ccl_cosmo is None:
            raise ValueError("CCLFactory object has not been created yet.")
        return self._ccl_cosmo
