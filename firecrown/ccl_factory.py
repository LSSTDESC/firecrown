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
from pyccl.modified_gravity import MuSigmaMG

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


def _validate_amplitude_parameter(value):
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


class CCLCreationMode(YAMLSerializable, str, Enum):
    """This class defines the CCL instance creation mode.

    The DEFAULT mode represents the current CCL behavior. It will use CCL's calculator
    mode if `prepare` is called with a `CCLCalculatorArgs` object. Otherwise, it will
    use the default CCL mode.

    The MU_SIGMA_ISITGR mode enables the mu-sigma modified gravity model with the ISiTGR
    transfer function, it is not compatible with the Calculator mode.

    The PURE_CCL_MODE mode will create a CCL instance with the default parameters. It is
    not compatible with the Calculator mode.
    """

    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.lower()

    DEFAULT = auto()
    MU_SIGMA_ISITGR = auto()
    PURE_CCL_MODE = auto()


def _validate_ccl_creation_mode(value):
    assert isinstance(value, str)
    try:
        return CCLCreationMode(value.lower())  # Convert from string to Enum
    except ValueError as exc:
        raise ValueError(f"Invalid value for CCLCreationMode: {value}") from exc


class MuSigmaModel(Updatable):
    """Model for the mu-sigma modified gravity model."""

    def __init__(self):
        """Initialize the MuSigmaModel object."""
        super().__init__(parameter_prefix="mg_musigma")

        self.mu = register_new_updatable_parameter(default_value=1.0)
        self.sigma = register_new_updatable_parameter(default_value=1.0)
        self.c1 = register_new_updatable_parameter(default_value=1.0)
        self.c2 = register_new_updatable_parameter(default_value=1.0)
        # We cannot clash with the lambda keyword
        self.lambda0 = register_new_updatable_parameter(default_value=1.0)

    def create(self) -> MuSigmaMG:
        """Create a `pyccl.modified_gravity.MuSigmaMG` object."""
        if not self.is_updated():
            raise ValueError("Parameters have not been updated yet.")

        return MuSigmaMG(self.mu, self.sigma, self.c1, self.c2, self.lambda0)


class CAMBExtraParams(BaseModel):
    """Extra parameters for CAMB."""

    model_config = ConfigDict(extra="forbid")

    halofit_version: Annotated[str | None, Field(frozen=True)] = None
    HMCode_A_baryon: Annotated[float | None, Field(frozen=True)] = None
    HMCode_eta_baryon: Annotated[float | None, Field(frozen=True)] = None
    HMCode_logT_AGN: Annotated[float | None, Field(frozen=True)] = None
    kmax: Annotated[float | None, Field(frozen=True)] = None
    lmax: Annotated[int | None, Field(frozen=True)] = None
    dark_energy_model: Annotated[str | None, Field(frozen=True)] = None

    def get_dict(self) -> dict:
        """Return the extra parameters as a dictionary."""
        return {
            key: value for key, value in self.model_dump().items() if value is not None
        }


class CCLFactory(Updatable, BaseModel):
    """Factory class for creating instances of the `pyccl.Cosmology` class."""

    model_config = ConfigDict(extra="allow")
    require_nonlinear_pk: Annotated[bool, Field(frozen=True)] = False
    amplitude_parameter: Annotated[
        PoweSpecAmplitudeParameter,
        BeforeValidator(_validate_amplitude_parameter),
        Field(frozen=True),
    ] = PoweSpecAmplitudeParameter.SIGMA8
    mass_split: Annotated[
        NeutrinoMassSplits,
        BeforeValidator(_validate_neutrino_mass_splits),
        Field(frozen=True),
    ] = NeutrinoMassSplits.NORMAL
    creation_mode: Annotated[
        CCLCreationMode,
        BeforeValidator(_validate_ccl_creation_mode),
        Field(frozen=True),
    ] = CCLCreationMode.DEFAULT
    camb_extra_params: Annotated[CAMBExtraParams | None, Field(frozen=True)] = None

    def __init__(self, **data):
        """Initialize the CCLFactory object."""
        parameter_prefix = parameter_prefix = data.pop("parameter_prefix", None)
        BaseModel.__init__(self, **data)
        Updatable.__init__(self, parameter_prefix=parameter_prefix)

        if set(data) - set(self.model_fields.keys()):
            raise ValueError(
                f"Invalid parameters: {set(data) - set(self.model_fields.keys())}"
            )

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

        self._mu_sigma_model: None | MuSigmaModel = None
        match self.creation_mode:
            case CCLCreationMode.MU_SIGMA_ISITGR:
                self._mu_sigma_model = MuSigmaModel()

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

    @field_serializer("creation_mode")
    @classmethod
    def serialize_creation_mode(cls, value: CCLCreationMode) -> str:
        """Serialize the creation mode parameter."""
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

            if (self.creation_mode != CCLCreationMode.DEFAULT) or (
                self.camb_extra_params is not None
            ):
                raise ValueError(
                    "Calculator Mode can only be used with the DEFAULT creation "
                    "mode and no CAMB extra parameters."
                )

            self._ccl_cosmo = pyccl.CosmologyCalculator(**ccl_args)
            return self._ccl_cosmo

        if self.require_nonlinear_pk:
            ccl_args["matter_power_spectrum"] = "halofit"

        if self.camb_extra_params is not None:
            ccl_args["extra_parameters"] = {"camb": self.camb_extra_params.get_dict()}
            ccl_args["matter_power_spectrum"] = "camb"

        assert self.creation_mode in CCLCreationMode
        match self.creation_mode:
            case CCLCreationMode.MU_SIGMA_ISITGR:
                assert self._mu_sigma_model is not None
                ccl_args.update(
                    mg_parametrization=self._mu_sigma_model.create(),
                    matter_power_spectrum="linear",
                    transfer_function="boltzmann_isitgr",
                )
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
