"""This module contains the CCLFactory class and it supporting classes.

The CCLFactory class is a factory class that creates instances of the
`pyccl.Cosmology` class.
"""

from typing import Annotated
from enum import Enum, auto

# To be moved to the import from typing when migrating to Python 3.11
from typing_extensions import NotRequired, TypedDict, assert_never

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
    model_validator,
    PrivateAttr,
)

import pyccl
from pyccl.neutrinos import NeutrinoMassSplits
from pyccl.modified_gravity import MuSigmaMG

from firecrown.updatable import Updatable
from firecrown.parameters import register_new_updatable_parameter
from firecrown.utils import YAMLSerializable

# PowerSpec is a type that represents a power spectrum.
PowerSpec = TypedDict(
    "PowerSpec",
    {
        "a": npt.NDArray[np.float64],
        "k": npt.NDArray[np.float64],
        "delta_matter:delta_matter": npt.NDArray[np.float64],
    },
)

# Background is a type that represents the cosmological background quantities.
Background = TypedDict(
    "Background",
    {
        "a": npt.NDArray[np.float64],
        "chi": npt.NDArray[np.float64],
        "h_over_h0": npt.NDArray[np.float64],
    },
)

# CCLCalculatorArgs is a type that represents the arguments for the
# CCLCalculator.
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


class CCLSplineParams(BaseModel):
    """Params to control CCL spline interpolation."""

    model_config = ConfigDict(extra="forbid")

    # Scale factor splines
    a_spline_na: Annotated[int | None, Field(frozen=True)] = None
    a_spline_min: Annotated[float | None, Field(frozen=True)] = None
    a_spline_minlog_pk: Annotated[float | None, Field(frozen=True)] = None
    a_spline_min_pk: Annotated[float | None, Field(frozen=True)] = None
    a_spline_minlog_sm: Annotated[float | None, Field(frozen=True)] = None
    a_spline_min_sm: Annotated[float | None, Field(frozen=True)] = None
    # a_spline_max is not defined because the CCL parameter A_SPLINE_MAX is
    # required to be 1.0.
    a_spline_minlog: Annotated[float | None, Field(frozen=True)] = None
    a_spline_nlog: Annotated[int | None, Field(frozen=True)] = None

    # mass splines
    logm_spline_delta: Annotated[float | None, Field(frozen=True)] = None
    logm_spline_nm: Annotated[int | None, Field(frozen=True)] = None
    logm_spline_min: Annotated[float | None, Field(frozen=True)] = None
    logm_spline_max: Annotated[float | None, Field(frozen=True)] = None

    # PS a and k spline
    a_spline_na_sm: Annotated[int | None, Field(frozen=True)] = None
    a_spline_nlog_sm: Annotated[int | None, Field(frozen=True)] = None
    a_spline_na_pk: Annotated[int | None, Field(frozen=True)] = None
    a_spline_nlog_pk: Annotated[int | None, Field(frozen=True)] = None

    # k-splines and integrals
    k_max_spline: Annotated[float | None, Field(frozen=True)] = None
    k_max: Annotated[float | None, Field(frozen=True)] = None
    k_min: Annotated[float | None, Field(frozen=True)] = None
    dlogk_integration: Annotated[float | None, Field(frozen=True)] = None
    dchi_integration: Annotated[float | None, Field(frozen=True)] = None
    n_k: Annotated[int | None, Field(frozen=True)] = None
    n_k_3dcor: Annotated[int | None, Field(frozen=True)] = None

    # Correlation function parameters
    ell_min_corr: Annotated[float | None, Field(frozen=True)] = None
    ell_max_corr: Annotated[float | None, Field(frozen=True)] = None
    n_ell_corr: Annotated[int | None, Field(frozen=True)] = None

    # Attributes that are used for the context manager functionality.
    # These are *not* part of the model.
    _spline_params: dict[str, float | int] = PrivateAttr()

    @model_validator(mode="after")
    def check_spline_params(self) -> "CCLSplineParams":
        """Check that the spline parameters are valid."""
        # Ensure the spline boundaries and breakpoint are valid.
        spline_breaks = [self.a_spline_minlog, self.a_spline_min, 1.0]
        spline_breaks = list(filter(lambda x: x is not None, spline_breaks))
        assert all(
            a is not None and b is not None and a < b
            for a, b in zip(spline_breaks, spline_breaks[1:])
        )

        # Ensure the mass spline boundaries are valid
        if self.logm_spline_min is not None and self.logm_spline_max is not None:
            assert self.logm_spline_min < self.logm_spline_max

        # Ensure the k-spline boundaries are valid
        if self.k_min is not None and self.k_max is not None:
            assert self.k_min < self.k_max

        # Ensure the ell-spline boundaries are valid
        if self.ell_min_corr is not None and self.ell_max_corr is not None:
            assert self.ell_min_corr < self.ell_max_corr

        return self

    def __enter__(self) -> "CCLSplineParams":
        """Enter the context manager.

        This method saves the current CCL global spline parameters,
        updates them with the values from this `CCLSplineParams` instance,
        and returns the instance itself. This allows for temporary modification
        of CCL spline parameters using a `with` statement.

        :return:  The current instance with updated spline parameters.
        """
        self._spline_params = pyccl.CCLParameters.get_params_dict(pyccl.spline_params)
        for key, value in self.model_dump().items():
            if value is not None:
                pyccl.spline_params[key.upper()] = value
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager.

        This method resets the CCL global spline parameters to their original
        values, as saved when entering the context manager. It ensures that
        any temporary modifications made to the CCL spline parameters within
        a `with` statement are reverted upon exit.

        :param exc_type: The exception type, if an exception occurred.
        :param exc_value: The exception value, if an exception occurred.
        :param traceback: The traceback object, if an exception occurred.
        """
        for key, value in self._spline_params.items():
            pyccl.spline_params[key] = value
        if exc_type is not None:
            raise exc_type(exc_value).with_traceback(traceback)


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
    ccl_spline_params: Annotated[CCLSplineParams | None, Field(frozen=True)] = None

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

        temp_cosmology = pyccl.CosmologyVanillaLCDM()

        self.Omega_c = register_new_updatable_parameter(
            default_value=temp_cosmology["Omega_c"]
        )
        self.Omega_b = register_new_updatable_parameter(
            default_value=temp_cosmology["Omega_b"]
        )
        self.h = register_new_updatable_parameter(default_value=temp_cosmology["h"])
        self.n_s = register_new_updatable_parameter(default_value=temp_cosmology["n_s"])
        self.Omega_k = register_new_updatable_parameter(
            default_value=temp_cosmology["Omega_k"]
        )
        self.Neff = register_new_updatable_parameter(
            default_value=temp_cosmology["Neff"]
        )
        self.m_nu = register_new_updatable_parameter(
            default_value=temp_cosmology["m_nu"]
        )
        self.w0 = register_new_updatable_parameter(default_value=temp_cosmology["w0"])
        self.wa = register_new_updatable_parameter(default_value=temp_cosmology["wa"])
        self.T_CMB = register_new_updatable_parameter(
            default_value=temp_cosmology["T_CMB"]
        )

        match self.amplitude_parameter:
            case PoweSpecAmplitudeParameter.AS:
                # VanillaLCDM has does not have A_s, so we need to add it
                self.A_s = register_new_updatable_parameter(default_value=2.1e-9)
            case PoweSpecAmplitudeParameter.SIGMA8:
                assert temp_cosmology["sigma8"] is not None
                self.sigma8 = register_new_updatable_parameter(
                    default_value=temp_cosmology["sigma8"]
                )
            case _ as unreachable:
                assert_never(unreachable)

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

    def model_post_init(self, _, /) -> None:
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
            case _ as unreachable:
                assert_never(unreachable)

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

            if self.ccl_spline_params is not None:
                with self.ccl_spline_params:
                    self._ccl_cosmo = pyccl.CosmologyCalculator(**ccl_args)
            else:
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
        if self.ccl_spline_params is not None:
            with self.ccl_spline_params:
                self._ccl_cosmo = pyccl.Cosmology(**ccl_args)
        else:
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
