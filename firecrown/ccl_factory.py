"""This module contains the CCLFactory class and it supporting classes.

The CCLFactory class is a factory class that creates instances of the
`pyccl.Cosmology` class.
"""

from enum import StrEnum, auto
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pyccl
from pyccl.modified_gravity import MuSigmaMG
from pyccl.neutrinos import NeutrinoMassSplits
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)
from pydantic_core import core_schema

# To be moved to the import from typing when migrating to Python 3.11
from typing_extensions import NotRequired, TypedDict, assert_never

from firecrown.parameters import (
    ParamsMap,
    SamplerParameter,
    register_new_updatable_parameter,
)
from firecrown.updatable import Updatable
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
class Background(TypedDict):
    """Type representing cosmological background quantities.

    Contains arrays for scale factor, comoving distance, and Hubble parameter ratio.
    """

    a: npt.NDArray[np.float64]
    chi: npt.NDArray[np.float64]
    h_over_h0: npt.NDArray[np.float64]


# CCLCalculatorArgs is a type that represents the arguments for the
# CCLCalculator.
class CCLCalculatorArgs(TypedDict):
    """Arguments for the CCLCalculator.

    Contains background cosmology and optional linear/nonlinear power spectra.
    """

    background: Background
    pk_linear: NotRequired[PowerSpec]
    pk_nonlin: NotRequired[PowerSpec]


def _validate_neutrino_mass_splits(value):
    if isinstance(value, str):
        try:
            return NeutrinoMassSplits(value)  # Convert from string to StrEnum
        except ValueError as exc:
            raise ValueError(f"Invalid value for NeutrinoMassSplits: {value}") from exc
    return value


class PoweSpecAmplitudeParameter(YAMLSerializable, StrEnum):
    """This class defines the two-point correlation space.

    The two-point correlation space can be either real or harmonic. The real space
    corresponds measurements in terms of angular separation, while the harmonic space
    corresponds to measurements in terms of spherical harmonics decomposition.
    """

    AS = auto()
    SIGMA8 = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the PoweSpecAmplitudeParameter class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class CCLCreationMode(StrEnum):
    """This class defines the CCL instance creation mode.

    The DEFAULT mode represents the current CCL behavior. It will use CCL's calculator
    mode if `prepare` is called with a `CCLCalculatorArgs` object. Otherwise, it will
    use the default CCL mode.

    The MU_SIGMA_ISITGR mode enables the mu-sigma modified gravity model with the ISiTGR
    transfer function, it is not compatible with the Calculator mode.

    The PURE_CCL_MODE mode will create a CCL instance with the default parameters. It is
    not compatible with the Calculator mode.
    """

    DEFAULT = auto()
    MU_SIGMA_ISITGR = auto()
    PURE_CCL_MODE = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the CCLCreationMode class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


class CCLPureModeTransferFunction(StrEnum):
    """This class defines the transfer function to use in PURE_CCL_MODE.

    The options are those available in CCL for the transfer_function argument.
    See https://ccl.readthedocs.io/en/latest/api/pyccl.cosmology.html
    """

    BBKS = auto()
    BOLTZMANN_CAMB = auto()
    BOLTZMANN_CLASS = auto()
    EISENSTEIN_HU = auto()
    EISENSTEIN_HU_NOWIGGLES = auto()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for the CCLPureModeTransferFunction class."""
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            core_schema.enum_schema(cls, list(cls), sub_type="str"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


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

    HMCode_A_baryon: Annotated[float | None, Field(frozen=False)] = None
    HMCode_eta_baryon: Annotated[float | None, Field(frozen=False)] = None
    HMCode_logT_AGN: Annotated[float | None, Field(frozen=False)] = None

    halofit_version: Annotated[str | None, Field(frozen=True)] = None
    kmax: Annotated[float | None, Field(frozen=True)] = None
    lmax: Annotated[int | None, Field(frozen=True)] = None
    dark_energy_model: Annotated[str | None, Field(frozen=True)] = None

    def get_dict(self) -> dict:
        """Return the extra parameters as a dictionary."""
        return {
            key: value for key, value in self.model_dump().items() if value is not None
        }

    def update(self, params: ParamsMap) -> None:
        """Update the CAMB sampling parameters.

        :param params: The parameters to update.
        :return: None
        """
        if "HMCode_A_baryon" in params:
            self.HMCode_A_baryon = params["HMCode_A_baryon"]
        if "HMCode_eta_baryon" in params:
            self.HMCode_eta_baryon = params["HMCode_eta_baryon"]
        if "HMCode_logT_AGN" in params:
            self.HMCode_logT_AGN = params["HMCode_logT_AGN"]


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
            for a, b in zip(spline_breaks, spline_breaks[1:], strict=False)
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


# pylint: disable=too-many-instance-attributes
# Inheriting from both Updatable and BaseModel gives this class many attributes.
class CCLFactory(Updatable, BaseModel):
    """Factory class for creating instances of the `pyccl.Cosmology` class."""

    model_config = ConfigDict(extra="allow")
    require_nonlinear_pk: Annotated[bool, Field(frozen=True)] = False
    amplitude_parameter: Annotated[PoweSpecAmplitudeParameter, Field(frozen=True)] = (
        PoweSpecAmplitudeParameter.SIGMA8
    )
    mass_split: Annotated[
        NeutrinoMassSplits,
        BeforeValidator(_validate_neutrino_mass_splits),
        Field(frozen=True),
    ] = NeutrinoMassSplits.NORMAL
    # num_neutrino_masses is None except when mass_split is LIST; then it must
    # be the length of the list.
    num_neutrino_masses: Annotated[int | None, Field(frozen=True, ge=1)] = None
    creation_mode: Annotated[CCLCreationMode, Field(frozen=True)] = (
        CCLCreationMode.DEFAULT
    )
    pure_ccl_transfer_function: Annotated[
        CCLPureModeTransferFunction, Field(frozen=True)
    ] = CCLPureModeTransferFunction.BOLTZMANN_CAMB
    use_camb_hm_sampling: Annotated[bool, Field(frozen=True)] = False
    allow_multiple_camb_instances: Annotated[bool, Field(frozen=True)] = False
    camb_extra_params: Annotated[CAMBExtraParams | None, Field(frozen=True)] = None
    ccl_spline_params: Annotated[CCLSplineParams | None, Field(frozen=True)] = None

    def __init__(self, **data):
        """Initialize the CCLFactory object."""
        parameter_prefix = parameter_prefix = data.pop("parameter_prefix", None)
        BaseModel.__init__(self, **data)
        Updatable.__init__(self, parameter_prefix=parameter_prefix)

        unexpected_keys: set[str] = set(data) - set(CCLFactory.model_fields.keys())
        if unexpected_keys:
            raise ValueError(f"Invalid parameters: {unexpected_keys}")

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
        self.m_nu: float | None
        match self.mass_split:
            case NeutrinoMassSplits.LIST:
                assert self.num_neutrino_masses is not None
                for i in range(self.num_neutrino_masses):
                    if i == 0:
                        self.set_sampler_parameter(
                            SamplerParameter(name="m_nu", default_value=0.0)
                        )
                    else:
                        self.set_sampler_parameter(
                            SamplerParameter(name=f"m_nu_{i + 1}", default_value=0.0)
                        )
            case _:
                assert self.num_neutrino_masses is None
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

        if (
            self.use_camb_hm_sampling
            or (self.camb_extra_params is not None)
            or self.allow_multiple_camb_instances
        ) and not self.using_camb():
            raise ValueError(
                "CAMB extra parameters, CAMB halo model sampling, and multiple CAMB "
                "instances are only compatible with the PURE_CCL_MODE creation mode "
                "when using the BOLTZMANN_CAMB transfer function."
            )

        self._mu_sigma_model: None | MuSigmaModel = None
        match self.creation_mode:
            case CCLCreationMode.MU_SIGMA_ISITGR:
                self._mu_sigma_model = MuSigmaModel()

        if self.use_camb_hm_sampling:
            if self.camb_extra_params is None:
                raise ValueError(
                    "To sample over the halo model, "
                    "you must include camb_extra_parameters."
                )
            # The default values are taken from CAMB v1.6.0
            self.HMCode_A_baryon = register_new_updatable_parameter(default_value=3.13)
            self.HMCode_eta_baryon = register_new_updatable_parameter(
                default_value=0.603
            )
            self.HMCode_logT_AGN = register_new_updatable_parameter(default_value=7.8)

    def _update(self, params: ParamsMap) -> None:
        """Update the CAMB parameters in this CCLFactory object.

        :param params: The parameters to update.
        :returns: None
        """
        if self.camb_extra_params is not None:
            self.camb_extra_params.update(params)

    def using_camb(self) -> bool:
        """Return True if the CCLFactory is using CAMB for the matter power spectrum.

        :return: True if the CCLFactory is using CAMB for the matter power spectrum.
        """
        if self.creation_mode == CCLCreationMode.PURE_CCL_MODE:
            return (
                self.pure_ccl_transfer_function
                == CCLPureModeTransferFunction.BOLTZMANN_CAMB
            )
        return False

    @model_serializer(mode="wrap")
    def serialize_model(self, nxt: SerializerFunctionWrapHandler, _: SerializationInfo):
        """Serialize the CCLFactory object."""
        model_dump = nxt(self)
        exclude_params = [param.name for param in self._sampler_parameters] + list(
            self._internal_parameters.keys()
        )

        return {k: v for k, v in model_dump.items() if k not in exclude_params}

    @field_serializer("mass_split")
    @classmethod
    def serialize_mass_split(cls, value: NeutrinoMassSplits) -> str:
        """Serialize the mass split parameter."""
        return value.value

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
        ccl_args: dict[str, Any] = {
            "Omega_c": self.Omega_c,
            "Omega_b": self.Omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "Omega_k": self.Omega_k,
            "Neff": self.Neff,
            "w0": self.w0,
            "wa": self.wa,
            "T_CMB": self.T_CMB,
            "mass_split": self.mass_split.value,
        }
        match self.mass_split:
            case NeutrinoMassSplits.LIST:
                assert self.num_neutrino_masses is not None
                mass_list = [self.m_nu] + [
                    getattr(self, f"m_nu_{i + 1}")
                    for i in range(1, self.num_neutrino_masses)
                ]
                ccl_args["m_nu"] = mass_list
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

            if self.creation_mode != CCLCreationMode.DEFAULT:
                raise ValueError(
                    "Calculator Mode can only be used with the DEFAULT creation."
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
            case CCLCreationMode.PURE_CCL_MODE:
                ccl_args.update(
                    transfer_function=self.pure_ccl_transfer_function.lower(),
                    matter_power_spectrum="linear",
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
