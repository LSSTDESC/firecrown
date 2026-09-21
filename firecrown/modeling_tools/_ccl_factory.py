"""CCLFactory class for creating pyccl.Cosmology instances."""

from typing import Annotated, Any, cast

import pyccl
from pyccl.neutrinos import NeutrinoMassSplits
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
)

# To be moved to the import from typing when migrating to Python 3.11
from typing_extensions import assert_never

from firecrown.modeling_tools._ccl_enums import (
    CCLCreationMode,
    CCLPureModeTransferFunction,
    PoweSpecAmplitudeParameter,
)
from firecrown.modeling_tools._ccl_models import (
    CAMBExtraParams,
    CCLSplineParams,
    MuSigmaModel,
)
from firecrown.modeling_tools._ccl_types import CCLCalculatorArgs
from firecrown.updatable import (
    ParamsMap,
    SamplerParameter,
    Updatable,
    register_new_updatable_parameter,
)


def _validate_neutrino_mass_splits(value: object) -> object:
    # Preserve non-strings for Pydantic's subsequent validation.
    if isinstance(value, str):
        try:
            return NeutrinoMassSplits(value)  # Convert from string to StrEnum
        except ValueError as exc:
            raise ValueError(f"Invalid value for NeutrinoMassSplits: {value}") from exc
    return value


# Inheriting from both Updatable and BaseModel gives this class many attributes.
# pylint: disable=too-many-instance-attributes
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

    # pylint: disable=too-many-branches
    def __init__(self, **data: Any) -> None:
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
            case NeutrinoMassSplits.LIST | NeutrinoMassSplits.SUM:
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
                self.m_nu = cast(
                    float | None,
                    register_new_updatable_parameter(
                        default_value=temp_cosmology["m_nu"]
                    ),
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
            hm_defaults = self.camb_extra_params.get_hm_sampling_defaults()
            if not hm_defaults:
                raise ValueError(
                    f"There are no HMCode parameters to sample for "
                    f"halofit_version={self.camb_extra_params.halofit_version}. "
                    f"Valid halofit versions are: mead, mead2015, mead2016, "
                    f"mead2020_feedback"
                )
            # Registering the parameters under the names CAMB uses lets the sampler
            # parameter machinery resolve them, which is what makes the lower-cased
            # names supplied by CosmoSIS match. Going through setattr rather than
            # set_sampler_parameter is what applies the parameter prefix.
            for name, default_value in hm_defaults.items():
                setattr(
                    self,
                    name,
                    register_new_updatable_parameter(default_value=default_value),
                )

    def _update(self, params: ParamsMap) -> None:
        """Update the CAMB parameters in this CCLFactory object.

        :param params: The parameters to update.
        :returns: None
        """
        if not self.use_camb_hm_sampling:
            return
        assert self.camb_extra_params is not None
        camb_extra_params: CAMBExtraParams = self.camb_extra_params
        # There is something here that is confusing pylint
        # pylint: disable=no-member
        hm_names = camb_extra_params.get_hm_sampling_defaults().keys()
        camb_extra_params.set_hm_parameters(
            {name: getattr(self, name) for name in hm_names}
        )
        # pylint: enable=no-member

    def using_camb(self) -> bool:
        """Return True if the CCLFactory is using CAMB for the matter power spectrum.

        :returns: True if the CCLFactory is using CAMB for the matter power spectrum.
        """
        if self.creation_mode == CCLCreationMode.PURE_CCL_MODE:
            return (
                self.pure_ccl_transfer_function
                == CCLPureModeTransferFunction.BOLTZMANN_CAMB
            )
        return False

    @model_serializer(mode="wrap")
    def serialize_model(
        self, nxt: SerializerFunctionWrapHandler, _: SerializationInfo
    ) -> dict[str, Any]:
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

    def model_post_init(self, _: object, /) -> None:
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
            case NeutrinoMassSplits.LIST | NeutrinoMassSplits.SUM:
                assert self.num_neutrino_masses is not None
                mass_list = [self.m_nu] + [
                    getattr(self, f"m_nu_{i + 1}")
                    for i in range(1, self.num_neutrino_masses)
                ]
                ccl_args["m_nu"] = mass_list
            case _:
                ccl_args["m_nu"] = self.m_nu
        # pylint: enable=duplicate-code
        match self.amplitude_parameter:
            case PoweSpecAmplitudeParameter.AS:
                ccl_args["A_s"] = self.A_s
            case PoweSpecAmplitudeParameter.SIGMA8:
                ccl_args["sigma8"] = self.sigma8
            case _ as unreachable_ap:
                assert_never(unreachable_ap)

        assert ("A_s" in ccl_args) or ("sigma8" in ccl_args)

        if (
            self.creation_mode != CCLCreationMode.DEFAULT
        ) and calculator_args is not None:
            raise ValueError(
                "Calculator Mode can only be used with the DEFAULT creation."
            )

        assert self.creation_mode in CCLCreationMode
        match self.creation_mode:
            case CCLCreationMode.DEFAULT:
                return self._create_default(ccl_args, calculator_args)
            case CCLCreationMode.MU_SIGMA_ISITGR:
                return self._create_mu_sigma_isitgr(ccl_args)
            case CCLCreationMode.PURE_CCL_MODE:
                return self._create_pure_ccl(ccl_args)
            case _ as unreachable_cm:
                assert_never(unreachable_cm)

    def _create_default(
        self, ccl_args: dict[str, Any], calculator_args: CCLCalculatorArgs | None
    ) -> pyccl.Cosmology:
        if calculator_args is not None:
            ccl_args.update(calculator_args)
            if ("pk_nonlin" not in ccl_args) and self.require_nonlinear_pk:
                ccl_args["nonlinear_model"] = "halofit"
            else:
                ccl_args["nonlinear_model"] = None

            if self.ccl_spline_params is not None:
                with self.ccl_spline_params:
                    self._ccl_cosmo = pyccl.CosmologyCalculator(**ccl_args)
            else:
                self._ccl_cosmo = pyccl.CosmologyCalculator(**ccl_args)

            return self._ccl_cosmo

        return self._create_pure_ccl(ccl_args)

    def _create_pure_ccl(self, ccl_args: dict[str, Any]) -> pyccl.Cosmology:
        ccl_args.update(transfer_function=self.pure_ccl_transfer_function.lower())
        nonlin_str: str = "halofit"
        if (
            self.pure_ccl_transfer_function
            == CCLPureModeTransferFunction.BOLTZMANN_CAMB
        ):
            if self.camb_extra_params is not None:
                nonlin_str = "camb"
                ccl_args["matter_power_spectrum"] = nonlin_str
                ccl_args["extra_parameters"] = {
                    "camb": self.camb_extra_params.get_dict()
                }
        if self.require_nonlinear_pk:
            ccl_args["matter_power_spectrum"] = nonlin_str

        if self.ccl_spline_params is not None:
            with self.ccl_spline_params:
                self._ccl_cosmo = pyccl.Cosmology(**ccl_args)
        else:
            self._ccl_cosmo = pyccl.Cosmology(**ccl_args)
        return self._ccl_cosmo

    def _create_mu_sigma_isitgr(self, ccl_args: dict[str, Any]) -> pyccl.Cosmology:
        assert self._mu_sigma_model is not None
        ccl_args.update(
            mg_parametrization=self._mu_sigma_model.create(),
            matter_power_spectrum="linear",
            transfer_function="boltzmann_isitgr",
        )
        if self.require_nonlinear_pk:
            ccl_args["matter_power_spectrum"] = "halofit"

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
