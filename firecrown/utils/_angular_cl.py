"""Angular power spectrum utilities for Firecrown."""

import functools
from collections.abc import Callable
from enum import Enum, auto
from typing import Annotated

import numpy as np
import pyccl
from numpy import typing as npt
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_serializer
from typing_extensions import assert_never

from firecrown.utils._yaml_serialization import YAMLSerializable


class ClLimberMethod(YAMLSerializable, str, Enum):
    """This class defines Cl limber methods."""

    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.lower()

    GSL_QAG_QUAD = auto()
    GSL_SPLINE = auto()


def _validate_cl_limber_method(value: ClLimberMethod | str):
    if isinstance(value, str):
        try:
            return ClLimberMethod(value.lower())  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(f"Invalid value for ClLimberMethod: {value}") from exc
    return value


class ClIntegrationMethod(YAMLSerializable, str, Enum):
    """This class defines Cl integration methods."""

    @staticmethod
    def _generate_next_value_(name, _start, _count, _last_values):
        return name.lower()

    LIMBER = auto()
    FKEM_AUTO = auto()
    FKEM_L_LIMBER = auto()


def _validate_cl_integration_method(value: ClIntegrationMethod | str):
    if isinstance(value, str):
        try:
            return ClIntegrationMethod(value.lower())  # Convert from string to Enum
        except ValueError as exc:
            raise ValueError(f"Invalid value for ClIntegrationMethod: {value}") from exc
    return value


class ClIntegrationOptions(BaseModel):
    """Options for angular power spectrum integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Annotated[
        ClIntegrationMethod, BeforeValidator(_validate_cl_integration_method)
    ]
    limber_method: Annotated[
        ClLimberMethod, BeforeValidator(_validate_cl_limber_method)
    ]
    l_limber: int | None = None
    limber_max_error: float | None = None
    fkem_chi_min: float | None = None
    fkem_Nchi: int | None = None

    @field_serializer("method")
    @classmethod
    def serialize_method(cls, value: ClIntegrationMethod) -> str:
        """Serialize the method parameter."""
        return value.name

    @field_serializer("limber_method")
    @classmethod
    def serialize_limber_method(cls, value: ClLimberMethod) -> str:
        """Serialize the limber_method parameter."""
        return value.name

    def model_post_init(self, _, /) -> None:
        """Initialize the WeakLensingFactory object."""
        match self.method:
            case ClIntegrationMethod.LIMBER:
                incompatible_options = [
                    "limber_max_error",
                    "l_limber",
                    "fkem_chi_min",
                    "fkem_Nchi",
                ]
            case ClIntegrationMethod.FKEM_AUTO:
                incompatible_options = ["l_limber"]
            case ClIntegrationMethod.FKEM_L_LIMBER:
                incompatible_options = ["limber_max_error"]
                if self.l_limber is None or self.l_limber < 0:
                    raise ValueError("l_limber must be set for FKEM_L_LIMBER.")
            case _ as unreachable:
                assert_never(unreachable)

        for option in incompatible_options:
            if getattr(self, option) is not None:
                raise ValueError(f"{option} is incompatible with {self.method!s}.")

    def get_angular_cl_args(self):
        """Get the arguments to pass to pyccl.angular_cl."""
        match self.limber_method:
            case ClLimberMethod.GSL_QAG_QUAD:
                arg = {"limber_integration_method": "qag_quad"}
            case ClLimberMethod.GSL_SPLINE:
                arg = {"limber_integration_method": "spline"}
            case _ as unreachable:
                assert_never(unreachable)

        out: dict[str, str | int | float]
        match self.method:
            case ClIntegrationMethod.LIMBER:
                return arg | {"l_limber": -1}
            case ClIntegrationMethod.FKEM_AUTO:
                out = {
                    "l_limber": "auto",
                    "non_limber_integration_method": "FKEM",
                }
                if self.limber_max_error is not None:
                    out["limber_max_error"] = self.limber_max_error
                if self.fkem_chi_min is not None:
                    out["fkem_chi_min"] = self.fkem_chi_min
                if self.fkem_Nchi is not None:
                    out["fkem_Nchi"] = self.fkem_Nchi

                return arg | out

            case ClIntegrationMethod.FKEM_L_LIMBER:
                assert self.l_limber is not None
                out = {
                    "l_limber": self.l_limber,
                    "non_limber_integration_method": "FKEM",
                }
                if self.fkem_chi_min is not None:
                    out["fkem_chi_min"] = self.fkem_chi_min
                if self.fkem_Nchi is not None:
                    out["fkem_Nchi"] = self.fkem_Nchi

                return arg | out
            case _ as unreachable_method:
                assert_never(unreachable_method)


@functools.lru_cache(maxsize=128)
def cached_angular_cl(
    cosmo: pyccl.Cosmology,
    tracers: tuple[pyccl.Tracer, pyccl.Tracer],
    ells: npt.NDArray[np.int64],
    p_of_k_a=None | Callable[[npt.NDArray[np.int64]], npt.NDArray[np.float64]],
    p_of_k_a_lin=None | pyccl.Pk2D | str,
    int_options: ClIntegrationOptions | None = None,
):
    """Wrapper for pyccl.angular_cl, with automatic caching.

    :param cosmo: the current cosmology
    :param tracers: tracers indicating the measurements to be correlated
    :param ells: ell values at which to calculate the power spectrum
    :param p_of_k_a: function that computes the power spectrum
    :param l_limber: the maximum ell for the non-limber integration
    :param p_of_k_a_lin: function that returns the linear power spectrum
    """
    return pyccl.angular_cl(
        cosmo,
        tracers[0],
        tracers[1],
        np.array(ells),
        p_of_k_a=p_of_k_a,
        p_of_k_a_lin=p_of_k_a_lin,
        **(int_options.get_angular_cl_args() if int_options else {}),
    )
