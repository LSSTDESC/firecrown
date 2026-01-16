"""Model classes for CCL factory module."""

from typing import Annotated

import pyccl
from pyccl.modified_gravity import MuSigmaMG
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from firecrown.parameters import ParamsMap, register_new_updatable_parameter
from firecrown.updatable import Updatable


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

    def model_post_init(self, _, /):
        """Validate that HMCode parameters are compatible with halofit_version."""
        if self.is_mead():
            if self.HMCode_logT_AGN is not None:
                raise ValueError(
                    f"HMCode_logT_AGN is not available for "
                    f"halofit_version={self.halofit_version}. "
                    f"It is only available for halofit_version=mead2020_feedback"
                )
        elif self.is_mead2020_feedback():
            if self.HMCode_A_baryon is not None or self.HMCode_eta_baryon is not None:
                raise ValueError(
                    "HMCode_A_baryon and HMCode_eta_baryon are only available for "
                    "halofit_version in (mead, mead2015, mead2016), "
                    "not mead2020_feedback"
                )
        else:
            # Unknown halofit_version
            if any(
                param is not None
                for param in [
                    self.HMCode_A_baryon,
                    self.HMCode_eta_baryon,
                    self.HMCode_logT_AGN,
                ]
            ):
                raise ValueError(
                    f"HMCode parameters are not compatible with "
                    f"halofit_version={self.halofit_version}. "
                    f"Valid halofit versions are: mead, mead2015, "
                    f"mead2016, mead2020_feedback"
                )

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

    def is_mead2020_feedback(self) -> bool:
        """Return True if the halofit_version is mead2020_feedback."""
        if self.halofit_version is None:
            return False
        return self.halofit_version == "mead2020_feedback"

    def is_mead(self) -> bool:
        """Return True if the halofit_version is mead, mead2015, or mead2016.

        CAMB treats None as mead.

        :return: True if the halofit_version is mead, mead2015, or mead2016
        """
        if self.halofit_version is None:
            return True
        return self.halofit_version in {"mead", "mead2015", "mead2016"}


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
