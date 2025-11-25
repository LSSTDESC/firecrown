"""Filtering functionality for two-point measurements."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import assert_never

from firecrown.data_types import TwoPointMeasurement
from firecrown.data_functions._types import (
    BinSpec,
    TwoPointBinFilter,
    bin_spec_from_metadata,
)
from firecrown.metadata_types import (
    TwoPointFilterMethod,
    TwoPointHarmonic,
    TwoPointReal,
)


class TwoPointBinFilterCollection(BaseModel):
    """Class defining a collection of bin filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    require_filter_for_all: bool = Field(
        default=False,
        description="If True, all bins should match a filter.",
    )
    allow_empty: bool = Field(
        default=False,
        description=(
            "When true, objects with no elements remaining after applying "
            "the filter will be ignored rather than treated as an error."
        ),
    )
    filters: list[TwoPointBinFilter] = Field(
        description="The list of bin filters.",
    )

    _bin_filter_dict: dict[BinSpec, TwoPointBinFilter] = PrivateAttr()

    @model_validator(mode="after")
    def check_bin_filters(self) -> "TwoPointBinFilterCollection":
        """Check the bin filters."""
        bin_specs = set()
        for bin_filter in self.filters:
            bin_spec = frozenset(bin_filter.spec)
            if bin_spec in bin_specs:
                raise ValueError(
                    f"The bin name {bin_filter.spec} is repeated "
                    f"in the bin filters."
                )
            bin_specs.add(bin_spec)

        self._bin_filter_dict = {
            frozenset(bin_filter.spec): bin_filter for bin_filter in self.filters
        }
        return self

    @property
    def bin_filter_dict(self) -> dict[BinSpec, TwoPointBinFilter]:
        """Return the bin filter dictionary."""
        return self._bin_filter_dict

    def filter_match(self, tpm: TwoPointMeasurement) -> bool:
        """Check if the TwoPointMeasurement matches the filter."""
        bin_spec_key = bin_spec_from_metadata(tpm.metadata)
        return bin_spec_key in self._bin_filter_dict

    def run_bin_filter(
        self,
        bin_filter: TwoPointBinFilter,
        vals: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> npt.NDArray[np.bool_]:
        """Run the filter merge."""
        return (vals >= bin_filter.interval[0]) & (vals <= bin_filter.interval[1])

    def _apply_filter_single_window(
        self, bin_filter: TwoPointBinFilter, tpm: TwoPointMeasurement
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Apply a filter to a TwoPointMeasurement with a window function.

        The window function is a matrix where each column contains weights over ell
        values for a given observation. The filtering process depends on the method:

        - If filtering by `LABEL`, `window_ells` are filtered directly.
        - Otherwise, the filter is applied to `ells`, producing a mask
          (`match_elements`).
        For each observation, the sum of weights inside this mask is computed.
        Observations are kept if their support lies within the allowed fraction:
            - `SUPPORT`: full support (sum < 1.0)
            - `SUPPORT_95`: >=95% of support must be inside the filtered ell range
        """
        assert isinstance(tpm.metadata, TwoPointHarmonic)

        match bin_filter.method:
            case TwoPointFilterMethod.LABEL:
                assert tpm.metadata.window_ells is not None
                match_obs = self.run_bin_filter(bin_filter, tpm.metadata.window_ells)
                match_elements = np.ones_like(tpm.metadata.ells, dtype=bool)
                return match_elements, match_obs
            case TwoPointFilterMethod.SUPPORT:
                assert tpm.metadata.window is not None
                match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
                support = tpm.metadata.window[match_elements].sum(axis=0)
                match_obs = support >= 0.999
                return match_elements, match_obs
            case TwoPointFilterMethod.SUPPORT_95:
                assert tpm.metadata.window is not None
                match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
                support = tpm.metadata.window[match_elements].sum(axis=0)
                match_obs = support >= 0.95
                return match_elements, match_obs
            case _ as unreachable:
                assert_never(unreachable)

    def apply_filter_single(
        self, tpm: TwoPointMeasurement
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Apply the filter to a single TwoPointMeasurement."""
        assert self.filter_match(tpm)
        bin_spec_key = bin_spec_from_metadata(tpm.metadata)
        bin_filter = self._bin_filter_dict[bin_spec_key]
        if tpm.is_real():
            assert isinstance(tpm.metadata, TwoPointReal)
            match_elements = self.run_bin_filter(bin_filter, tpm.metadata.thetas)
            return match_elements, match_elements

        assert isinstance(tpm.metadata, TwoPointHarmonic)
        if tpm.metadata.window is None:
            match_elements = self.run_bin_filter(bin_filter, tpm.metadata.ells)
            return match_elements, match_elements

        return self._apply_filter_single_window(bin_filter, tpm)

    def __call__(
        self, tpms: Sequence[TwoPointMeasurement]
    ) -> list[TwoPointMeasurement]:
        """Filter the two-point measurements."""
        result = []

        for tpm in tpms:
            if not self.filter_match(tpm):
                if not self.require_filter_for_all:
                    result.append(tpm)
                    continue
                raise ValueError(f"The bin name {tpm.metadata} does not have a filter.")

            match_elements, match_obs = self.apply_filter_single(tpm)
            if not match_obs.any():
                if not self.allow_empty:
                    # If empty results are not allowed, we raise an error
                    raise ValueError(
                        f"The TwoPointMeasurement {tpm.metadata} does not "
                        f"have any elements matching the filter."
                    )
                # If the filter is empty, we skip this measurement
                continue

            assert isinstance(tpm.metadata, (TwoPointReal, TwoPointHarmonic))
            new_metadata: TwoPointReal | TwoPointHarmonic
            match tpm.metadata:
                case TwoPointReal():
                    new_metadata = TwoPointReal(
                        XY=tpm.metadata.XY,
                        thetas=tpm.metadata.thetas[match_elements],
                    )
                case TwoPointHarmonic():
                    # If the window function is not None, we need to filter it as well
                    # and update the metadata accordingly.
                    new_metadata = TwoPointHarmonic(
                        XY=tpm.metadata.XY,
                        window=(
                            tpm.metadata.window[:, match_obs][match_elements, :]
                            if tpm.metadata.window is not None
                            else None
                        ),
                        window_ells=(
                            tpm.metadata.window_ells[match_obs]
                            if tpm.metadata.window_ells is not None
                            else None
                        ),
                        ells=tpm.metadata.ells[match_elements],
                    )
                case _ as unreachable:
                    assert_never(unreachable)

            result.append(
                TwoPointMeasurement(
                    data=tpm.data[match_obs],
                    indices=tpm.indices[match_obs],
                    covariance_name=tpm.covariance_name,
                    metadata=new_metadata,
                )
            )

        return result
