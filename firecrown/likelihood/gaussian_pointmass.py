"""Provides GaussFamily concrete types."""

from __future__ import annotations
import warnings
from collections.abc import Sequence
from typing import final

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
import pyccl

from firecrown.likelihood.gaussfamily import GaussFamily, State, enforce_states
from firecrown.likelihood.statistic import Statistic
from firecrown.modeling_tools import ModelingTools


class ConstGaussianPM(GaussFamily):
    """A Gaussian log-like with constant covariance, marginalizing over a point mass."""

    def __init__(self, statistics: Sequence[Statistic]) -> None:
        """Initialize the ConstGaussianPM object.

        :param statistics: A list of statistics for chi-squared calculations
        """
        super().__init__(statistics)
        # Initialize point mass marginalization attributes
        self._pm_maps_ready: bool = False
        self._pm_theta: np.ndarray | None = None
        self._pm_row_lens_idx: np.ndarray | None = None
        self._pm_row_src_idx: np.ndarray | None = None
        self._pm_lens_tracers: list | None = None
        self._pm_src_tracers: list | None = None
        self._pm_z_l: np.ndarray | None = None
        self._pm_z_s: np.ndarray | None = None
        self._pm_nzL_norm: np.ndarray | None = None
        self._pm_nzS_norm: np.ndarray | None = None
        self._pm_inv_cov_original: np.ndarray | None = None
        self.inv_cov_correction: np.ndarray | None = None

    def compute_loglike(self, tools: ModelingTools) -> float:
        """Compute the log-likelihood.

        :params tools: The modeling tools used to compute the likelihood.
        :return: The log-likelihood.
        """
        return -0.5 * self.compute_chisq(tools)

    def make_realization_vector(self) -> np.ndarray:
        """Create a new (randomized) realization of the model.

        :return: A new realization of the model
        """
        theory_vector = self.get_theory_vector()
        assert self.cholesky is not None
        new_data_vector = theory_vector + np.dot(
            self.cholesky, np.random.randn(len(theory_vector))
        )

        return new_data_vector

    def _generate_maps(self) -> None:
        """Build maps and masks for the data vectors.

        These are not needed for a constant cosmology, but will become useful
        if we want to update the point mass correction when the cosmology changes.
        """
        # The function should only be run one time.
        if self._pm_maps_ready:
            warnings.warn(
                "The point mass pre-computation step was already performed, "
                "but it is being called again. ",
            )
            return

        data_types = np.concatenate(
            [
                np.repeat(
                    stat.statistic.sacc_data_type,
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        lens_keys = np.concatenate(
            [
                np.repeat(
                    stat.statistic.source0.sacc_tracer,
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        src_keys = np.concatenate(
            [
                np.repeat(
                    stat.statistic.source1.sacc_tracer,
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        theta = np.radians(
            np.concatenate([stat.statistic.thetas for stat in self.statistics]) / 60
        )
        assert (
            "galaxy_shearDensity_xi_t" in data_types
        ), "Data must contain at least one 'galaxy_shearDensity_xi_t' datavector."

        xi_mask = np.array([dt == "galaxy_shearDensity_xi_t" for dt in data_types])
        xi_rows = np.where(xi_mask)[0]
        xi_pairs = np.array(list(zip(lens_keys, src_keys)))[xi_rows]

        lens_tracers = list(np.unique(xi_pairs[:, 0]))
        src_tracers = list(np.unique(xi_pairs[:, 1]))
        lens_index = {t: i for i, t in enumerate(lens_tracers)}
        src_index = {t: j for j, t in enumerate(src_tracers)}

        row_lens_idx = np.full(len(lens_keys), -1, dtype=int)
        row_src_idx = np.full(len(src_keys), -1, dtype=int)
        for ridx in xi_rows:
            row_lens_idx[ridx] = lens_index[lens_keys[ridx]]
            row_src_idx[ridx] = src_index[src_keys[ridx]]

        # Use the first xi_t statistic as the template for N(z) grid,
        # assuming all the lens/source tracers share the same N(z) sampling
        idx_is_xit = (
            np.array([s.statistic.sacc_data_type for s in self.statistics])
            == "galaxy_shearDensity_xi_t"
        )
        z_l_arr = [
            s.statistic.source0.tracer_args.z
            for s in np.array(self.statistics)[idx_is_xit]
        ]
        z_s_arr = [
            s.statistic.source1.tracer_args.z
            for s in np.array(self.statistics)[idx_is_xit]
        ]
        z_l = z_l_arr[0]
        z_s = z_s_arr[0]
        # Double-check that all the tracers share the same N(z) sampling
        assert all(
            np.array_equal(z_l, z_l_tracer) for z_l_tracer in z_l_arr
        ), "N(z) sampling must be the same for all lens tracers."
        assert all(
            np.array_equal(z_s, z_s_tracer) for z_s_tracer in z_s_arr
        ), "N(z) sampling must be the same for all source tracers."

        # Build dN/dz libraries once per unique tracer
        def _get_lens_statistic(lt):
            """Get a statistic for a given lens tracer."""
            return next(
                s
                for s in self.statistics
                if s.statistic.sacc_data_type == "galaxy_shearDensity_xi_t"
                and s.statistic.source0.sacc_tracer == lt
            )

        def _get_src_statistic(st):
            """Get a statistic for a given source tracer."""
            return next(
                s
                for s in self.statistics
                if s.statistic.sacc_data_type == "galaxy_shearDensity_xi_t"
                and s.statistic.source1.sacc_tracer == st
            )

        nzL = np.stack(
            [
                _get_lens_statistic(lt).statistic.source0.tracer_args.dndz
                for lt in lens_tracers
            ]
        )
        nzS = np.stack(
            [
                _get_src_statistic(st).statistic.source1.tracer_args.dndz
                for st in src_tracers
            ]
        )

        # Normalize dN/dz
        nzL_norm = nzL / simpson(nzL, x=z_l, axis=1)[:, None]
        nzS_norm = nzS / simpson(nzS, x=z_s, axis=1)[:, None]

        # Cache
        self._pm_maps_ready = True
        self._pm_theta = theta
        self._pm_row_lens_idx = row_lens_idx
        self._pm_row_src_idx = row_src_idx
        self._pm_lens_tracers = lens_tracers
        self._pm_src_tracers = src_tracers
        self._pm_z_l = z_l
        self._pm_z_s = z_s
        self._pm_nzL_norm = nzL_norm
        self._pm_nzS_norm = nzS_norm
        self._pm_inv_cov_original = self.inv_cov

    def _prepare_integrand(self, cosmo: pyccl.Cosmology) -> np.ndarray:
        """Compute the cosmology-dependent portion of the integrand."""
        assert self._pm_z_l is not None
        assert self._pm_z_s is not None
        z_l = self._pm_z_l
        z_s = self._pm_z_s
        a_l = 1.0 / (1.0 + z_l)
        a_s = 1.0 / (1.0 + z_s)

        # Mask out pairs with z_s <= z_l (and z > 0)
        mask = (z_l[:, None] < z_s[None, :]) & (z_l[:, None] > 0) & (z_s[None, :] > 0)
        aL_all = np.broadcast_to(a_l[:, None], mask.shape)[mask]
        aS_all = np.broadcast_to(a_s[None, :], mask.shape)[mask]

        # Cosmology calculations
        sc_valid = pyccl.sigma_critical(cosmo, a_lens=aL_all, a_source=aS_all)
        sc = np.full(mask.shape, np.nan, dtype=float)
        sc[mask] = sc_valid
        D_l_valid = pyccl.angular_diameter_distance(cosmo, a_l)[z_l > 0]
        D_l = np.full(z_l.shape, np.nan, dtype=float)
        D_l[z_l > 0] = D_l_valid
        D_l = D_l[:, None]
        integrand = 1.0 / sc / (D_l * D_l)
        return np.nan_to_num(integrand, nan=0.0)

    def _compute_betas(self, cosmo: pyccl.Cosmology) -> np.ndarray:
        """Compute beta_ij factors for all bin combinations."""
        integrand = self._prepare_integrand(cosmo)
        assert self._pm_nzL_norm is not None
        assert self._pm_nzS_norm is not None
        assert self._pm_z_s is not None
        assert self._pm_z_l is not None
        nzL_norm = self._pm_nzL_norm
        nzS_norm = self._pm_nzS_norm

        inner_integral = simpson(
            integrand[:, None, :] * nzS_norm[None, :, :],
            x=self._pm_z_s,
            axis=2,
        )

        betas = (
            simpson(
                nzL_norm[:, :, None] * inner_integral[None, :, :],
                x=self._pm_z_l,
                axis=1,
            )
            / cosmo["h"]
        )

        return betas

    def _build_V(self, betas: np.ndarray) -> np.ndarray:
        """Construct the template matrix."""
        assert self._pm_row_src_idx is not None
        assert self._pm_row_lens_idx is not None
        assert self._pm_theta is not None
        V = np.zeros((len(self._pm_row_src_idx), len(self._pm_lens_tracers)))
        valid = (self._pm_row_lens_idx >= 0) & (self._pm_row_src_idx >= 0)
        beta_rows = np.zeros(valid.sum(), dtype=float)
        beta_rows = betas[self._pm_row_lens_idx[valid], self._pm_row_src_idx[valid]]
        V[valid, self._pm_row_lens_idx[valid]] = beta_rows / (
            self._pm_theta[valid] ** 2
        )
        return V

    def _compute_correction(
        self, cosmo: pyccl.Cosmology, sigma_B: float, point_mass: float
    ) -> np.ndarray:
        """Generate the inverse covariance correction."""
        inv_cov = self._pm_inv_cov_original
        assert inv_cov is not None

        betas = self._compute_betas(cosmo)
        V = self._build_V(betas)
        V = sigma_B * point_mass * V

        VCV = V.T @ (inv_cov @ V)
        IVCV = np.linalg.inv(np.eye(VCV.shape[0]) + VCV)
        CV_VCV = inv_cov @ (V @ IVCV)
        return (CV_VCV @ V.T) @ inv_cov

    def compute_pointmass(
        self,
        cosmo: pyccl.Cosmology,
        sigma_B: float = 10000.0,
        point_mass: float = 1e13,
    ) -> np.ndarray:
        """Update the inverse covariance matrix to marginalize over the point mass.

        The procedure follows MacCrann et al. (2019).

        :param cosmo: A pre-initialized CCL Cosmology object to compute distances.
        :param sigma_B: The PM prior width defined in  MacCrann et al. (2019).
        :param point_mass: The fiducial value of the point mass in units of Mpc/h.
        :return: The new inverse covariance matrix.
        """
        self._generate_maps()
        self.inv_cov_correction = self._compute_correction(cosmo, sigma_B, point_mass)
        self.inv_cov = self._pm_inv_cov_original - self.inv_cov_correction
        return self.inv_cov_correction

    @final
    @enforce_states(
        initial=[State.UPDATED, State.COMPUTED],
        terminal=State.COMPUTED,
        failure_message="update() must be called before compute_chisq()",
    )
    def compute_chisq(self, tools: ModelingTools) -> float:
        """Calculate and return the chi-squared for the given cosmology.

        We need to override the default calculation because the cholesky
        is incompatible with the correction term. Note that in some cases
        this may reduce the numerical stability of the Chi2 calculation.

        :param tools: ModelingTools to be used in the calculation of the theory vector
        :return: the chi-squared
        """
        if not self.inv_cov_correction:
            warnings.warn(
                "The inverse covariance correction has not yet been computed."
            )

        theory_vector: npt.NDArray[np.float64]
        data_vector: npt.NDArray[np.float64]
        residuals: npt.NDArray[np.float64]
        try:
            theory_vector = self.compute_theory_vector(tools)
            data_vector = self.get_data_vector()
        except NotImplementedError:
            data_vector, theory_vector = self.compute(tools)

        assert len(data_vector) == len(theory_vector)
        residuals = np.array(data_vector - theory_vector, dtype=np.float64)

        assert self.inv_cov is not None
        chisq = residuals @ self.inv_cov @ residuals
        return float(chisq)
