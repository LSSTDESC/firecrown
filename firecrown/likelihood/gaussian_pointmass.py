"""Provides GaussFamily concrete types."""

from __future__ import annotations
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
import pyccl

from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.statistic import Statistic


# Default values for point mass marginalization
DEFAULT_SIGMA_B = 10000.0
DEFAULT_POINT_MASS = 1e13


@dataclass
class PointMassData:
    """Container for precomputed point mass data."""

    theta: np.ndarray
    row_lens_idx: np.ndarray
    row_src_idx: np.ndarray
    lens_tracers: list
    src_tracers: list
    z_l: np.ndarray
    z_s: np.ndarray
    nzL_norm: np.ndarray
    nzS_norm: np.ndarray


class ConstGaussianPM(ConstGaussian):
    """A Gaussian log-like with constant covariance, marginalizing over a point mass."""

    def __init__(self, statistics: Sequence[Statistic]) -> None:
        """Initialize the ConstGaussianPM object.

        :param statistics: A list of statistics for chi-squared calculations
        """
        # Use direct inverse covariance instead of Cholesky for chi-squared
        # calculation, because point mass correction makes Cholesky incompatible
        super().__init__(statistics, use_cholesky=False)
        # Initialize point mass marginalization attributes
        self._pm_maps_ready: bool = False
        self._pm_data: PointMassData | None = None
        self._pm_inv_cov_original: np.ndarray | None = None
        self.inv_cov_correction: np.ndarray | None = None

    def compute_chisq_impl(self, residuals: npt.NDArray[np.float64]) -> float:
        """Override chi-squared calculation to use direct inv_cov.

        We must use direct inv_cov multiplication because the point mass
        correction makes the Cholesky decomposition incompatible.

        :param residuals: The residuals (data - theory)
        :return: The chi-squared value
        """
        if self.inv_cov_correction is None:
            warnings.warn(
                "The inverse covariance correction has not yet been computed."
            )
        return self._compute_chisq_direct(residuals)

    def _collect_data_vectors(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect data types, lens keys, source keys, and theta from statistics.

        All statistics must have TwoPoint-compatible attributes.

        :return: Tuple of (data_types, lens_keys, src_keys, theta)
        """
        # Validate that all statistics have the required attributes
        required_attrs = ["sacc_data_type", "source0", "source1", "thetas"]
        for stat in self.statistics:
            stat_obj = stat.statistic
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(stat_obj, attr)
            ]
            if missing_attrs:
                raise StopIteration(
                    f"Statistic {type(stat_obj)} missing attributes: {missing_attrs}"
                )

        data_types = np.concatenate(
            [
                np.repeat(
                    stat.statistic.sacc_data_type,  # type: ignore[attr-defined]
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        lens_keys = np.concatenate(
            [
                np.repeat(
                    stat.statistic.source0.sacc_tracer,  # type: ignore[attr-defined]
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        src_keys = np.concatenate(
            [
                np.repeat(
                    stat.statistic.source1.sacc_tracer,  # type: ignore[attr-defined]
                    len(stat.statistic.get_data_vector()),
                )
                for stat in self.statistics
            ]
        )
        # Collect theta values (requires TwoPoint statistics)
        theta_list = [
            stat.statistic.thetas  # type: ignore[attr-defined]
            for stat in self.statistics
        ]
        theta = np.radians(np.concatenate(theta_list) / 60)
        return data_types, lens_keys, src_keys, theta

    def _extract_xi_t_pairs(
        self, data_types: np.ndarray, lens_keys: np.ndarray, src_keys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract galaxy_shearDensity_xi_t pairs from data.

        :param data_types: Array of data types
        :param lens_keys: Array of lens tracer keys
        :param src_keys: Array of source tracer keys
        :return: Tuple of (xi_rows, xi_pairs)
        """
        assert (
            "galaxy_shearDensity_xi_t" in data_types
        ), "Data must contain at least one 'galaxy_shearDensity_xi_t' datavector."

        xi_mask = np.array([dt == "galaxy_shearDensity_xi_t" for dt in data_types])
        xi_rows = np.where(xi_mask)[0]
        xi_pairs = np.array(list(zip(lens_keys, src_keys)))[xi_rows]
        return xi_rows, xi_pairs

    def _create_tracer_indices(
        self,
        xi_pairs: np.ndarray,
        xi_rows: np.ndarray,
        lens_keys: np.ndarray,
        src_keys: np.ndarray,
    ) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Create tracer index mappings.

        :param xi_pairs: Array of (lens, source) tracer pairs
        :param xi_rows: Row indices for xi_t data
        :param lens_keys: Array of lens tracer keys
        :param src_keys: Array of source tracer keys
        :return: Tuple of (lens_tracers, src_tracers, row_lens_idx, row_src_idx)
        """
        lens_tracers = list(np.unique(xi_pairs[:, 0]))
        src_tracers = list(np.unique(xi_pairs[:, 1]))
        lens_index = {t: i for i, t in enumerate(lens_tracers)}
        src_index = {t: j for j, t in enumerate(src_tracers)}

        row_lens_idx = np.full(len(lens_keys), -1, dtype=int)
        row_src_idx = np.full(len(src_keys), -1, dtype=int)
        for ridx in xi_rows:
            row_lens_idx[ridx] = lens_index[lens_keys[ridx]]
            row_src_idx[ridx] = src_index[src_keys[ridx]]

        return lens_tracers, src_tracers, row_lens_idx, row_src_idx

    def _get_redshift_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Get and validate redshift grids from xi_t statistics.

        :return: Tuple of (z_l, z_s) redshift arrays
        """
        # Use the first xi_t statistic as the template for N(z) grid,
        # assuming all the lens/source tracers share the same N(z) sampling
        sacc_types = [
            s.statistic.sacc_data_type  # type: ignore[attr-defined]
            for s in self.statistics
        ]
        idx_is_xit = np.array(sacc_types) == "galaxy_shearDensity_xi_t"
        xi_t_stats = np.array(self.statistics)[idx_is_xit]
        z_l_arr = [s.statistic.source0.tracer_args.z for s in xi_t_stats]
        z_s_arr = [s.statistic.source1.tracer_args.z for s in xi_t_stats]
        z_l = z_l_arr[0]
        z_s = z_s_arr[0]

        # Double-check that all the tracers share the same N(z) sampling
        assert all(
            np.array_equal(z_l, z_l_tracer) for z_l_tracer in z_l_arr
        ), "N(z) sampling must be the same for all lens tracers."
        assert all(
            np.array_equal(z_s, z_s_tracer) for z_s_tracer in z_s_arr
        ), "N(z) sampling must be the same for all source tracers."

        return z_l, z_s

    def _compute_normalized_dndz(
        self, lens_tracers: list, src_tracers: list, z_l: np.ndarray, z_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalized dN/dz distributions for all tracers.

        :param lens_tracers: List of lens tracer names
        :param src_tracers: List of source tracer names
        :param z_l: Lens redshift grid
        :param z_s: Source redshift grid
        :return: Tuple of (nzL_norm, nzS_norm) normalized dN/dz arrays
        """
        # Build dN/dz libraries once per unique tracer
        nzL_list = [
            self._get_lens_statistic(lt).statistic.source0.tracer_args.dndz
            for lt in lens_tracers
        ]
        nzS_list = [
            self._get_src_statistic(st).statistic.source1.tracer_args.dndz
            for st in src_tracers
        ]
        nzL = np.stack(nzL_list)
        nzS = np.stack(nzS_list)

        # Normalize dN/dz
        nzL_norm = nzL / simpson(nzL, x=z_l, axis=1)[:, None]
        nzS_norm = nzS / simpson(nzS, x=z_s, axis=1)[:, None]

        return nzL_norm, nzS_norm

    def _generate_maps(self) -> None:  # pylint: disable=too-many-locals
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
            assert self._pm_data is not None
            return

        # Collect data vectors
        data_types, lens_keys, src_keys, theta = self._collect_data_vectors()

        # Extract xi_t pairs
        xi_rows, xi_pairs = self._extract_xi_t_pairs(data_types, lens_keys, src_keys)

        # Create tracer index mappings
        lens_tracers, src_tracers, row_lens_idx, row_src_idx = (
            self._create_tracer_indices(xi_pairs, xi_rows, lens_keys, src_keys)
        )

        # Get and validate redshift grids
        z_l, z_s = self._get_redshift_grids()

        # Compute normalized dN/dz
        nzL_norm, nzS_norm = self._compute_normalized_dndz(
            lens_tracers, src_tracers, z_l, z_s
        )

        # Create precomputed data
        pm_data = PointMassData(
            theta=theta,
            row_lens_idx=row_lens_idx,
            row_src_idx=row_src_idx,
            lens_tracers=lens_tracers,
            src_tracers=src_tracers,
            z_l=z_l,
            z_s=z_s,
            nzL_norm=nzL_norm,
            nzS_norm=nzS_norm,
        )
        self._pm_maps_ready = True
        self._pm_data = pm_data
        self._pm_inv_cov_original = self.inv_cov

    def _get_statistic(self, tracer: str, is_lens: bool):
        """Get a statistic for a given tracer.

        :param tracer: The tracer name
        :param is_lens: True for lens tracer, False for source tracer
        :return: The GuardedStatistic for this tracer
        """
        source_attr = "source0" if is_lens else "source1"

        for s in self.statistics:
            stat = s.statistic
            is_xi_t = (
                stat.sacc_data_type  # type: ignore[attr-defined]
                == "galaxy_shearDensity_xi_t"
            )
            source_obj = getattr(stat, source_attr)
            is_match = source_obj.sacc_tracer == tracer
            if is_xi_t and is_match:
                # Validate it has TwoPoint-compatible attributes
                assert hasattr(stat, source_attr) and hasattr(stat, "sacc_data_type")
                return s
        tracer_type = "lens" if is_lens else "source"
        raise StopIteration(f"No {tracer_type} statistic found for {tracer}")

    def _get_lens_statistic(self, lens_tracer: str):
        """Get a statistic for a given lens tracer.

        :param lens_tracer: The lens tracer name
        :return: The GuardedStatistic for this lens tracer
        """
        return self._get_statistic(lens_tracer, is_lens=True)

    def _get_src_statistic(self, src_tracer: str):
        """Get a statistic for a given source tracer.

        :param src_tracer: The source tracer name
        :return: The GuardedStatistic for this source tracer
        """
        return self._get_statistic(src_tracer, is_lens=False)

    def _prepare_integrand(self, cosmo: pyccl.Cosmology) -> np.ndarray:
        """Compute the cosmology-dependent portion of the integrand."""
        assert self._pm_data is not None
        z_l = self._pm_data.z_l
        z_s = self._pm_data.z_s
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
        assert self._pm_data is not None
        nzL_norm = self._pm_data.nzL_norm
        nzS_norm = self._pm_data.nzS_norm

        inner_integral = simpson(
            integrand[:, None, :] * nzS_norm[None, :, :],
            x=self._pm_data.z_s,
            axis=2,
        )

        betas = (
            simpson(
                nzL_norm[:, :, None] * inner_integral[None, :, :],
                x=self._pm_data.z_l,
                axis=1,
            )
            / cosmo["h"]
        )

        return betas

    def _build_V(self, betas: np.ndarray) -> np.ndarray:
        """Construct the template matrix."""
        assert self._pm_data is not None
        V = np.zeros((len(self._pm_data.row_src_idx), len(self._pm_data.lens_tracers)))
        valid = (self._pm_data.row_lens_idx >= 0) & (self._pm_data.row_src_idx >= 0)
        beta_rows = np.zeros(valid.sum(), dtype=float)
        beta_rows = betas[
            self._pm_data.row_lens_idx[valid], self._pm_data.row_src_idx[valid]
        ]
        V[valid, self._pm_data.row_lens_idx[valid]] = beta_rows / (
            self._pm_data.theta[valid] ** 2
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
        sigma_B: float = DEFAULT_SIGMA_B,
        point_mass: float = DEFAULT_POINT_MASS,
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
