"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt

from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift
from firecrown.models.cluster.mass_proxy import MurataBinned
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe
from firecrown.models.cluster.deltasigma import ClusterDeltaSigma


class MurataBinnedSpecZDeltaSigmaRecipe(ClusterRecipe):
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
    """

    def __init__(self) -> None:
        super().__init__()

        self.integrator = NumCosmoIntegrator()
        self.redshift_distribution = SpectroscopicRedshift()
        pivot_mass, pivot_redshift = 14.625862906, 0.6
        self.mass_distribution = MurataBinned(pivot_mass, pivot_redshift)
        self.my_updatables.append(self.mass_distribution)

    def get_theory_prediction(
        self,
        cluster_theory: ClusterDeltaSigma,
        average_on: None | ClusterProperty = None,  # pylint: disable=unused-argument
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            tuple[float, float],
            float,
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass proxy limits,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy_limits: tuple[float, float],
            sky_area: float,
            radius_center: float,
        ):
            prediction = (
                cluster_theory.comoving_volume(z, sky_area)
                * cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )
            if average_on is None:
                # pylint: disable=no-member
                raise ValueError(
                    f"The property should be" f" {ClusterProperty.DELTASIGMA}."
                )

            for cluster_prop in ClusterProperty:
                if cluster_prop == ClusterProperty.DELTASIGMA:
                    prediction *= cluster_theory.delta_sigma(mass, z, radius_center, True, 0.3)
            return prediction

        return theory_prediction

    def get_function_to_integrate(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                tuple[float, float],
                float,
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            mass_proxy_low = extra_args[0]
            mass_proxy_high = extra_args[1]
            sky_area = extra_args[2]
            radius_center = extra_args[3]
            return prediction(
                mass, z, (mass_proxy_low, mass_proxy_high), sky_area, radius_center
            )

        return function_mapper

    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterDeltaSigma,
        this_bin: NDimensionalBin,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        self.integrator.integral_bounds = [
            (cluster_theory.min_mass, cluster_theory.max_mass),
            this_bin.z_edges,
        ]
        radius_center = this_bin.radius_center
        self.integrator.extra_args = np.array(
            [*this_bin.mass_proxy_edges, sky_area, radius_center]
        )
        theory_prediction = self.get_theory_prediction(cluster_theory, average_on)
        prediction_wrapper = self.get_function_to_integrate(theory_prediction)
        deltasigma = self.integrator.integrate(prediction_wrapper)
        return deltasigma

    def get_theory_prediction_counts(
        self,
        cluster_theory: ClusterDeltaSigma,
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[float, float], float],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass proxy limits,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy_limits: tuple[float, float],
            sky_area: float,
        ):
            prediction = (
                cluster_theory.comoving_volume(z, sky_area)
                * cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )
            return prediction

        return theory_prediction

    def get_function_to_integrate_counts(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                tuple[float, float],
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            mass_proxy_low = extra_args[0]
            mass_proxy_high = extra_args[1]
            sky_area = extra_args[2]

            return prediction(mass, z, (mass_proxy_low, mass_proxy_high), sky_area)

        return function_mapper

    def evaluate_theory_prediction_counts(
        self,
        cluster_theory: ClusterDeltaSigma,
        this_bin: NDimensionalBin,
        sky_area: float,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        self.integrator.integral_bounds = [
            (cluster_theory.min_mass, cluster_theory.max_mass),
            this_bin.z_edges,
        ]
        self.integrator.extra_args = np.array([*this_bin.mass_proxy_edges, sky_area])

        theory_prediction = self.get_theory_prediction_counts(cluster_theory)
        prediction_wrapper = self.get_function_to_integrate_counts(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts
