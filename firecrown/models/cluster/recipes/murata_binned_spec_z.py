from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift
from firecrown.models.cluster.mass_proxy import MurataBinned
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe


class MurataBinnedSpecZRecipe(ClusterRecipe):
    def __init__(self) -> None:
        super().__init__()

        self.integrator = NumCosmoIntegrator()
        self.redshift_distribution = SpectroscopicRedshift()
        pivot_mass, pivot_redshift = 14.625862906, 0.6
        self.mass_distribution = MurataBinned(pivot_mass, pivot_redshift)
        self.my_updatables.append(self.mass_distribution)

    def get_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        average_on: Optional[ClusterProperty] = None,
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[float, float], float],
        npt.NDArray[np.float64],
    ]:
        """_summary_"""

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy_limits: Tuple[float, float],
            sky_area: float,
        ):
            prediction = (
                cluster_theory.comoving_volume(z, sky_area)
                * cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )

            if average_on is None:
                return prediction

            for cluster_prop in ClusterProperty:
                include_prop = cluster_prop & average_on
                if not include_prop:
                    continue
                if cluster_prop == ClusterProperty.MASS:
                    prediction *= mass
                elif cluster_prop == ClusterProperty.REDSHIFT:
                    prediction *= z
                else:
                    raise NotImplementedError(f"Average for {cluster_prop}.")

            return prediction

        return theory_prediction

    def get_function_to_integrate(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                Tuple[float, float],
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
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

    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        bin: NDimensionalBin,
        sky_area: float,
        average_on: Optional[ClusterProperty] = None,
    ) -> float:
        self.integrator.integral_bounds = [
            (cluster_theory.min_mass, cluster_theory.max_mass),
            bin.z_edges,
        ]
        self.integrator.extra_args = np.array([*bin.mass_proxy_edges, sky_area])

        theory_prediction = self.get_theory_prediction(cluster_theory, average_on)
        prediction_wrapper = self.get_function_to_integrate(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts
