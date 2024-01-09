from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift, TrueMass
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe


class TrueMassSpecZRecipe(ClusterRecipe):
    def __init__(self) -> None:
        super().__init__()

        self.integrator = NumCosmoIntegrator()
        self.redshift_distribution = SpectroscopicRedshift()
        self.mass_distribution = TrueMass()

    def get_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        average_on: Optional[ClusterProperty] = None,
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], float],
        npt.NDArray[np.float64],
    ]:
        """_summary_"""

        def theory_prediction(
            mass: npt.NDArray[np.float64], z: npt.NDArray[np.float64], sky_area: float
        ):
            prediction = (
                cluster_theory.comoving_volume(z, sky_area)
                * cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution()
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
            [npt.NDArray[np.float64], npt.NDArray[np.float64], float],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]:
        def numcosmo_wrapper(
            int_args: npt.NDArray[np.float64], extra_args: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]
            sky_area = extra_args[0]
            return prediction(mass, z, sky_area)

        return numcosmo_wrapper

    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        bin: NDimensionalBin,
        sky_area: float,
        average_on: Optional[ClusterProperty] = None,
    ) -> float:
        # Mock some fake mass to richness relation:
        mass_low = bin.mass_proxy_edges[0] + 13
        mass_high = bin.mass_proxy_edges[1] + 13
        self.integrator.integral_bounds = [
            (mass_low, mass_high),
            bin.z_edges,
        ]
        print(mass_low, mass_high)
        self.integrator.extra_args = np.array([sky_area], dtype=np.float64)

        theory_prediction = self.get_theory_prediction(cluster_theory, average_on)
        prediction_wrapper = self.get_function_to_integrate(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts
