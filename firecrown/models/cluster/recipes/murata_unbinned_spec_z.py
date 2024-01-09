"""Module for defining the classes used in the MurataUnbinnedSpecZ cluster recipe."""
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import NDimensionalBin
from firecrown.models.cluster.integrator.numcosmo_integrator import NumCosmoIntegrator
from firecrown.models.cluster.kernel import SpectroscopicRedshift
from firecrown.models.cluster.mass_proxy import MurataUnbinned
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe


# pylint: disable=R0801
class MurataUnbinnedSpecZ(ClusterRecipe):
    """Cluster recipe using the Murata 2019 unbinned mass-richness relation and assuming
    perfectly measured spec-zs."""

    def __init__(self) -> None:
        super().__init__()

        self.integrator = NumCosmoIntegrator()
        self.redshift_distribution = SpectroscopicRedshift()
        pivot_mass, pivot_redshift = 14.625862906, 0.6
        self.mass_distribution = MurataUnbinned(pivot_mass, pivot_redshift)
        self.my_updatables.append(self.mass_distribution)

    def get_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        average_on: Optional[ClusterProperty] = None,
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        import pdb

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy: npt.NDArray[np.float64],
            sky_area: float,
        ):
            pdb.set_trace()
            prediction = (
                cluster_theory.comoving_volume(z, sky_area)
                * cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy)
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
                npt.NDArray[np.float64],
                float,
            ],
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
            mass_proxy = int_args[:, 2]
            sky_area = extra_args[0]
            return prediction(mass, z, mass_proxy, sky_area)

        return numcosmo_wrapper

    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        this_bin: NDimensionalBin,
        sky_area: float,
        average_on: Optional[ClusterProperty] = None,
    ) -> float:
        """Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 unbinned mass-richness relation and assuming perfectly
        measured redshifts."""
        self.integrator.integral_bounds = [
            (cluster_theory.min_mass, cluster_theory.max_mass),
            this_bin.z_edges,
            this_bin.mass_proxy_edges,
        ]
        self.integrator.extra_args = np.array([*this_bin.mass_proxy_edges, sky_area])

        theory_prediction = self.get_theory_prediction(cluster_theory, average_on)
        prediction_wrapper = self.get_function_to_integrate(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts
