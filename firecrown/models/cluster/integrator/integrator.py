from abc import ABC, abstractmethod
from firecrown.models.cluster.abundance import ClusterAbundance
from typing import Tuple, Callable
import numpy.typing as npt
import numpy as np


class Integrator(ABC):
    @abstractmethod
    def integrate(
        self,
        integrand: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                Tuple[float, float],
                Tuple[float, float],
            ],
            npt.NDArray[np.float64],
        ],
    ) -> float:
        """Integrate the integrand over the bounds and include extra_args to integral"""

    @abstractmethod
    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        """Set the limits of integration and extra arguments for the integral"""
