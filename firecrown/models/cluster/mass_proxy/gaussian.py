from typing import List, Union
import numpy.typing as npt
import numpy as np
from scipy import special
from firecrown.models.cluster.kernel import ArgReader, Kernel, KernelType


class MassRichnessGaussian(Kernel):
    def get_proxy_mean(
        self,
        mass: Union[float, npt.NDArray[np.float64]],
        z: Union[float, npt.NDArray[np.float64]],
    ):
        """Return observed quantity corrected by redshift and mass."""

    def get_proxy_sigma(
        self,
        mass: Union[float, npt.NDArray[np.float64]],
        z: Union[float, npt.NDArray[np.float64]],
    ):
        """Return observed scatter corrected by redshift and mass."""

    def _distribution_binned(self, args: List[float], args_map: ArgReader):
        mass = args_map.get_independent_val(args, KernelType.mass)
        z = args_map.get_independent_val(args, KernelType.z)
        mass_proxy_limits = args_map.get_extra_args(args, self.kernel_type)

        proxy_mean = self.get_proxy_mean(mass, z)
        proxy_sigma = self.get_proxy_sigma(mass, z)

        x_min = (proxy_mean - mass_proxy_limits[0] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
        )
        x_max = (proxy_mean - mass_proxy_limits[1] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
        )

        return_vals = np.empty_like(x_min)
        mask1 = (x_max > 3.0) | (x_min < -3.0)
        mask2 = ~mask1

        return_vals[mask1] = (
            -(special.erfc(x_min[mask1]) - special.erfc(x_max[mask1])) / 2.0
        )
        return_vals[mask2] = (
            special.erf(x_min[mask2]) - special.erf(x_max[mask2])
        ) / 2.0

        return return_vals

    def _distribution_unbinned(self, args: List[float], args_map: ArgReader):
        mass = args_map.get_independent_val(args, KernelType.mass)
        z = args_map.get_independent_val(args, KernelType.z)
        mass_proxy = args_map.get_extra_args(args, self.kernel_type)

        proxy_mean = self.get_proxy_mean(mass, z)
        proxy_sigma = self.get_proxy_sigma(mass, z)

        return np.exp(-0.5 * (mass_proxy - proxy_mean) ** 2 / proxy_sigma**2) / (
            2 * np.pi * proxy_sigma
        )
