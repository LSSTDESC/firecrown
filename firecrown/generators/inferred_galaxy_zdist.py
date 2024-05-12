"""Generation of inferred galaxy redshift distributions."""

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, erf, erfc
from scipy.integrate import quad

from numcosmo_py import Ncm

from firecrown.metadata.two_point import InferredGalaxyZDist, MeasuredType

Y1_ALPHA = 0.94
Y1_BETA = 2.0
Y1_Z0 = 0.26
Y1_LENS_BINS = {"edges": np.linspace(0.2, 1.2, 5 + 1), "sigma_z": 0.03}
Y1_SOURCE_BINS = {"edges": np.linspace(0.2, 1.2, 5 + 1), "sigma_z": 0.05}

Y10_ALPHA = 0.90
Y10_BETA = 2.0
Y10_Z0 = 0.28
Y10_LENS_BINS = {"edges": np.linspace(0.2, 1.2, 10 + 1), "sigma_z": 0.03}
Y10_SOURCE_BINS = {"edges": np.linspace(0.2, 1.2, 10 + 1), "sigma_z": 0.05}


class ZDistLSSTSRD:
    """LSST Inferred galaxy redshift distributions.

    Inferred galaxy redshift distribution based on the LSST Science Requirements
    Document (SRD).
    """

    def __init__(self, alpha: float, beta: float, z0: float) -> None:
        """Initialize the LSST Inferred galaxy redshift distribution.

        :param alpha: The alpha parameter of the distribution.
        :param beta: The beta parameter of the distribution.
        :param z0: The z0 parameter of the distribution.
        """
        self.alpha = alpha
        self.beta = beta
        self.z0 = z0

    @classmethod
    def year_1(
        cls, alpha: float = Y1_ALPHA, beta: float = Y1_BETA, z0: float = Y1_Z0
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the first year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 1.

        :param alpha: The alpha parameter using the default value of 0.94.
        :param beta: The beta parameter using the default value of 2.0.
        :param z0: The z0 parameter using the default value of 0.26.
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0)

    @classmethod
    def year_10(
        cls, alpha: float = Y10_ALPHA, beta: float = Y10_BETA, z0: float = Y10_Z0
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the tenth year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 10.

        :param alpha: The alpha parameter using the default value of 0.90.
        :param beta: The beta parameter using the default value of 2.0.
        :param z0: The z0 parameter using the default value of 0.28.
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0)

    def distribution(self, z: npt.NDArray) -> npt.NDArray:
        """Generate the inferred galaxy redshift distribution."""
        norma = self.alpha / (self.z0 * gamma((1.0 + self.beta) / self.alpha))

        return (
            norma * (z / self.z0) ** self.beta * np.exp(-((z / self.z0) ** self.alpha))
        )

    def _integrated_gaussian_scalar(
        self, zpl: float, zpu: float, sigma_z: float, z: float
    ) -> float:
        """Generate the integrated Gaussian distribution."""
        denom = np.sqrt(2.0) * sigma_z * (1.0 + z)
        if (z - zpu) > 0.0:
            return -(erfc((z - zpl) / denom) - erfc((z - zpu) / denom)) / erfc(
                -z / denom
            )
        if (z - zpl) < 0.0:
            return (erfc((zpl - z) / denom) - erfc((zpu - z) / denom)) / erfc(
                -z / denom
            )

        return (erf((z - zpl) / denom) - erf((z - zpu) / denom)) / erfc(-z / denom)

    def _integrated_gaussian(
        self, zpl: float, zpu: float, sigma_z: float, z: npt.NDArray
    ) -> npt.NDArray:
        """Generate the integrated Gaussian distribution."""
        denom = np.sqrt(2.0) * sigma_z * (1.0 + z)
        result = np.zeros_like(denom)
        erfc_up = (z - zpu) > 0.0
        erfc_low = (z - zpl) < 0.0
        rest = ~(erfc_up | erfc_low)

        result[erfc_up] = -(
            erfc((z[erfc_up] - zpl) / denom[erfc_up])
            - erfc((z[erfc_up] - zpu) / denom[erfc_up])
        ) / erfc(-z[erfc_up] / denom[erfc_up])
        result[erfc_low] = (
            erfc((zpl - z[erfc_low]) / denom[erfc_low])
            - erfc((zpu - z[erfc_low]) / denom[erfc_low])
        ) / erfc(-z[erfc_low] / denom[erfc_low])
        result[rest] = (
            erf((z[rest] - zpl) / denom[rest]) - erf((z[rest] - zpu) / denom[rest])
        ) / erfc(-z[rest] / denom[rest])

        return result

    def binned_distribution(
        self,
        *,
        zpl: float,
        zpu: float,
        sigma_z: float,
        z: npt.NDArray,
        name: str,
        measured_type: MeasuredType,
        use_autoknot: bool = False,
        autoknots_reltol: float = 1.0e-4,
        autoknots_abstol: float = 1.0e-15,
    ) -> InferredGalaxyZDist:
        """Generate the inferred galaxy redshift distribution in bins."""

        def _P(z, _):
            return (
                self.distribution(z)
                * self._integrated_gaussian_scalar(zpl, zpu, sigma_z, z)
                + autoknots_abstol
            )

        norma = quad(_P, z[0], z[-1], args=None)[0]

        if not use_autoknot:
            z_knots = z
            dndz = (
                self._integrated_gaussian(zpl, zpu, sigma_z, z)
                * self.distribution(z)
                / norma
            )
        else:
            s = Ncm.SplineCubicNotaknot.new()
            s.set_func1(
                Ncm.SplineFuncType.FUNCTION_SPLINE,
                _P,
                None,
                z[0],
                z[-1],
                0,
                autoknots_reltol,
            )
            z_knots = np.array(s.peek_xv().dup_array())
            dndz = np.array(s.peek_yv().dup_array()) / norma

        return InferredGalaxyZDist(
            bin_name=name, z=z_knots, dndz=dndz, measured_type=measured_type
        )
