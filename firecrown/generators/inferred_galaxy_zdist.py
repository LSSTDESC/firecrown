"""Generation of inferred galaxy redshift distributions."""

from typing import TypedDict, Annotated, Any
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, field_serializer, BeforeValidator

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, erf, erfc
from scipy.integrate import quad

from numcosmo_py import Ncm

from firecrown.metadata_types import (
    InferredGalaxyZDist,
    ALL_MEASUREMENT_TYPES,
    make_measurements_dict,
    Galaxies,
    CMB,
    Clusters,
)
from firecrown.metadata_functions import Measurement


BinsType = TypedDict("BinsType", {"edges": npt.NDArray, "sigma_z": float})

Y1_ALPHA = 0.94
Y1_BETA = 2.0
Y1_Z0 = 0.26
Y1_LENS_BINS: BinsType = {"edges": np.linspace(0.2, 1.2, 5 + 1), "sigma_z": 0.03}
Y1_SOURCE_BINS: BinsType = {"edges": np.linspace(0.2, 1.2, 5 + 1), "sigma_z": 0.05}

Y10_ALPHA = 0.90
Y10_BETA = 2.0
Y10_Z0 = 0.28
Y10_LENS_BINS: BinsType = {"edges": np.linspace(0.2, 1.2, 10 + 1), "sigma_z": 0.03}
Y10_SOURCE_BINS: BinsType = {"edges": np.linspace(0.2, 1.2, 10 + 1), "sigma_z": 0.05}


class ZDistLSSTSRD:
    r"""LSST Inferred galaxy redshift distributions.

    Inferred galaxy redshift distribution is based on the LSST Science
    Requirements Document (SRD), equation 5. Note that the SRD fixes
    $\beta = 2$.

    The values of $\alpha$ and $z_0$ are different for Year 1 and Year 10.
    `ZDistLLSTSRD` provides these values as defaults and allows for greater
    flexibility when desired.
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

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
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

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0)

    def distribution(self, z: npt.NDArray) -> npt.NDArray:
        """Generate the inferred galaxy redshift distribution.

        :param z: The redshifts at which to evaluate the distribution
        :return: The inferred galaxy redshift distribution
        """
        norma = self.alpha / (self.z0 * gamma((1.0 + self.beta) / self.alpha))

        return (
            norma * (z / self.z0) ** self.beta * np.exp(-((z / self.z0) ** self.alpha))
        )

    def _integrated_gaussian_scalar(
        self, zpl: float, zpu: float, sigma_z: float, z: float
    ) -> float:
        """Generate the integrated Gaussian distribution.

        :param zpl: The lower bound of the integration
        :param zpu: The upper bound of the integration
        :param sigma_z: The resolution parameter
        :param z: The redshifts at which to evaluate the distribution
        :return: The integrated distribution
        """
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
        measurements: set[Measurement],
        use_autoknot: bool = False,
        autoknots_reltol: float = 1.0e-4,
        autoknots_abstol: float = 1.0e-15,
    ) -> InferredGalaxyZDist:
        """Generate the inferred galaxy redshift distribution in bins.

        :param zpl: The lower bound of the integration
        :param zpu: The upper bound of the integration
        :param sigma_z: The resolution parameter
        :param z: The redshifts at which to evaluate the distribution
        :param name: The name of the distribution
        :param measurements: The set of measurements of the distribution
        :param use_autoknot: Whether to use the NotAKnot algorithm of NumCosmo
        :param autoknots_reltol: The relative tolerance for the NotAKnot algorithm
        :param autoknots_abstol: The absolute tolerance for the NotAKnot algorithm
        :return: The inferred galaxy redshift distribution
        """

        def _P(z, _):
            """A local closure.

            Used to create a function that captures the ZDistLSSTSRD state and
            provides an integrand suitable for scipy.integrate.quad.
            """
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
            s = Ncm.SplineCubicNotaknot()
            s.set_func1(
                Ncm.SplineFuncType.FUNCTION_SPLINE,
                _P,
                None,
                z[0],
                z[-1],
                0,
                autoknots_reltol,
            )
            z_knots = np.array(s.peek_xv().dup_array())  # pylint: disable-msg=no-member
            dndz = (
                np.array(s.peek_yv().dup_array())  # pylint: disable-msg=no-member
                / norma
            )

        return InferredGalaxyZDist(
            bin_name=name, z=z_knots, dndz=dndz, measurements=measurements
        )


class LinearGrid1D(BaseModel):
    """A 1D linear grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: float
    end: float
    num: int

    def generate(self) -> npt.NDArray:
        """Generate the 1D linear grid."""
        return np.linspace(self.start, self.end, self.num)


class RawGrid1D(BaseModel):
    """A 1D grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: list[float]

    def generate(self) -> npt.NDArray:
        """Generate the 1D grid."""
        return np.array(self.values)


Grid1D = LinearGrid1D | RawGrid1D


def make_measurements(
    value: set[Measurement] | list[dict[str, Any]]
) -> set[Measurement]:
    """Create a Measurement object from a dictionary."""
    if isinstance(value, set) and all(
        isinstance(v, ALL_MEASUREMENT_TYPES) for v in value
    ):
        return value

    measurements: set[Measurement] = set()
    for measurement_dict in value:
        if not isinstance(measurement_dict, dict):
            raise ValueError(f"Invalid Measurement: {value} is not a dictionary")

        if "subject" not in measurement_dict:
            raise ValueError(
                "Invalid Measurement: dictionary does not contain 'subject'"
            )

        subject = measurement_dict["subject"]

        match subject:
            case "Galaxies":
                measurements.update({Galaxies[measurement_dict["property"]]})
            case "CMB":
                measurements.update({CMB[measurement_dict["property"]]})
            case "Clusters":
                measurements.update({Clusters[measurement_dict["property"]]})
            case _:
                raise ValueError(
                    f"Invalid Measurement: subject: '{subject}' is not recognized"
                )
    return measurements


class ZDistLSSTSRDBin(BaseModel):
    """LSST Inferred galaxy redshift distributions in bins."""

    model_config = ConfigDict(extra="forbid")

    zpl: float
    zpu: float
    sigma_z: float
    z: Annotated[Grid1D, Field(union_mode="left_to_right")]
    bin_name: str
    measurements: Annotated[set[Measurement], BeforeValidator(make_measurements)]
    use_autoknot: bool = False
    autoknots_reltol: float = 1.0e-4
    autoknots_abstol: float = 1.0e-15

    @field_serializer("measurements")
    @classmethod
    def serialize_measurements(cls, value: set[Measurement]) -> list[dict]:
        """Serialize the Measurement."""
        return make_measurements_dict(value)

    def generate(self, zdist: ZDistLSSTSRD) -> InferredGalaxyZDist:
        """Generate the inferred galaxy redshift distribution in bins."""
        return zdist.binned_distribution(
            zpl=self.zpl,
            zpu=self.zpu,
            sigma_z=self.sigma_z,
            z=self.z.generate(),
            name=self.bin_name,
            measurements=self.measurements,
            use_autoknot=self.use_autoknot,
            autoknots_reltol=self.autoknots_reltol,
            autoknots_abstol=self.autoknots_abstol,
        )


class ZDistLSSTSRDBinCollection(BaseModel):
    """LSST Inferred galaxy redshift distributions in bins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float
    beta: float
    z0: float
    bins: list[ZDistLSSTSRDBin]

    def generate(self) -> list[InferredGalaxyZDist]:
        """Generate the inferred galaxy redshift distributions in bins."""
        zdist = ZDistLSSTSRD(alpha=self.alpha, beta=self.beta, z0=self.z0)
        return [bin.generate(zdist) for bin in self.bins]


LSST_Y1_LENS_BIN_COLLECTION = ZDistLSSTSRDBinCollection(
    alpha=Y1_ALPHA,
    beta=Y1_BETA,
    z0=Y1_Z0,
    bins=[
        ZDistLSSTSRDBin(
            zpl=zpl,
            zpu=zpu,
            sigma_z=Y1_LENS_BINS["sigma_z"],
            z=RawGrid1D(values=[0.0, 3.0]),
            bin_name=f"lens_{zpl:.1f}_{zpu:.1f}_y1",
            measurements={Galaxies.COUNTS},
            use_autoknot=True,
            autoknots_reltol=1.0e-5,
        )
        for zpl, zpu in pairwise(Y1_LENS_BINS["edges"])
    ],
)

LSST_Y1_SOURCE_BIN_COLLECTION = ZDistLSSTSRDBinCollection(
    alpha=Y1_ALPHA,
    beta=Y1_BETA,
    z0=Y1_Z0,
    bins=[
        ZDistLSSTSRDBin(
            zpl=zpl,
            zpu=zpu,
            sigma_z=Y1_SOURCE_BINS["sigma_z"],
            z=RawGrid1D(values=[0.0, 3.0]),
            bin_name=f"source_{zpl:.1f}_{zpu:.1f}_y1",
            measurements={Galaxies.SHEAR_E},
            use_autoknot=True,
            autoknots_reltol=1.0e-5,
        )
        for zpl, zpu in pairwise(Y1_SOURCE_BINS["edges"])
    ],
)

LSST_Y10_LENS_BIN_COLLECTION = ZDistLSSTSRDBinCollection(
    alpha=Y10_ALPHA,
    beta=Y10_BETA,
    z0=Y10_Z0,
    bins=[
        ZDistLSSTSRDBin(
            zpl=zpl,
            zpu=zpu,
            sigma_z=Y10_LENS_BINS["sigma_z"],
            z=RawGrid1D(values=[0.0, 3.0]),
            bin_name=f"lens_{zpl:.1f}_{zpu:.1f}_y10",
            measurements={Galaxies.COUNTS},
            use_autoknot=True,
            autoknots_reltol=1.0e-5,
        )
        for zpl, zpu in pairwise(Y10_LENS_BINS["edges"])
    ],
)

LSSST_Y10_SOURCE_BIN_COLLECTION = ZDistLSSTSRDBinCollection(
    alpha=Y10_ALPHA,
    beta=Y10_BETA,
    z0=Y10_Z0,
    bins=[
        ZDistLSSTSRDBin(
            zpl=zpl,
            zpu=zpu,
            sigma_z=Y10_SOURCE_BINS["sigma_z"],
            z=RawGrid1D(values=[0.0, 3.0]),
            bin_name=f"source_{zpl:.1f}_{zpu:.1f}_y10",
            measurements={Galaxies.SHEAR_E},
            use_autoknot=True,
            autoknots_reltol=1.0e-5,
        )
        for zpl, zpu in pairwise(Y10_SOURCE_BINS["edges"])
    ],
)
