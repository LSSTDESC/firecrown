"""Generation of inferred galaxy redshift distributions."""

from typing import TypedDict, Annotated, Any, Unpack
from itertools import pairwise
from functools import cache

from pydantic import BaseModel, ConfigDict, Field, field_serializer, BeforeValidator

import numpy as np
import numpy.typing as npt
from scipy.special import gamma, erf, erfc
from scipy.integrate import quad

from numcosmo_py import Ncm

from firecrown.metadata_types import InferredGalaxyZDist, Galaxies
from firecrown.metadata_functions import (
    Measurement,
    make_measurements,
    make_measurements_dict,
)


BinsType = TypedDict("BinsType", {"edges": npt.NDArray, "sigma_z": float})

Y1_LENS_ALPHA = 0.94
Y1_LENS_BETA = 2.0
Y1_LENS_Z0 = 0.26

Y1_SOURCE_ALPHA = 0.78
Y1_SOURCE_BETA = 2.0
Y1_SOURCE_Z0 = 0.13

Y10_LENS_ALPHA = 0.90
Y10_LENS_BETA = 2.0
Y10_LENS_Z0 = 0.28

Y10_SOURCE_ALPHA = 0.68
Y10_SOURCE_BETA = 2.0
Y10_SOURCE_Z0 = 0.11


class ZDistLSSTSRDOpt(TypedDict, total=False):
    """Optional parameters for the LSST Inferred galaxy redshift distribution."""

    max_z: float
    use_autoknot: bool
    autoknots_reltol: float
    autoknots_abstol: float


class ZDistLSSTSRD:
    r"""LSST Inferred galaxy redshift distributions.

    Inferred galaxy redshift distribution is based on the LSST Science
    Requirements Document (SRD), equation 5. Note that the SRD fixes
    :math:`\beta = 2`.

    The values of :math:`\alpha` and :math:`z_0` are different for Year 1 and Year 10.
    `ZDistLLSTSRD` provides these values as defaults and allows for greater
    flexibility when desired.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        z0: float,
        max_z: float = 5.0,
        use_autoknot: bool = False,
        autoknots_reltol: float = 1.0e-4,
        autoknots_abstol: float = 1.0e-15,
    ) -> None:
        """Initialize the LSST Inferred galaxy redshift distribution.

        The true redshift distribution is integrated up to `max_z` to generate the
        photo-z distribution.

        :param alpha: The alpha parameter of the distribution.
        :param beta: The beta parameter of the distribution.
        :param z0: The z0 parameter of the distribution.
        :param max_z: The maximum true redshift to consider.
        :param use_autoknot: Whether to use the AutoKnots algorithm of NumCosmo
        :param autoknots_reltol: The relative tolerance for the AutoKnots algorithm
        :param autoknots_abstol: The absolute tolerance for the AutoKnots algorithm
        """
        self.alpha = alpha
        self.beta = beta
        self.z0 = z0
        self.max_z = max_z
        self.use_autoknot = use_autoknot
        self.autoknots_reltol = autoknots_reltol
        self.autoknots_abstol = autoknots_abstol

    @classmethod
    def year_1_lens(
        cls,
        alpha: float = Y1_LENS_ALPHA,
        beta: float = Y1_LENS_BETA,
        z0: float = Y1_LENS_Z0,
        **kwargs: Unpack[ZDistLSSTSRDOpt],
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the first year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 1 for the lens distribution.

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0, **kwargs)

    @classmethod
    def year_1_source(
        cls,
        alpha: float = Y1_SOURCE_ALPHA,
        beta: float = Y1_SOURCE_BETA,
        z0: float = Y1_SOURCE_Z0,
        **kwargs: Unpack[ZDistLSSTSRDOpt],
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the first year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 1 for the source distribution.

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0, **kwargs)

    @classmethod
    def year_10_lens(
        cls,
        alpha: float = Y10_LENS_ALPHA,
        beta: float = Y10_LENS_BETA,
        z0: float = Y10_LENS_Z0,
        **kwargs: Unpack[ZDistLSSTSRDOpt],
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the tenth year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 10 for the lens distribution.

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0, **kwargs)

    @classmethod
    def year_10_source(
        cls,
        alpha: float = Y10_SOURCE_ALPHA,
        beta: float = Y10_SOURCE_BETA,
        z0: float = Y10_SOURCE_Z0,
        **kwargs: Unpack[ZDistLSSTSRDOpt],
    ) -> "ZDistLSSTSRD":
        """Create a ZDistLSSTSRD object for the tenth year of LSST.

        It uses the default values of the alpha, beta and z0 parameters from
        the LSST SRD Year 10 for the source distribution.

        :param alpha: The alpha parameter of the distribution
        :param beta: The beta parameter of the distribution
        :param z0: The z0 parameter of the distribution
        :return: A ZDistLSSTSRD object.
        """
        return cls(alpha=alpha, beta=beta, z0=z0, **kwargs)

    def distribution(self, z: npt.NDArray) -> npt.NDArray:
        """Generate the inferred galaxy redshift distribution.

        :param z: The redshifts at which to evaluate the distribution
        :return: The inferred galaxy redshift distribution
        """
        norma = self.alpha / (self.z0 * gamma((1.0 + self.beta) / self.alpha))

        return (
            norma * (z / self.z0) ** self.beta * np.exp(-((z / self.z0) ** self.alpha))
        )

    def distribution_zp(self, zp: float, sigma_z: float) -> float:
        """Generate the Gaussian convolution of the distribution.

        :param sigma_z: The resolution parameter
        :param zp: The photometric redshift
        :return: The Gaussian distribution
        """
        sqrt_2 = np.sqrt(2.0)
        sqrt_2pi = np.sqrt(2.0 * np.pi)

        def integrand(z):
            lsigma_z = sigma_z * (1.0 + z)
            return (
                self.distribution(z)
                * np.exp(-((zp - z) ** 2) / (2.0 * lsigma_z**2))
                / (sqrt_2pi * lsigma_z * 0.5 * (erf(z / (sqrt_2 * lsigma_z)) + 1.0))
            )

        return quad(integrand, 0.0, self.max_z)[0]

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

    def compute_distribution(self, sigma_z: float) -> Ncm.StatsDist1d:
        """Generate the inferred galaxy redshift distribution.

        Computes the distribution by convolving the true distribution with a
        Gaussian. The convolution is computed using the AutoKnots algorithm of
        NumCosmo.

        :param sigma_z: The resolution parameter
        :return: The inferred galaxy redshift distribution
        """

        def m2lndist(z, _):
            return -2.0 * np.log(
                self.distribution_zp(z, sigma_z) + self.autoknots_abstol
            )

        s = Ncm.SplineCubicNotaknot()
        s.set_func1(
            Ncm.SplineFuncType.FUNCTION_SPLINE,
            m2lndist,
            None,
            0.0,
            self.max_z,
            0,
            self.autoknots_reltol,
        )

        stats = Ncm.StatsDist1dSpline.new(s)
        stats.props.abstol = self.autoknots_abstol
        stats.set_compute_cdf(True)
        stats.prepare()

        return stats

    def compute_true_distribution(self) -> Ncm.StatsDist1d:
        """Generate the inferred galaxy redshift distribution.

        Computes the distribution without the convolution with the Gaussian.
        That is, the true redshift distribution.

        :return: The inferred galaxy redshift distribution
        """

        def m2lndist(z, _):
            return -2.0 * np.log(self.distribution(z) + self.autoknots_abstol)

        s = Ncm.SplineCubicNotaknot()
        s.set_func1(
            Ncm.SplineFuncType.FUNCTION_SPLINE,
            m2lndist,
            None,
            0.0,
            self.max_z,
            0,
            self.autoknots_reltol,
        )

        stats = Ncm.StatsDist1dSpline.new(s)
        stats.props.abstol = self.autoknots_abstol
        stats.set_compute_cdf(True)
        stats.prepare()

        return stats

    def equal_area_bins(
        self,
        n_bins: int,
        sigma_z: float,
        last_z: float,
        use_true_distribution: bool = False,
    ) -> npt.NDArray:
        """Generate equal area bins for the distribution.

        In order to compute the bin edges, the convolution of the distribution
        with a Gaussian is computed. The bin edges are then computed by
        inverting the cumulative distribution function of the convolution.

        If the true distribution is used, the convolution is not computed.
        This provides a faster way to compute the bin edges.

        :param n_bins: The number of bins
        :param sigma_z: The resolution parameter
        :param last_z: The last redshift to consider
        :param use_true_distribution: Whether to use the true distribution

        :return: The bin edges
        """
        if use_true_distribution:
            stats = self.compute_true_distribution()
        else:
            stats = self.compute_distribution(sigma_z)

        total_area = stats.eval_pdf(last_z)

        return np.array(
            [stats.eval_inv_pdf(i * total_area / n_bins) for i in range(n_bins + 1)]
        )

    def binned_distribution(
        self,
        *,
        zpl: float,
        zpu: float,
        sigma_z: float,
        z: npt.NDArray,
        name: str,
        measurements: set[Measurement],
    ) -> InferredGalaxyZDist:
        """Generate the inferred galaxy redshift distribution in bins.

        :param zpl: The lower bound of the integration
        :param zpu: The upper bound of the integration
        :param sigma_z: The resolution parameter
        :param z: The redshifts at which to evaluate the distribution
        :param name: The name of the distribution
        :param measurements: The set of measurements of the distribution

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
                + self.autoknots_abstol
            )

        norma = quad(lambda x: _P(x, None), z[0], z[-1], args=())[0]

        if not self.use_autoknot:
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
                self.autoknots_reltol,
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


class ZDistLSSTSRDBin(BaseModel):
    """LSST Inferred galaxy redshift distributions in bins."""

    model_config = ConfigDict(extra="forbid")

    zpl: float
    zpu: float
    sigma_z: float
    z: Annotated[Grid1D, Field(union_mode="left_to_right")]
    bin_name: str
    measurements: Annotated[set[Measurement], BeforeValidator(make_measurements)]

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
        )


class ZDistLSSTSRDBinCollection(BaseModel):
    """LSST Inferred galaxy redshift distributions in bins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float
    beta: float
    z0: float
    bins: list[ZDistLSSTSRDBin]
    max_z: float = 5.0
    use_autoknot: bool = True
    autoknots_reltol: float = 1.0e-4
    autoknots_abstol: float = 1.0e-15

    def generate(self) -> list[InferredGalaxyZDist]:
        """Generate the inferred galaxy redshift distributions in bins."""
        zdist = ZDistLSSTSRD(
            alpha=self.alpha,
            beta=self.beta,
            z0=self.z0,
            use_autoknot=self.use_autoknot,
            autoknots_reltol=self.autoknots_reltol,
            autoknots_abstol=self.autoknots_abstol,
        )
        return [bin.generate(zdist) for bin in self.bins]


@cache
def get_y1_lens_bins() -> BinsType:
    """Get the Year 1 lens bins."""
    return {"edges": np.linspace(0.2, 1.2, 5 + 1), "sigma_z": 0.03}


@cache
def get_y1_source_bins() -> BinsType:
    """Get the Year 1 source bins."""
    return {
        "edges": ZDistLSSTSRD.year_1_source().equal_area_bins(5, 0.05, 3.5),
        "sigma_z": 0.05,
    }


@cache
def get_y10_lens_bins() -> BinsType:
    """Get the Year 10 lens bins."""
    return {"edges": np.linspace(0.2, 1.2, 10 + 1), "sigma_z": 0.03}


@cache
def get_y10_source_bins() -> BinsType:
    """Get the Year 10 source bins."""
    return {
        "edges": ZDistLSSTSRD.year_10_source().equal_area_bins(5, 0.05, 3.5),
        "sigma_z": 0.05,
    }


@cache
def get_lsst_y1_lens_harmonic_bin_collection() -> ZDistLSSTSRDBinCollection:
    """Get the LSST Year 1 lens bin collection."""
    y1_lens_bins = get_y1_lens_bins()
    return ZDistLSSTSRDBinCollection(
        alpha=Y1_LENS_ALPHA,
        beta=Y1_LENS_BETA,
        z0=Y1_LENS_Z0,
        bins=[
            ZDistLSSTSRDBin(
                zpl=zpl,
                zpu=zpu,
                sigma_z=y1_lens_bins["sigma_z"],
                z=RawGrid1D(values=[0.0, 3.5]),
                bin_name=f"lens_{zpl:.1f}_{zpu:.1f}_y1",
                measurements={Galaxies.COUNTS},
            )
            for zpl, zpu in pairwise(y1_lens_bins["edges"])
        ],
    )


@cache
def get_lsst_y1_source_harmonic_bin_collection() -> ZDistLSSTSRDBinCollection:
    """Get the LSST Year 1 source bin collection."""
    y1_source_bins = get_y1_source_bins()
    return ZDistLSSTSRDBinCollection(
        alpha=Y1_SOURCE_ALPHA,
        beta=Y1_SOURCE_BETA,
        z0=Y1_SOURCE_Z0,
        bins=[
            ZDistLSSTSRDBin(
                zpl=zpl,
                zpu=zpu,
                sigma_z=y1_source_bins["sigma_z"],
                z=RawGrid1D(values=[0.0, 3.5]),
                bin_name=f"source_{zpl:.1f}_{zpu:.1f}_y1",
                measurements={Galaxies.SHEAR_E},
            )
            for zpl, zpu in pairwise(y1_source_bins["edges"])
        ],
    )


@cache
def get_lsst_y10_lens_harmonic_bin_collection() -> ZDistLSSTSRDBinCollection:
    """Get the LSST Year 10 lens bin collection."""
    y10_lens_bins = get_y10_lens_bins()
    return ZDistLSSTSRDBinCollection(
        alpha=Y10_LENS_ALPHA,
        beta=Y10_LENS_BETA,
        z0=Y10_LENS_Z0,
        bins=[
            ZDistLSSTSRDBin(
                zpl=zpl,
                zpu=zpu,
                sigma_z=y10_lens_bins["sigma_z"],
                z=RawGrid1D(values=[0.0, 3.5]),
                bin_name=f"lens_{zpl:.1f}_{zpu:.1f}_y10",
                measurements={Galaxies.COUNTS},
            )
            for zpl, zpu in pairwise(y10_lens_bins["edges"])
        ],
    )


@cache
def get_lsst_y10_source_harmonic_bin_collection() -> ZDistLSSTSRDBinCollection:
    """Get the LSST Year 10 source bin collection."""
    y10_source_bins = get_y10_source_bins()
    return ZDistLSSTSRDBinCollection(
        alpha=Y10_SOURCE_ALPHA,
        beta=Y10_SOURCE_BETA,
        z0=Y10_SOURCE_Z0,
        bins=[
            ZDistLSSTSRDBin(
                zpl=zpl,
                zpu=zpu,
                sigma_z=y10_source_bins["sigma_z"],
                z=RawGrid1D(values=[0.0, 3.5]),
                bin_name=f"source_{zpl:.1f}_{zpu:.1f}_y10",
                measurements={Galaxies.SHEAR_E},
            )
            for zpl, zpu in pairwise(y10_source_bins["edges"])
        ],
    )


def __getattr__(name: str) -> Any:  # pylint: disable-msg=too-many-return-statements
    """Lazy evaluation of the bins."""
    match name:
        case "Y1_LENS_BINS":
            return get_y1_lens_bins()
        case "Y1_SOURCE_BINS":
            return get_y1_source_bins()
        case "Y10_LENS_BINS":
            return get_y10_lens_bins()
        case "Y10_SOURCE_BINS":
            return get_y10_source_bins()
        case "LSST_Y1_LENS_HARMONIC_BIN_COLLECTION":
            return get_lsst_y1_lens_harmonic_bin_collection()
        case "LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION":
            return get_lsst_y1_source_harmonic_bin_collection()
        case "LSST_Y10_LENS_HARMONIC_BIN_COLLECTION":
            return get_lsst_y10_lens_harmonic_bin_collection()
        case "LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION":
            return get_lsst_y10_source_harmonic_bin_collection()
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


Y1_LENS_BINS: BinsType
Y1_SOURCE_BINS: BinsType
Y10_LENS_BINS: BinsType
Y10_SOURCE_BINS: BinsType
LSST_Y1_LENS_HARMONIC_BIN_COLLECTION: ZDistLSSTSRDBinCollection
LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION: ZDistLSSTSRDBinCollection
LSST_Y10_LENS_HARMONIC_BIN_COLLECTION: ZDistLSSTSRDBinCollection
LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION: ZDistLSSTSRDBinCollection
