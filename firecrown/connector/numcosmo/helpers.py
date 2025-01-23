"""Helper functions and classes for the NumCosmo connector."""

from typing_extensions import assert_never
from numcosmo_py import Nc
import firecrown.ccl_factory as fac


def get_hiprim(hi_cosmo: Nc.HICosmo) -> Nc.HIPrimPowerLaw:
    """Return the HIPrim object from a NumCosmo HICosmo object.

    If hi_cosmo does not have a HIPrim object, a ValueError is raised.
    If the HIPrim object is not of type HIPrimPowerLaw, a ValueError is raised.

    :param hi_cosmo: NumCosmo HICosmo object
    :return: the HIPrim object contained in hi_cosmo
    """
    hiprim = hi_cosmo.peek_submodel_by_mid(Nc.HIPrim.id())
    if not hiprim:
        raise ValueError("NumCosmo object must include a HIPrim object.")
    if not isinstance(hiprim, Nc.HIPrimPowerLaw):
        raise ValueError(f"NumCosmo HIPrim object type {type(hiprim)} not supported.")
    return hiprim


def get_amplitude_parameters(
    ccl_factory: fac.CCLFactory,
    p_ml: None | Nc.PowspecML,
    hi_cosmo: Nc.HICosmo,
) -> tuple[float | None, float | None]:
    """
    Calculate the amplitude parameters for CCL.

    :param ccl_factory: the CCL factory object
    :param p_ml: the NumCosmo PowspecML object, or None
    :param hi_cosmo: the NumCosmo cosmology object
    :return: a tuple of the amplitude parameters, (A_s, sigma8), with only one set.
    """
    A_s: float | None = None
    sigma8: float | None = None

    # mypy verifies that the match statement below is exhaustive
    match ccl_factory.amplitude_parameter:
        case fac.PoweSpecAmplitudeParameter.SIGMA8:
            if p_ml is None:
                raise ValueError("PowspecML object must be provided when using sigma8.")
            sigma8 = p_ml.sigma_tophat_R(hi_cosmo, 1.0e-7, 0.0, 8.0 / hi_cosmo.h())
        case fac.PoweSpecAmplitudeParameter.AS:
            A_s = get_hiprim(hi_cosmo).SA_Ampl()
        case _ as unreachable:
            assert_never(unreachable)
    assert A_s is not None or sigma8 is not None
    return A_s, sigma8
