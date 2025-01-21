"""Facilities for mapping cosmological parameters between frameworks.

The module :mod:`mapping` provides facilities for mapping the cosmological
constants and functions used by one body of code to another.

Each supported body of code has its own dedicated class.
"""

import typing
import warnings
from abc import ABC
from typing import Type, final

import cosmosis.datablock
import numpy as np
import numpy.typing as npt
from pyccl import physical_constants as physics

from firecrown.descriptors import TypeFloat, TypeString
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.ccl_factory import CCLCalculatorArgs, PowerSpec, Background


def build_ccl_background_dict(
    *,
    a: npt.NDArray[np.float64],
    chi: npt.NDArray[np.float64],
    h_over_h0: npt.NDArray[np.float64],
) -> Background:
    """Builds the CCL dictionary of background quantities.

    :param a: The scale factor array
    :param chi: The comoving distance array
    :param h_over_h0: The Hubble parameter divided by H0
    :return: the dictionary of background quantities
    """
    return {"a": a, "chi": chi, "h_over_h0": h_over_h0}


class Mapping(ABC):
    """Abstract base class defining the interface for each supported code.

    The interface describes a mapping of cosmological constants from some
    concrete Boltzmann calculator to the form those constants take in CCL. Each
    supported Boltzmann calculator must have its own concrete subclass.

    The class attributes are all :mod:`firecrown.connector.descriptors`. This is
    to control the allowed types for the instance attributes. A descriptor of
    name 'x' will provide an apparent attribue with name `x` in
    each object, as well as an entry `_x` in the object's `__dict__`.

    """

    # pylint: disable-msg=too-many-instance-attributes
    Omega_c = TypeFloat(minvalue=0.0, maxvalue=1.0)
    Omega_b = TypeFloat(minvalue=0.0, maxvalue=1.0)
    Omega_g = TypeFloat(allow_none=True)
    h = TypeFloat(minvalue=0.3, maxvalue=1.2)
    A_s = TypeFloat(allow_none=True)
    sigma8 = TypeFloat(allow_none=True)
    n_s = TypeFloat(allow_none=True)
    Omega_k = TypeFloat(minvalue=-1.0, maxvalue=1.0)
    Neff = TypeFloat(minvalue=0.0)
    # m_nu = TypeFloat(minvalue=0.0)
    m_nu_type = TypeString()  # "inverted", "normal" or "list"
    w0 = TypeFloat()
    wa = TypeFloat()
    T_CMB = TypeFloat()

    def __init__(self) -> None:
        """Initialize the Mapping object."""
        # We can have:
        #    a single neutrino mass (must be non-negative)
        #    a list of 3 neutrino masses (all must be non-negative)
        #    None, indicating that all neutrinos are massless
        self.m_nu: float | list[float] | None = None

    def get_params_names(self) -> list[str]:
        """Return the names of the expected cosmological parameters for this mapping."""
        warnings.warn(
            "This method is implementation specific and should only be "
            "implemented on the appropriated subclasses. This method"
            "is going to be removed in the next major release.",
            category=DeprecationWarning,
        )
        return []

    def transform_k_h_to_k(self, k_h):
        """Transform the given k_h (k over h) to k.

        :param k_h: the array of wavenumber/h to be transformed
        :return: the transformed array
        """
        assert k_h is not None  # use assertion to silence pylint warning
        warnings.warn(
            "This method is implementation specific and should only be "
            "implemented in the appropriat subclasses. This method is going to"
            "be removed in the next major release.",
            category=DeprecationWarning,
        )

    def transform_p_k_h3_to_p_k(self, p_k_h3):
        r"""Transform the given :math:`p_k h^3 \to p_k`.

        :param p_k_h3: the array of :math:`p_k h^3` to be transformed
        """
        assert p_k_h3 is not None  # use assertion to silence pylint warning
        warnings.warn(
            "This method is implementation specific and should only be "
            "implemented on the appropriated subclasses. This method"
            "is going to be removed in the next major release.",
            category=DeprecationWarning,
        )

    def transform_h_to_h_over_h0(self, h):
        """Transform distances h to :math:`h/h_0`.

        :param h: the array of distances to be transformed
        :return: the transformed array
        """
        assert h is not None  # use assertion to silence pylint warning
        warnings.warn(
            "This method is implementation specific and should only be "
            "implemented on the appropriated subclasses. This method"
            "is going to be removed in the next major release.",
            category=DeprecationWarning,
        )

    @final
    def set_params(
        self,
        *,
        Omega_c: float,
        Omega_b: float,
        h: float,
        A_s: None | float = None,
        sigma8: None | float = None,
        n_s: float,
        Omega_k: float,
        Neff: float,
        m_nu: float | list[float] | None,
        w0: float,
        wa: float,
        T_CMB: float,
    ):
        """Sets the cosmological constants suitable a pyccl.core.CosmologyCalculator.

        See the documentation of that class for an explanation of the choices and
        meanings of default values of None.

        :param Omega_c: fraction of cold dark matter
        :param Omega_b: fraction of baryons
        :param h: dimensionless Hubble parameter;  h = H0/(100 km/s/Mpc)
        :param A_s: amplitude of the primordial power spectrum
        :param sigma8: s.d. of the matter contrast on a scale of (8 Mpc)/h
        :param n_s: scalar spectral index of primordial power spectrum
        :param Omega_k: curvature of the universe
        :param Neff: effective number of relativistic neutrino species
        :param m_nu: effective mass of neutrinos; None for massless neutrinos
        :param w0: constant of the CPL parameterization of the dark energy
            equation of state
        :param wa: linear coefficient of the CPL parameterization of the
            dark energy equation of state
        :param T_CMB: cosmic microwave background temperature today
        """
        # Typecheck is done automatically using the descriptors and is done to
        # avoid void very confusing error messages at a later time in case of
        # error.
        self.Omega_c = Omega_c
        self.Omega_b = Omega_b
        self.h = h

        if (A_s is not None) == (sigma8 is not None):
            raise ValueError("Exactly one of A_s and sigma8 must be supplied")
        if sigma8 is None:
            self.A_s = A_s
            self.sigma8 = None
        else:
            self.A_s = None
            self.sigma8 = sigma8

        self.n_s = n_s
        self.Omega_k = Omega_k
        self.Omega_g = None
        self.Neff = Neff
        self.m_nu = m_nu
        self.w0 = w0
        self.wa = wa
        self.T_CMB = T_CMB

    @staticmethod
    def redshift_to_scale_factor(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Converts redshift to scale factor.

        Given arrays of redshift returns an array of scale factor with the
        inverse order.

        :param z: array of redshifts
        :return: array of scale factors
        """
        scale = np.flip(1.0 / (1.0 + z))
        return scale

    @staticmethod
    def redshift_to_scale_factor_p_k(
        p_k: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Converts power spectrum from redshift to scale factor.

        Given an 2d arrays power spectrum ordered by (redshift, mode)
        return a 2d array with the rows flipped to match the reordering
        from redshift to scale factor.

        :param p_k: a power spectrum, ordered by (redshift, mode)
        :return: array of power spectrum, ordered by (scale factor, mode)
        """
        p_k_out = np.flipud(p_k)
        return p_k_out

    def asdict(self) -> dict[str, float | list[float]]:
        """Return a dictionary containing the cosmological constants.

        :return: the dictionary, containing keys:

            - ``Omega_c``: fraction of cold dark matter
            - ``Omega_b``: fraction of baryons
            - ``h``: dimensionless Hubble parameter;  h = H0/(100 km/s/Mpc)
            - ``A_s``: amplitude of the primordial power spectrum
            - ``sigma8``: s.d. of the matter contrast on a scale of (8 Mpc)/h
            - ``n_s``: scalar spectral index of primordial power spectrum
            - ``Omega_k``: curvature of the universe
            - ``Neff``: effective number of relativistic neutrino species
            - ``m_nu``: effective mass of neutrinos
            - ``w0``: constant of the CPL parameterization of the dark energy
                equation of state
            - ``wa``: linear coefficient of the CPL parameterization of the
                dark energy equation of state
            - ``T_CMB``: cosmic microwave background temperature today
        """
        cosmo_dict: dict[str, float | list[float]] = {
            "Omega_c": self.Omega_c,
            "Omega_b": self.Omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "Omega_k": self.Omega_k,
            "Neff": self.Neff,
            "w0": self.w0,
            "wa": self.wa,
            "T_CMB": self.T_CMB,
        }
        if self.A_s is not None:
            cosmo_dict["A_s"] = self.A_s
        if self.sigma8 is not None:
            cosmo_dict["sigma8"] = self.sigma8
        # Currently we do not support Omega_g
        # if self.Omega_g is not None:
        #    cosmo_dict["Omega_g"] = self.Omega_g
        if self.m_nu is None:
            cosmo_dict["m_nu"] = 0.0
        else:
            cosmo_dict["m_nu"] = self.m_nu

        return cosmo_dict

    def get_H0(self) -> float:
        """Return the value of H0.

        :return: H0 in km/s/Mpc
        """
        return self.h * 100.0


class MappingCLASS(Mapping):
    """This class is not yet implemented.

    This stub is here to satisfy IDEs that complain about using the names of
    missing classes.
    """


class MappingCosmoSIS(Mapping):
    """Mapping support for CosmoSIS."""

    def get_params_names(self) -> list[str]:
        """Return the names of the expected cosmological parameters for this mapping.

        :return: a list of the cosmological parameter names
        """
        return [
            "h0",
            "omega_b",
            "omega_c",
            "sigma_8",
            "n_s",
            "omega_k",
            "delta_neff",
            "omega_nu",
            "w",
            "wa",
        ]

    def transform_k_h_to_k(
        self, k_h: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transform the given k_h (k over h) to k.

        :param k_h: the array of wavenumber/h to be transformeed
        :return: the transformed array
        """
        return np.array(k_h * self.h, dtype=np.float64)

    def transform_p_k_h3_to_p_k(
        self, p_k_h3: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Transform the given :math:`p_k h^3 \to p_k`.

        :param p_k_h3: the array of :math:`p_k h^3` to be transformed
        :return: the transformed array
        """
        return np.array(p_k_h3 / (self.h**3), dtype=np.float64)

    def transform_h_to_h_over_h0(
        self, h: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Transform distances h to :math:`h/h_0`.

        :param h: the array of distances to be transformed
        :return: the transformed distances
        """
        hubble_radius_today = physics.CLIGHT * 1e-5 / self.h
        return np.flip(h) * hubble_radius_today

    def set_params_from_cosmosis(self, cosmosis_params: NamedParameters) -> None:
        """Set the parameters in this mapping from the given CosmoSIS parameters.

        :param cosmosis_params: the cosmological parameters read from CosmoSIS
        """
        # TODO: Verify that CosmoSIS/CAMB does not use Omega_g
        # TODO: Verify that CosmoSIS/CAMB uses delta_neff, not N_eff

        h = cosmosis_params.get_float("h0")
        Omega_b = cosmosis_params.get_float("omega_b")
        Omega_c = cosmosis_params.get_float("omega_c")
        sigma8 = cosmosis_params.get_float("sigma_8", 0.8)
        n_s = cosmosis_params.get_float("n_s", 0.96)
        Omega_k = cosmosis_params.get_float("omega_k")
        # Read omega_nu from CosmoSIS (in newer CosmoSIS)
        # Read m_nu from CosmoSIS (in newer CosmoSIS)
        delta_neff = cosmosis_params.get_float("delta_neff", 0.0)
        Neff = delta_neff + 3.046
        omega_nu = cosmosis_params.get_float("omega_nu")
        m_nu = omega_nu * h * h * 93.14
        w0 = cosmosis_params.get_float("w")
        wa = cosmosis_params.get_float("wa")

        # pylint: disable=duplicate-code
        self.set_params(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            sigma8=sigma8,
            n_s=n_s,
            Omega_k=Omega_k,
            Neff=Neff,
            m_nu=m_nu,
            w0=w0,
            wa=-wa,  # Is this minus sign here correct?
            T_CMB=2.7255,
            # Modify CosmoSIS to make this available in the datablock
        )
        # pylint: enable=duplicate-code

    def calculate_ccl_args(self, sample: cosmosis.datablock) -> CCLCalculatorArgs:
        """Calculate the arguments necessary for CCL for this sample.

        :param sample: the datablock for the current sample
        :return: the arguments required by CCL
        """
        pk_linear: None | PowerSpec = None
        pk_nonlin: None | PowerSpec = None
        if sample.has_section("matter_power_lin"):
            k = self.transform_k_h_to_k(sample["matter_power_lin", "k_h"])
            z_mpl = sample["matter_power_lin", "z"]
            scale_mpl = self.redshift_to_scale_factor(z_mpl)
            p_k = self.transform_p_k_h3_to_p_k(sample["matter_power_lin", "p_k"])
            p_k = self.redshift_to_scale_factor_p_k(p_k)

            pk_linear = {
                "a": scale_mpl,
                "k": k,
                "delta_matter:delta_matter": p_k,
            }

        if sample.has_section("matter_power_nl"):
            k = self.transform_k_h_to_k(sample["matter_power_nl", "k_h"])
            z_mpl = sample["matter_power_nl", "z"]
            scale_mpl = self.redshift_to_scale_factor(z_mpl)
            p_k = self.transform_p_k_h3_to_p_k(sample["matter_power_nl", "p_k"])
            p_k = self.redshift_to_scale_factor_p_k(p_k)

            pk_nonlin = {
                "a": scale_mpl,
                "k": k,
                "delta_matter:delta_matter": p_k,
            }

        # TODO: We should have several configurable modes for this module.
        # In all cases, an exception will be raised (causing a program
        # shutdown) if something that is to be read from the DataBlock is not
        # present in the DataBlock.
        #
        # background: read only background information from the DataBlock; it
        # will generate a runtime error if the configured likelihood attempts
        # to use anything else.
        #
        # linear: read also the linear power spectrum from the DataBlock. Any
        # non-linear power spectrum present will be ignored. It will generate
        # a runtime error if the configured likelihood attempts to make use
        # of a non-linear spectrum.
        #
        #  nonlinear: read also the nonlinear power spectrum from the DataBlock.
        #
        # halofit, halomodel, emu: use CCL to calculate the nonlinear power
        # spectrum according to the named technique. In all cases, the linear
        # power spectrum read from the DataBlock is used as input. In all
        # cases, it is an error if the DataBlock also contains a nonlinear
        # power spectrum.

        chi = np.flip(sample["distances", "d_m"])
        scale_distances = self.redshift_to_scale_factor(sample["distances", "z"])
        # h0 = sample["cosmological_parameters", "h0"]
        # NOTE: The first value of the h_over_h0 array is non-zero because of the way
        # CAMB does it calculation. We do not modify this, because we want consistency.

        # hubble_radius_today = (ccl.physical_constants.CLIGHT * 1e-5) / h0
        # h_over_h0 = np.flip(sample["distances", "h"]) * hubble_radius_today
        h_over_h0 = self.transform_h_to_h_over_h0(sample["distances", "h"])

        background: Background = build_ccl_background_dict(
            a=scale_distances, chi=chi, h_over_h0=h_over_h0
        )
        ccl_args: CCLCalculatorArgs = {"background": background}

        if pk_linear is not None:
            ccl_args.update({"pk_linear": pk_linear})
        if pk_nonlin is not None:
            ccl_args.update({"pk_nonlin": pk_nonlin})

        return ccl_args


class MappingCAMB(Mapping):
    """Mapping support for PyCAMB, the Python version of CAMB.

    Note that the Python version of CAMB uses some different conventions from
    the Fortran version of CAMB. The two are not interchangeable.
    """

    def get_params_names(self) -> list[str]:
        """Return the names of the expected cosmological parameters for this mapping.

        :return: a list of the cosmological parameter names

        """
        return [
            "H0",
            "ombh2",
            "omch2",
            "mnu",
            "nnu",
            "tau",
            "YHe",
            "As",
            "ns",
            "w",
            "wa",
        ]

    def set_params_from_camb(self, **params_values) -> None:
        """Set the parameters in this mapping from the given CAMB-style parameters."""
        # pylint: disable-msg=R0914

        # CAMB can use different parameters in place of H0, we must deal with this
        # possibility here.

        H0 = params_values["H0"]
        As = params_values["As"]
        ns = params_values["ns"]
        ombh2 = params_values["ombh2"]
        omch2 = params_values["omch2"]
        Neff = params_values["nnu"]
        m_nu = params_values["mnu"]
        Omega_k0 = params_values["omk"]

        h0 = H0 / 100.0
        h02 = h0 * h0
        Omega_b0 = ombh2 / h02
        Omega_c0 = omch2 / h02

        w = params_values.get("w", -1.0)
        wa = params_values.get("wa", 0.0)

        # Here we have the following problem, some parameters used by CAMB
        # are implicit, i.e., since they are not explicitly set the default
        # ones are used. Thus, for instance, here we do not know which type of
        # neutrino hierarchy is used. Reading cobaya interface to CAMB
        # I noticed that it does not touch the neutrino variables and consequently
        # in that case, we could assume that the defaults are being used.
        # Nevertheless, we need a better solution.

        self.set_params(
            Omega_c=Omega_c0,
            Omega_b=Omega_b0,
            h=h0,
            n_s=ns,
            Omega_k=Omega_k0,
            sigma8=None,
            A_s=As,
            m_nu=m_nu,
            w0=w,
            wa=wa,
            Neff=Neff,
            T_CMB=2.7255,  # Can we make cobaya set this?
        )


mapping_classes: typing.Mapping[str, Type[Mapping]] = {
    "CAMB": MappingCAMB,
    "CLASS": MappingCLASS,
    "CosmoSIS": MappingCosmoSIS,
}


def mapping_builder(*, input_style: str, **kwargs) -> Mapping:
    """Return the Mapping class for the given input_style.

    If input_style is not recognized raise an exception.

    :param input_style: the name of the mapping
    :param kwargs: the parameters of the mapping
    :return: the mapping object
    """
    if input_style not in mapping_classes:
        raise ValueError(f"input_style must be {*mapping_classes, }, not {input_style}")

    return mapping_classes[input_style](**kwargs)
