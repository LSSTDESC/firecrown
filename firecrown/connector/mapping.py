"""The module mapping provides facilities for mapping the cosmological constants
and functions used by one body of code to another. This is done by defining one 
of the codes (pyccl) as being the 'standard', and for all other supported code, 
providing functions from_ccl and to_ccl.

The supported codes include:
    pyccl
    CAMB (Fortran convention)
    pycamb (similar but not identical to the CAMB Fortran convention)
    CLASS
    Cobaya
    CosmoSIS
"""
import numpy as np

from abc import ABC, abstractmethod
from ..descriptors import Float, String


class Mapping(ABC):

    Omega_c = Float(minvalue=0.0, maxvalue=1.0)
    Omega_b = Float(minvalue=0.0, maxvalue=1.0)
    h = Float(minvalue=0.3, maxvalue=1.2)
    A_s = Float(allow_none=True)
    sigma8 = Float(allow_none=True)
    n_s = Float()
    Omega_k = Float(minvalue=-1.0, maxvalue=1.0)
    Neff = Float(minvalue=0.0)
    m_nu = Float(minvalue=0.0)
    m_nu_type = String()
    w0 = Float()
    wa = Float()
    T_CMB = Float()

    """
    Mapping is an abstract base class providing the interface that describes
    a mapping of cosmological constants from some concrete Boltzmann calculator
    to the form those constants take in CCL. Each supported Boltzmann calculator
    will have its own concrete subclass.

    ...

    Attributes
    ----------
    ... : str
        ...

    Methods
    -------
    ...(...)
        ....
    """

    def __init__(self, **kwargs):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """

        super().__init__()

    @abstractmethod
    def get_params_names(self):
        """Return the names of the cosmological parameters that this
        mapping is expected to deliver.
        """
        pass


    @abstractmethod
    def transform_k_h_to_k(self, k_h):
        """Transform the given k_h (k over h) to k.
        """
        pass


    @abstractmethod
    def transform_p_k_h3_to_p_k(self, p_k_h3):
        """Transform the given p_k * h^3 to p_k.
        """
        pass


    def set_params(
        self,
        *,
        Omega_c: float,
        Omega_b: float,
        h: float,
        A_s: float = None,
        sigma8: float = None,
        n_s: float,
        Omega_k: float,
        Neff: float,
        m_nu: float,
        m_nu_type: str,
        w0: float,
        wa: float,
        T_CMB: float,
    ):
        """Sets the cosmological constants suitable for use in constructing a pyccl.core.CosmologyCalculator.
        See the documentation of that class for an explanation of the choices and meanings of default values
        of None.
        ...
        Parameters
        ----------
        ... : str
            ...
        """

        # Typecheck is done automatically using the descriptorsa and is done to avoid very confusing error
        # messages at a later time in case of error.
        self.Omega_c = Omega_c
        self.Omega_b = Omega_b
        self.h = h

        if A_s is not None and sigma8 is not None:
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
        self.m_nu_type = m_nu_type
        self.w0 = w0
        self.wa = wa
        self.T_CMB = T_CMB

    def redshift_to_scale_factor(self, z):
        """Given arrays of redshift returns an array of scale factor with the inverse
        order."""

        scale = np.flip(1.0 / (1.0 + z))
        return scale

    def redshift_to_scale_factor_p_k(self, p_k):
        """Given an 2d arrays power spectrum ordered by (redshift, mode)
        return a 2d array with the rows flipped to match the reorderning
        from redshift to scale factor."""

        p_k_out = np.flipud(p_k)
        return p_k_out

    def asdict(self):
        return {
            "Omega_c": self.Omega_c,
            "Omega_b": self.Omega_b,
            "h": self.h,
            "A_s": self.A_s,
            "sigma8": self.sigma8,
            "n_s": self.n_s,
            "Omega_k": self.Omega_k,
            "Omega_g": self.Omega_g,
            "Neff": self.Neff,
            "m_nu": self.m_nu,
            "m_nu_type": self.m_nu_type,
            "w0": self.w0,
            "wa": self.wa,
            "T_CMB": self.T_CMB,
        }

    def get_H0(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self.h * 100.0



class MappingCLASS(Mapping):
    """
    This class is not yet implemented; this stub is here to satisfy IDEs that
    complain about using the names of missing classes.
    """

    pass


class MappingCosmoSIS(Mapping):
    """
    Implementation of the mapping class between CosmoSIS datablock parameters
    and CCL.
    """
     
    def get_params_names(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
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


    def transform_k_h_to_k(self, k_h):
        return k_h * self.h


    def transform_p_k_h3_to_p_k(self, p_k_h3):
        return p_k_h3 / (self.h ** 3)

    
    def set_params_from_cosmosis(self, cosmosis_params: dict):
        """Return a PyCCLCosmologyConstants object with parameters equivalent to
        those read from CosmoSIS when using CAMB."""
        # TODO: Verify that CosmoSIS/CAMB does not use Omega_g
        # TODO: Verify that CosmoSIS/CAMB uses delta_neff, not N_eff
        h = cosmosis_params["h0"]  # Not 'hubble' !
        Omega_b = cosmosis_params["omega_b"]
        Omega_c = cosmosis_params["omega_c"]
        sigma8 = cosmosis_params["sigma_8"]
        n_s = cosmosis_params["n_s"]
        Omega_k = cosmosis_params["omega_k"]
        # Read omega_nu from CosmoSIS (in newer CosmoSIS)
        # Read m_nu from CosmoSIS (in newer CosmoSIS)
        Neff = cosmosis_params.get("delta_neff", 0.0) + 3.046  # Verify this with Joe
        m_nu = cosmosis_params["omega_nu"] * h * h * 93.14
        m_nu_type = "normal"
        w0 = cosmosis_params["w"]
        wa = cosmosis_params["wa"]

        self.set_params(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            sigma8=sigma8,
            n_s=n_s,
            Omega_k=Omega_k,
            Neff=Neff,
            m_nu=m_nu,
            m_nu_type=m_nu_type,
            w0=w0,
            wa=-wa,  # Is this minus sign here correct?
            T_CMB=2.7255,  # Modify CosmoSIS to make this available in the datablock
        )


class MappingCAMB(Mapping):
    """
    A class implementing Mapping ...

    ...

    Attributes
    ----------
    ... : str
        ...

    Methods
    -------
    ...(...)
        ....
    """

    def __init__(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def get_params_names(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
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

    def set_params_from_camb(self, **params_values):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """

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
        # pprint (params_values)

        m_nu_type = "normal"
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
            m_nu_type=m_nu_type,
            w0=w,
            wa=wa,
            Neff=Neff,
            T_CMB=2.7255,  # Can we make cobaya set this?
        )


mapping_classes = {
    "CAMB": MappingCAMB,
    "CLASS": MappingCLASS,
    "CosmoSIS": MappingCosmoSIS,
}


def mapping_builder(*, input_style, **kwargs):
    if not input_style in mapping_classes.keys():
        raise ValueError(f"input_style must be {*mapping_classes,}, not {input_style}")

    return mapping_classes[input_style](**kwargs)
