from abc import ABC, abstractmethod

from pprint import pprint


def firecrown_convert_builder(input_style, **kwargs):
    if input_style == "CAMB":
        return FirecrownConvertCAMB(**kwargs)
    elif input_style == "CLASS":
        return FirecrownConvertCLASS(**kwargs)
    else:
        raise ValueError(f'style must be "CAMB" or "CLASS", not {style}')


class FirecrownConvert(ABC):
    """
    A class ...

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
    def get_names(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    @abstractmethod
    def set_params(self, **params_values):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    @abstractmethod
    def get_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    @abstractmethod
    def get_H0(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass


class FirecrownConvertCAMB(FirecrownConvert):
    """
    A class implementing FirecrownConvert ...

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
        self.params_values = None

    def get_names(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return ["H0", "ombh2", "omch2", "mnu", "nnu", "tau", "YHe", "As", "ns"]

    def set_params(self, **params_values):
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

        # pprint (params_values)

        h0 = H0 / 100.0
        h02 = h0 * h0
        Omega_b0 = ombh2 / h02
        Omega_c0 = omch2 / h02

        # Here we have the following problem, some parameters used by CAMB
        # are implicit, i.e., since they are not explicitly set the default
        # ones are used. Thus, for instance, here we do not know which type of
        # neutrino hierarchy is used. Reading cobaya interface to CAMB
        # I noticed that it does not touch the neutrino variables and consequently
        # in that case, we could assume that the defaults are being used.
        # Nevertheless, we need a better solution.

        self.params_values = {
            "Omega_c": Omega_c0,
            "Omega_b": Omega_b0,
            "h": h0,
            "n_s": ns,
            "A_s": As,
            "m_nu": m_nu,
        }

    def get_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self.params_values

    def get_H0(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self.params_values["h"] * 100.0
