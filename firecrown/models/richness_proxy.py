"""An abstract class for the richness proxy for the number counts likelihood.
"""


import numpy as np


class RMProxy:
    """A class for the richness proxy for the number counts likelihood."""

    def __init__(self):
        self.pivot_mass = 4.2253521e14
        self.pivot_redshift = 0.6
        self.log_pivot_mass = np.log(self.pivot_mass)
        self.log_1_p_pivot_redshift = np.log(1.0 + self.pivot_redshift)
        self.likelihood_parameters = None

    # pylint: disable-next=invalid-name
    def compute_theory_vector(self, logN, logm, z):
        """Compute the theory vector for the richness proxy likelihood."""
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2 = self.likelihood_parameters
        lnN = np.log(10**logN)
        lnm = np.log(10.0**logm)
        x_mu = (
            mu_p0
            + mu_p1 * (lnm - self.log_pivot_mass)
            + mu_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        x = lnN - x_mu  # pylint: disable=invalid-name
        sigma = (
            sigma_p0
            + sigma_p1 * (lnm - self.log_pivot_mass)
            + sigma_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        return [x, sigma]

    def mass_proxy_likelihood(
        self,
        logN,  # pylint: disable=invalid-name
        logm,
        z,  # pylint: disable=invalid-name
        mu_p0=1.0,
        mu_p1=0.86,
        mu_p2=-1.21,
        sigma_p0=2.32,
        sigma_p1=-3.08,
        sigma_p2=2.03,
    ):
        """Compute the likelihood for the richness proxy likelihood."""
        self.likelihood_parameters = [mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2]
        # pylint: disable-next=invalid-name
        x, sigma = self.compute_theory_vector(
            logN,
            logm,
            z,
        )
        chisq = np.dot(x, x) / (2 * sigma**2)
        # pylint: disable-next=invalid-name
        lnnorm = np.log(np.sqrt(2.0 * np.pi * sigma**2))
        lk = np.exp(-chisq - lnnorm)
        return lk
