import numpy as np


class RMProxy:
    def __init__(self):
        self.pivot_mass = 104516977892721.36
        self.pivot_redshift = 0.7259996508853891
        self.log_pivot_mass = np.log(self.pivot_mass)
        self.log_1_p_pivot_redshift = np.log(1.0 + self.pivot_redshift)
        self.likelihood_parameters = None

    def compute_theory_vector(self, lnN, logm, z):
        mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2 = self.likelihood_parameters
        lnN = np.array(lnN)
        lnm = np.log(10.0**logm)
        x_mu = (
            mu_p0
            + mu_p1 * (lnm - self.log_pivot_mass)
            + mu_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        x = lnN - x_mu
        sigma = (
            sigma_p0
            + sigma_p1 * (lnm - self.log_pivot_mass)
            + sigma_p2 * (np.log(1.0 + z) - self.log_1_p_pivot_redshift)
        )
        return [x, sigma]

    def mass_proxy_likelihood(
        self,
        lnN,
        logm,
        z,
        mu_p0=1.0,
        mu_p1=0.86,
        mu_p2=-1.21,
        sigma_p0=2.32,
        sigma_p1=-3.08,
        sigma_p2=2.03,
    ):
        self.likelihood_parameters = [mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2]
        x, sigma = self.compute_theory_vector(
            lnN,
            logm,
            z,
        )
        chisq = np.dot(x, x) / (2 * sigma**2)
        lk = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-chisq)
        return lk
