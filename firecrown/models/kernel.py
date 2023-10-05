from enum import Enum
import numpy as np
from firecrown.parameters import ParamsMap
from firecrown.updatable import Updatable


class Kernel(Updatable):
    def __init__():
        super().__init__()

    def distribution(self, mass, z, mass_proxy, z_proxy):
        return 1.0


class Completeness(Kernel):
    def __init__():
        super().__init__()

    # TODO get better names here
    def distribution(self, mass, z, mass_proxy, z_proxy):
        a_nc = 1.1321
        b_nc = 0.7751
        a_mc = 13.31
        b_mc = 0.2025
        log_mc = a_mc + b_mc * (1.0 + z)
        nc = a_nc + b_nc * (1.0 + z)
        completeness = (mass / log_mc) ** nc / ((mass / log_mc) ** nc + 1.0)
        return completeness


class Purity(Kernel):
    def __init__():
        super().__init__()

    # TODO get better names here
    def distribution(mass, z, mass_proxy, z_proxy):
        ln_r = np.log(10**mass_proxy)
        a_nc = np.log(10) * 0.8612
        b_nc = np.log(10) * 0.3527
        a_rc = 2.2183
        b_rc = -0.6592
        nc = a_nc + b_nc * (1.0 + z)
        ln_rc = a_rc + b_rc * (1.0 + z)
        purity = (ln_r / ln_rc) ** nc / ((ln_r / ln_rc) ** nc + 1.0)
        return purity


class Miscentering(Kernel):
    pass
