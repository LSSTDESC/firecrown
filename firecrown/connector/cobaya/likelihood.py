import math
import numpy as np
import importlib
import importlib.util
import os

from pprint import pprint

import firecrown
from ...parser_constants import FIRECROWN_RESERVED_NAMES

from cobaya.likelihood import Likelihood

from firecrown.descriptors import TypeLikelihood

class LikelihoodConnector(Likelihood):
    """
    A class implementing cobaya.likelihood.Likelihood ...

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

    firecrownIni = None

    def initialize(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        
        filename, file_extension = os.path.splitext(self.firecrownIni)        
        
        ext = file_extension.lower ()
        
        if ext == '.yaml':
            self.config, self.data = firecrown.parse(self.firecrownIni)
            
            analyses = list(set(list(self.data.keys())) - set(FIRECROWN_RESERVED_NAMES) - set(["priors"]))
            
            if len (analyses) != 1:
                raise ValueError("Only a single likelihood per file is supported")
            
            for analysis in analyses:
                self.likelihood = self.data[analysis]['data']['likelihood']
                self.likelihood.set_sources (self.data[analysis]['data']['sources'])
                self.likelihood.set_systematics (self.data[analysis]['data']['systematics'])
                self.likelihood.set_statistics (self.data[analysis]['data']['statistics'])

        elif ext == '.py':            
            inifile = os.path.basename(self.firecrownIni)
            inipath = os.path.dirname(self.firecrownIni)
            modname, _ = os.path.splitext(inifile)
        
            spec = importlib.util.spec_from_file_location(modname, self.firecrownIni)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            if not hasattr(mod, 'likelihood'):
                raise ValueError(f"Firecrown initialization file {self.firecrownIni} does not define a likelihood.")

            self.likelihood = mod.likelihood
        else:
            raise ValueError(f"Unrecognized Firecrown initialization file {self.firecrownIni}.")

    def get_param(self, p):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return self._current_state["derived"][p]

    def initialize_with_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def initialize_with_provider(self, provider):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        self.provider = provider

    def get_can_provide_params(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return []

    def get_allow_agnostic(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        return False

    def get_requirements(self):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        likelihood_requires = {"ccl": None}
        for param_name in self.likelihood.get_params_names ():
            likelihood_requires[param_name] = None

        return likelihood_requires

    def must_provide(self, **requirements):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        pass

    def logp(self, **params_values):
        """...
        ...
        Parameters
        ----------
        ... : str
            ...
        """
        ccl = self.provider.get_ccl()
        #loglikes, _, _, _, _, _ = firecrown.compute_loglike(cosmo=ccl, data=self.data)
        #return np.sum([v for v in loglikes.values() if v is not None])
        
        return self.likelihood.compute_loglike (ccl, params_values)
