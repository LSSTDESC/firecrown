"""The mixins in this file define the firecrown likelihood API.

Notes:
 - Each class which inherits from a given mixin is expected to define any
   methods defined in the mixin with the same call signature.
 - If the mixin includes a class-level doc string, then the `__init__` function
   should define at least those arguments and/or keyword arguments.
"""


class SourceMixin(object):
    """The source mixin.

    Parameters
    ----------
    systematics : list of str
        A list of the source-level systematics to apply to the source.
    """
    def render(self, cosmo, params):
        """Render a source by applying systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.

        Returns
        -------
        self
        """
        raise NotImplementedError(
            "Method `render` is not implemented!")


class LogLikeMixin(object):
    """The log-likelihood mixin.

    Parameters
    ----------
    data_vector : list of str
        A list of the statistics in the config file in the order they appear in
        the covariance matrix.
    """
    def compute_loglike(self, data, theory, **kwargs):
        """Compute the log-likelihood.

        Parameters
        ----------
        data : dict of arrays
            A dictionary mapping the names of the statistics to their
            values in the data.
        theory : dict of arrays
            A dictionary mapping the names of the statistics to their
            predictions.
        **kwargs : extra keyword arguments
            Any extra keyword arguments can be used by subclasses.

        Returns
        -------
        loglike : float
            The log-likelihood.
        """
        raise NotImplementedError(
            "Method `compute_loglike` is not implemented!")
