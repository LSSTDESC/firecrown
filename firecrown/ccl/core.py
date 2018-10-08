"""The classes in this file define the firecrown-CCL API.

Notes:
 - Each subclass which inherits from a given class is expected to define any
   methods defined in the mixin with the same call signature.
 - If the class nelow  includes a class-level doc string, then
   the `__init__` function of the subclass should define at least those
   arguments and/or keyword arguments.
 - Attributed ending with an underscore are set after the call to
   `apply`/`compute`/`render`.
 - Attributes define in the `__init__` method should be considered constant
   and not changed after instantiation.
 - Objects inheriting from `Systematic` should only adjust
   source/statistic properties ending with an underscore.
"""


class Statistic(object):
    """The statistic mixin.

    Parameters
    ----------
    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    """
    def compute(self, cosmo, params, sources):
        """Compute a statistic from sources.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        sources : dict
            A dictionary mapping sources to their objects. The sources must
            already have been rendered by calling `render` on them.
        """
        raise NotImplementedError(
            "Method `compute` is not implemented!")


class Systematic(object):
    """The systematic mixin."""
    def apply(self, cosmo, params, source_or_statistic):
        """Apply systematics to a source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source_or_statistic : a source or statistic object
            The source or statistic to which apply systematics.
        """
        raise NotImplementedError(
            "Method `apply` is not implemented!")


class Source(object):
    """The source mixin.

    Parameters
    ----------
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.
    """
    def render(self, cosmo, params):
        """Render a source by applying systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        systematics : dict
            A dictionary mapping systematic names to their objects
        """
        raise NotImplementedError(
            "Method `render` is not implemented!")


class LogLike(object):
    """The log-likelihood mixin.

    Parameters
    ----------
    data_vector : list of str
        A list of the statistics in the config file in the order they appear in
        the covariance matrix.
    """
    def compute(self, data, theory, **kwargs):
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
