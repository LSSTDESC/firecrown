class BaseLogLike:
    def compute_loglike(self, data, theory, **kwargs):
        """Compute the log-likelihood."""
        raise NotImplementedError(
            "Method `compute_loglike` is not implemented!")
