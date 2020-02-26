from ..core import Systematic

__all__ = [
    'IdentityFunctionMOR',
    'TopHatSelectionFunction',
]


class IdentityFunctionMOR(Systematic):
    """An identity function mass-observable relationship.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self):
        pass

    def apply(self, cosmo, params, source):
        """Apply this MOR to the source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        source.mor_ = self._gen_mor()
        source.inv_mor_ = self._gen_mor()

    def _gen_mor():
        def _mor(lnmass, a):
            return lnmass

        return _mor


class TopHatSelectionFunction(Systematic):
    """Top-hat selection function for clusters.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self):
        pass

    def apply(self, cosmo, params, source):
        """Apply this selection function to the source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        source.selfunc_ = self._gen_selection_function(source)

    def _gen_selection_function(self, source):

        def _selfunc(lnmass, a):
            lnmass_min = source.inv_mor_(source.lnlam_min_, a)
            lnmass_max = source.inv_mor_(source.lnlam_max_, a)
            if lnmass_max < lnmass_min:
                lnmass_min, lnmass_max = lnmass_max, lnmass_min

            if lnmass_min <= lnmass and lnmass <= lnmass_max:
                return 1.0
            else:
                return 0.0

        return _selfunc
