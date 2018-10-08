from ..core import SystematicMixin

__all__ = ['MultiplicativeShearBias']


class MultiplicativeShearBias(SystematicMixin):
    """Multiplicative shear bias systematic.

    This systematic adjusts the `scale_` of a source by `(1 + m)`.

    Parameters
    ----------
    m : str
        The name of the multiplicative bias parameter.

    Methods
    -------
    apply : appaly the systematic to a source
    """
    def __init__(self, m):
        self.m = m

    def apply(self, cosmo, params, source):
        """Apply multiplicative shear bias to a source. The `scale_` of the
        source is multiplied by `(1 + m)`.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : a source object
            The source to which apply the shear bias.
        """
        source.scale_ *= (1.0 + params[self.m])


# class LinearAlignment(SourceSystematic):
#     modified_source_properties = ['ia_amplitude', 'f_red']
#     required_source_properties = ['z']
#     params = ['biasia']
#     optional_params = {
#         'alphaz': 0.0,
#         'z_piv': 0.0,
#         'fred': 1.0,
#         'alphag': 0.0
#     }
#
#     def adjust_source(self, cosmo, source):
#         if self.adjust_requirements(source):
#             pref = 1.0
#             if self.values['alphaz']:
#                 pref *= (
#                     ((1.0 + source.z) / (1.0 + self.values['z_piv'])) **
#                     self.values['alphaz'])
#             if self.values['alphag']:
#                 pref *= ccl.growth_factor(
#                     cosmo, 1.0 / (1.0 + source.z)) ** self.values['alphag']
#
#             source.ia_amplitude[:] = pref*self.values['biasia']
#             source.f_red[:] = self.values['fred']
#             return 0
#         else:
#             print(
#                 f"{self.__class__.__name__} did not find all "
#                 "required source parameters")
#             return 1
