"""DES Y1 cosmic shear likelihood with power spectrum modifier.

This module demonstrates cosmic shear analysis with custom power spectrum
modifications, such as baryonic effects using van Daalen et al. 2019 model.
"""

import os
import pyccl

from firecrown.likelihood.factories import load_sacc_data
from firecrown.likelihood.likelihood import NamedParameters
import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools, PowerspectrumModifier
from firecrown.ccl_factory import CCLFactory
from firecrown.parameters import register_new_updatable_parameter


# pylint: disable-next=invalid-name
class vanDaalen19Baryonfication(PowerspectrumModifier):
    """Power spectrum modifier for van Daalen et al. 2019 baryon model."""

    name: str = "delta_matter_baryons:delta_matter_baryons"

    def __init__(self, pk_to_modify: str = "delta_matter:delta_matter"):
        super().__init__()
        self.pk_to_modify = pk_to_modify
        self.vD19 = pyccl.baryons.BaryonsvanDaalen19()
        self.f_bar = register_new_updatable_parameter(default_value=0.5)

    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        """Compute the 3D power spectrum P(k, z) with baryonic effects."""
        self.vD19.update_parameters(fbar=self.f_bar)
        return self.vD19.include_baryonic_effects(
            cosmo=tools.get_ccl_cosmology(), pk=tools.get_pk(self.pk_to_modify)
        )


def build_likelihood(params: NamedParameters) -> tuple[ConstGaussian, ModelingTools]:
    """Build DES Y1 cosmic shear likelihood with power spectrum modifier.

    Demonstrates custom power spectrum modifications for baryonic effects
    using the van Daalen et al. 2019 model.

    Required parameters:
    - sacc_file: Path to SACC data file

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood and ModelingTools
    """
    sacc_file = os.path.expandvars(params.get_string("sacc_file"))
    sacc_data = load_sacc_data(sacc_file)

    # Create systematic to select modified power spectrum
    baryon_systematic = wl.SelectField(field="delta_matter_baryons")

    # Create weak lensing source with photo-z shift and baryon field
    pzshift = wl.PhotoZShift(sacc_tracer="src0")
    src0 = wl.WeakLensing(sacc_tracer="src0", systematics=[pzshift, baryon_systematic])

    # Create two-point statistics for cosmic shear
    stats = [
        TwoPoint("galaxy_shear_xi_plus", src0, src0),
        TwoPoint("galaxy_shear_xi_minus", src0, src0),
    ]

    # Add power spectrum modifier for baryonic effects
    pk_modifier = vanDaalen19Baryonfication(pk_to_modify="delta_matter:delta_matter")
    modeling_tools = ModelingTools(
        pk_modifiers=[pk_modifier], ccl_factory=CCLFactory(require_nonlinear_pk=True)
    )

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(sacc_data)

    return likelihood, modeling_tools
