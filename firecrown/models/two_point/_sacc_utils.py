"""SACC and CCL utility functions for two-point theory."""


def determine_ccl_kind(sacc_data_type: str) -> str:
    """Determine the CCL kind for this SACC data type.

    :param sacc_data_type: the name of the SACC data type
    :return: the CCL kind
    """
    match sacc_data_type:
        case "galaxy_density_cl" | "galaxy_shearDensity_cl_e" | "galaxy_shear_cl_ee":
            result = "cl"
        case "galaxy_density_xi":
            result = "NN"
        case "galaxy_shearDensity_xi_t":
            result = "NG"
        case "galaxy_shear_xi_minus":
            result = "GG-"
        case "galaxy_shear_xi_plus":
            result = "GG+"
        case "cmbGalaxy_convergenceDensity_xi":
            result = "NN"
        case "cmbGalaxy_convergenceShear_xi_t":
            result = "NG"
        case "cmbGalaxy_convergenceDensity_cl":
            result = "cl"
        case "cmbGalaxy_convergenceShear_cl_e":
            result = "cl"
        case "cmb_convergence_cl":
            result = "cl"
        case _:
            raise ValueError(f"The SACC data type {sacc_data_type} is not supported!")
    return result
